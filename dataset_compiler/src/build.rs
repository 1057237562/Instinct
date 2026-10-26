//! Conversion from parsed JSON rows into Arrow record batches.
//!
//! Cells are produced per row by the parse pool and appended column-major, so
//! each column downcasts its builder once per batch instead of once per value.

use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use arrow_array::builder::{
    ArrayBuilder, BooleanBuilder, Float64Builder, Int64Builder, ListBuilder, StringBuilder,
    StructBuilder,
};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Fields};
use serde_json::Value;

use crate::schema::{Column, ColumnKind, ElemKind, ScalarType, SchemaPlan};

/// One schema-aligned value.  ``List`` holds either scalars or messages; which
/// one is decided by the column kind held by the plan.
#[derive(Debug, Clone, PartialEq)]
pub enum Cell {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<Cell>),
    /// Sparse message: unlisted fields are null.  Indexes refer to the field
    /// order of the column's struct.
    Msg(Vec<(usize, Cell)>),
}

/// Map one source row onto the compiled schema.  Values that do not fit the
/// column's type are dropped to null rather than mis-encoded; the pre-scan
/// guarantees this only happens for values the schema never saw.
pub fn convert_row(row: &Value, plan: &SchemaPlan) -> Result<Vec<Cell>> {
    let object = row
        .as_object()
        .ok_or_else(|| anyhow!("expected a JSON object per line"))?;
    Ok(plan
        .columns
        .iter()
        .map(|column| convert_column(object.get(&column.name), column))
        .collect())
}

fn convert_column(value: Option<&Value>, column: &Column) -> Cell {
    let value = match value {
        None | Some(Value::Null) => return Cell::Null,
        Some(value) => value,
    };
    match &column.kind {
        ColumnKind::Unknown => Cell::Null,
        ColumnKind::Scalar(scalar) => convert_scalar(value, *scalar),
        ColumnKind::List(ElemKind::Unknown) => Cell::List(Vec::new()),
        ColumnKind::List(ElemKind::Scalar(scalar)) => match value.as_array() {
            Some(items) => Cell::List(items.iter().map(|item| convert_scalar(item, *scalar)).collect()),
            None => Cell::Null,
        },
        ColumnKind::List(ElemKind::Object(fields)) => match value.as_array() {
            Some(items) => Cell::List(
                items
                    .iter()
                    .map(|item| match item.as_object() {
                        Some(object) => Cell::Msg(
                            object
                                .iter()
                                .filter_map(|(name, item)| {
                                    fields
                                        .iter()
                                        .position(|field| field == name)
                                        .map(|index| (index, Cell::Str(json_text(item))))
                                })
                                .collect(),
                        ),
                        // Mixed lists degrade to text during the pre-scan, so a
                        // message slot without an object becomes an empty one.
                        None => Cell::Msg(Vec::new()),
                    })
                    .collect(),
            ),
            None => Cell::Null,
        },
    }
}

fn convert_scalar(value: &Value, scalar: ScalarType) -> Cell {
    match scalar {
        ScalarType::Str => Cell::Str(json_text(value)),
        ScalarType::Int => match value.as_i64() {
            Some(number) => Cell::Int(number),
            None => Cell::Null,
        },
        ScalarType::Float => match value.as_f64() {
            Some(number) => Cell::Float(number),
            None => Cell::Null,
        },
        ScalarType::Bool => match value.as_bool() {
            Some(flag) => Cell::Bool(flag),
            None => Cell::Null,
        },
    }
}

/// Strings stay verbatim; every other JSON value becomes its compact text so a
/// widened column still carries the original data.
fn json_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

/// Reverse of [`convert_row`], used by verification and tooling.  Message
/// fields that were absent stay absent, so a later canonicalization can treat
/// "omitted" and "explicit null" as the same thing.
pub fn cell_to_value(cells: &[Cell], plan: &SchemaPlan) -> Value {
    let mut object = serde_json::Map::new();
    for (index, column) in plan.columns.iter().enumerate() {
        let cell = cells.get(index).unwrap_or(&Cell::Null);
        object.insert(column.name.clone(), cell_value(cell, &column.kind));
    }
    Value::Object(object)
}

fn cell_value(cell: &Cell, kind: &ColumnKind) -> Value {
    match (cell, kind) {
        (Cell::List(items), ColumnKind::List(ElemKind::Object(fields))) => Value::Array(
            items
                .iter()
                .map(|item| match item {
                    Cell::Msg(entries) => {
                        let mut object = serde_json::Map::new();
                        for (field_index, value) in entries {
                            if let Some(name) = fields.get(*field_index) {
                                object.insert(name.clone(), scalar_value(value));
                            }
                        }
                        Value::Object(object)
                    }
                    other => scalar_value(other),
                })
                .collect(),
        ),
        (other, _) => scalar_value(other),
    }
}

fn scalar_value(cell: &Cell) -> Value {
    match cell {
        Cell::Null => Value::Null,
        Cell::Str(text) => Value::String(text.clone()),
        Cell::Int(number) => Value::Number((*number).into()),
        Cell::Float(number) => serde_json::Number::from_f64(*number)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        Cell::Bool(flag) => Value::Bool(*flag),
        Cell::List(items) => Value::Array(items.iter().map(scalar_value).collect()),
        Cell::Msg(_) => Value::Null,
    }
}

pub fn build_batch(plan: &SchemaPlan, rows: &[Vec<Cell>]) -> Result<RecordBatch> {
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(plan.columns.len());
    for (index, column) in plan.columns.iter().enumerate() {
        arrays.push(build_column(column, index, rows)?);
    }
    RecordBatch::try_new(plan.arrow_schema(), arrays)
        .context("compiled row does not match the inferred schema")
}

fn build_column(column: &Column, index: usize, rows: &[Vec<Cell>]) -> Result<ArrayRef> {
    Ok(match &column.kind {
        ColumnKind::Unknown => build_scalar(ScalarType::Str, index, rows),
        ColumnKind::Scalar(scalar) => build_scalar(*scalar, index, rows),
        ColumnKind::List(ElemKind::Unknown) => {
            list_of(rows, index, StringBuilder::new(), |builder, cell| match cell {
                Cell::Str(text) => builder.append_value(text),
                _ => builder.append_null(),
            })
        }
        ColumnKind::List(ElemKind::Scalar(ScalarType::Str)) => {
            list_of(rows, index, StringBuilder::new(), |builder, cell| match cell {
                Cell::Str(text) => builder.append_value(text),
                _ => builder.append_null(),
            })
        }
        ColumnKind::List(ElemKind::Scalar(ScalarType::Int)) => {
            list_of(rows, index, Int64Builder::new(), |builder, cell| match cell {
                Cell::Int(number) => builder.append_value(*number),
                _ => builder.append_null(),
            })
        }
        ColumnKind::List(ElemKind::Scalar(ScalarType::Float)) => {
            list_of(rows, index, Float64Builder::new(), |builder, cell| match cell {
                Cell::Float(number) => builder.append_value(*number),
                _ => builder.append_null(),
            })
        }
        ColumnKind::List(ElemKind::Scalar(ScalarType::Bool)) => {
            list_of(rows, index, BooleanBuilder::new(), |builder, cell| match cell {
                Cell::Bool(flag) => builder.append_value(*flag),
                _ => builder.append_null(),
            })
        }
        ColumnKind::List(ElemKind::Object(fields)) => build_messages(fields, index, rows)?,
    })
}

fn build_scalar(scalar: ScalarType, index: usize, rows: &[Vec<Cell>]) -> ArrayRef {
    match scalar {
        ScalarType::Str => {
            let mut builder = StringBuilder::new();
            for row in rows {
                match &row[index] {
                    Cell::Str(text) => builder.append_value(text),
                    _ => builder.append_null(),
                }
            }
            Arc::new(builder.finish())
        }
        ScalarType::Int => {
            let mut builder = Int64Builder::new();
            for row in rows {
                match &row[index] {
                    Cell::Int(number) => builder.append_value(*number),
                    _ => builder.append_null(),
                }
            }
            Arc::new(builder.finish())
        }
        ScalarType::Float => {
            let mut builder = Float64Builder::new();
            for row in rows {
                match &row[index] {
                    Cell::Float(number) => builder.append_value(*number),
                    _ => builder.append_null(),
                }
            }
            Arc::new(builder.finish())
        }
        ScalarType::Bool => {
            let mut builder = BooleanBuilder::new();
            for row in rows {
                match &row[index] {
                    Cell::Bool(flag) => builder.append_value(*flag),
                    _ => builder.append_null(),
                }
            }
            Arc::new(builder.finish())
        }
    }
}

fn list_of<T: ArrayBuilder>(
    rows: &[Vec<Cell>],
    index: usize,
    builder: T,
    mut push: impl FnMut(&mut T, &Cell),
) -> ArrayRef {
    let mut list = ListBuilder::new(builder);
    for row in rows {
        match &row[index] {
            Cell::List(items) => {
                for item in items {
                    push(list.values(), item);
                }
                list.append(true);
            }
            _ => list.append(false),
        }
    }
    Arc::new(list.finish())
}

/// Build ``list<struct<utf8...>>``: the shape every chat corpus shares.  Each
/// struct slot needs exactly one value per field before the slot is appended.
fn build_messages(fields: &[String], index: usize, rows: &[Vec<Cell>]) -> Result<ArrayRef> {
    let struct_fields: Fields = fields
        .iter()
        .map(|name| Field::new(name, DataType::Utf8, true))
        .collect();
    let field_builders: Vec<Box<dyn ArrayBuilder>> = fields
        .iter()
        .map(|_| Box::new(StringBuilder::new()) as Box<dyn ArrayBuilder>)
        .collect();
    let mut list = ListBuilder::new(StructBuilder::new(struct_fields, field_builders));
    let mut values: Vec<Option<&Cell>> = vec![None; fields.len()];
    for row in rows {
        let Cell::List(items) = &row[index] else {
            list.append(false);
            continue;
        };
        for item in items {
            values.iter_mut().for_each(|slot| *slot = None);
            if let Cell::Msg(entries) = item {
                for (field_index, value) in entries {
                    if let Some(slot) = values.get_mut(*field_index) {
                        *slot = Some(value);
                    }
                }
            }
            let messages = list.values();
            for (field_index, slot) in values.iter().enumerate() {
                let builder = messages
                    .field_builder::<StringBuilder>(field_index)
                    .ok_or_else(|| anyhow!("message field {field_index} has an unexpected type"))?;
                match slot {
                    Some(Cell::Str(text)) => builder.append_value(text),
                    _ => builder.append_null(),
                }
            }
            messages.append(true);
        }
        list.append(true);
    }
    Ok(Arc::new(list.finish()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{Preset, SchemaBuilder};
    use arrow_array::cast::AsArray;
    use arrow_array::Array;
    use serde_json::json;

    fn plan_for(rows: &[Value], preset: Preset) -> SchemaPlan {
        let mut builder = SchemaBuilder::new(preset, "text");
        for row in rows {
            builder.observe(row).unwrap();
        }
        builder.finish().unwrap()
    }

    /// Position of a message field in the compiled ``list<struct<...>>``.
    fn message_field(batch: &arrow_array::RecordBatch, name: &str) -> usize {
        let DataType::List(child) = batch.column(0).data_type() else {
            panic!("expected a list column");
        };
        let DataType::Struct(fields) = child.data_type() else {
            panic!("expected a struct element");
        };
        fields
            .iter()
            .position(|field| field.name() == name)
            .unwrap_or_else(|| panic!("no {name:?} message field"))
    }

    #[test]
    fn text_corpus_round_trips_through_arrow() {
        let rows = vec![json!({"text": "hello", "token_count": 2}), json!({"text": "world"})];
        let plan = plan_for(&rows, Preset::Pretrain);
        let cells: Vec<Vec<Cell>> = rows
            .iter()
            .map(|row| convert_row(row, &plan).unwrap())
            .collect();
        let batch = build_batch(&plan, &cells).unwrap();
        assert_eq!(batch.num_rows(), 2);
        let text = batch.column(0).as_string::<i32>();
        assert_eq!(text.value(0), "hello");
        assert_eq!(text.value(1), "world");
        let counts = batch.column(1).as_primitive::<arrow_array::types::Int64Type>();
        assert_eq!(counts.value(0), 2);
        assert!(counts.is_null(1));
    }

    #[test]
    fn chat_corpus_keeps_messages_and_tool_payloads() {
        let rows = vec![
            json!({"conversations": [
                {"role": "system", "tools": [{"name": "search"}]},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo", "reasoning_content": "think"},
            ]}),
            json!({"conversations": []}),
        ];
        let plan = plan_for(&rows, Preset::Sft);
        let cells: Vec<Vec<Cell>> = rows
            .iter()
            .map(|row| convert_row(row, &plan).unwrap())
            .collect();
        let batch = build_batch(&plan, &cells).unwrap();
        let conversations = batch.column(0).as_list::<i32>();
        assert_eq!(conversations.len(), 2);
        let first = conversations.value(0);
        let messages = first.as_struct();
        assert_eq!(messages.len(), 3);
        let roles = messages.column(0).as_string::<i32>();
        assert_eq!(roles.value(0), "system");
        assert_eq!(roles.value(2), "assistant");
        let tools = messages
            .column(message_field(&batch, "tools"))
            .as_string::<i32>();
        assert_eq!(tools.value(0), "[{\"name\":\"search\"}]");
        assert!(tools.is_null(1));
        assert_eq!(conversations.value(1).len(), 0);
    }

    #[test]
    fn scalar_lists_stay_lists() {
        let rows = vec![
            json!({"conversations": [{"role": "user", "content": "q"}], "gt": ["1", "2"]}),
            json!({"conversations": [{"role": "user", "content": "q2"}], "gt": []}),
        ];
        let plan = plan_for(&rows, Preset::Agent);
        let cells: Vec<Vec<Cell>> = rows
            .iter()
            .map(|row| convert_row(row, &plan).unwrap())
            .collect();
        let batch = build_batch(&plan, &cells).unwrap();
        let gt = batch.column(1).as_list::<i32>();
        assert_eq!(gt.value(0).as_string::<i32>().len(), 2);
        assert_eq!(gt.value(1).as_string::<i32>().len(), 0);
    }
}
