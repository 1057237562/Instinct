//! Column model shared by the schema pre-scan and the conversion pass.
//!
//! Two properties drive the design.  First, a parquet file has one fixed
//! schema, so the compiler must know the exact shape of every row before it
//! writes the first record batch; the pre-scan pass establishes that.  Second,
//! the Python loaders consume a small, well-understood set of shapes: text
//! corpora are one string column, chat corpora are a list of message structs
//! whose fields are all nullable strings (the loaders `json.loads` the fields
//! that carry structured tool payloads).  Everything else is stored as JSON
//! text so no value is ever lost.

use std::sync::Arc;

use anyhow::{bail, Result};
use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef};
use serde_json::Value;

/// Message fields always written as `utf8`, in this order when present.
///
/// ``tools`` and ``tool_calls`` are JSON payloads in the source corpora and the
/// loaders decode them with ``json.loads`` when they are strings, so keeping
/// them as text preserves both the schema and the loader contract.
pub const CANONICAL_MESSAGE_FIELDS: [&str; 5] = [
    "role",
    "content",
    "reasoning_content",
    "tools",
    "tool_calls",
];

/// Incremental scalar model.  ``Str`` is the universal fallback so that no
/// source value can fail to be represented.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    Bool,
    Int,
    Float,
    Str,
}

impl ScalarType {
    /// Least upper bound of two scalar types.
    pub fn widen(self, other: ScalarType) -> ScalarType {
        if self == other {
            return self;
        }
        match (self, other) {
            (ScalarType::Str, _) | (_, ScalarType::Str) => ScalarType::Str,
            (ScalarType::Float, ScalarType::Int) | (ScalarType::Int, ScalarType::Float) => {
                ScalarType::Float
            }
            // Booleans mixed with numbers are a corpus inconsistency; text is
            // the only representation that keeps both values readable.
            _ => ScalarType::Str,
        }
    }

    pub fn data_type(self) -> DataType {
        match self {
            ScalarType::Bool => DataType::Boolean,
            ScalarType::Int => DataType::Int64,
            ScalarType::Float => DataType::Float64,
            ScalarType::Str => DataType::Utf8,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ScalarType::Bool => "bool",
            ScalarType::Int => "int64",
            ScalarType::Float => "float64",
            ScalarType::Str => "utf8",
        }
    }

    /// Recognise a scalar arrow type; ``None`` for nested and null types.
    pub fn from_data_type(data_type: &DataType) -> Option<ScalarType> {
        Some(match data_type {
            DataType::Boolean => ScalarType::Bool,
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64 => ScalarType::Int,
            DataType::Float16 | DataType::Float32 | DataType::Float64 => ScalarType::Float,
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => ScalarType::Str,
            _ => return None,
        })
    }

    /// ``None`` for null, arrays and objects, which are not scalars.
    pub fn of(value: &Value) -> Option<ScalarType> {
        match value {
            Value::Bool(_) => Some(ScalarType::Bool),
            Value::Number(number) => Some(if number.is_i64() {
                ScalarType::Int
            } else if number.is_u64() {
                // Above ``i64::MAX`` the value still survives as text.
                ScalarType::Str
            } else {
                ScalarType::Float
            }),
            Value::String(_) => Some(ScalarType::Str),
            _ => None,
        }
    }
}

/// Element shape of a list column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElemKind {
    /// Only empty lists seen so far; adopts the element shape that arrives.
    Unknown,
    Scalar(ScalarType),
    /// List of objects: stored as structs with the union of the keys ever
    /// observed, all ``utf8``.  This is the shape of chat messages.
    Object(Vec<String>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColumnKind {
    /// Only null and empty-list values seen so far.
    Unknown,
    Scalar(ScalarType),
    List(ElemKind),
}

impl ColumnKind {
    /// Merge one observation into another; ``true`` reports that the shapes
    /// were irreconcilable and the column fell back to JSON text.
    pub fn widen(&self, other: &ColumnKind) -> (ColumnKind, bool) {
        use ColumnKind::{List, Scalar, Unknown};
        match (self, other) {
            (Unknown, kind) | (kind, Unknown) => (kind.clone(), false),
            (Scalar(left), Scalar(right)) => {
                let widened = left.widen(*right);
                let fell_back = widened == ScalarType::Str
                    && *left != ScalarType::Str
                    && *right != ScalarType::Str;
                (Scalar(widened), fell_back)
            }
            (List(left), List(right)) => match (left, right) {
                (ElemKind::Unknown, kind) | (kind, ElemKind::Unknown) => {
                    (List(kind.clone()), false)
                }
                (ElemKind::Scalar(left), ElemKind::Scalar(right)) => {
                    (List(ElemKind::Scalar(left.widen(*right))), false)
                }
                (ElemKind::Object(left), ElemKind::Object(right)) => {
                    let mut fields = left.clone();
                    for field in right {
                        if !fields.contains(field) {
                            fields.push(field.clone());
                        }
                    }
                    (List(ElemKind::Object(fields)), false)
                }
                _ => (Scalar(ScalarType::Str), true),
            },
            // A key that is sometimes a list and sometimes a scalar cannot be
            // typed structurally; JSON text keeps every value present.
            _ => (Scalar(ScalarType::Str), true),
        }
    }

    pub fn label(&self) -> String {
        match self {
            ColumnKind::Unknown => "null".to_string(),
            ColumnKind::Scalar(scalar) => scalar.label().to_string(),
            ColumnKind::List(ElemKind::Unknown) => "list<null>".to_string(),
            ColumnKind::List(ElemKind::Scalar(scalar)) => format!("list<{}>", scalar.label()),
            ColumnKind::List(ElemKind::Object(fields)) => {
                format!("list<struct<{}>>", fields.join(", "))
            }
        }
    }

    fn scalar_or_text(&self) -> ScalarType {
        match self {
            ColumnKind::Scalar(scalar) => *scalar,
            _ => ScalarType::Str,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Column {
    pub name: String,
    pub kind: ColumnKind,
    /// Rows that carried the key, and rows where it was explicitly null.
    seen: u64,
    nulls: u64,
}

impl Column {
    pub fn new(name: impl Into<String>, kind: ColumnKind) -> Column {
        Column {
            name: name.into(),
            kind,
            seen: 0,
            nulls: 0,
        }
    }

    /// Every column is written nullable: the loaders see the same shapes as
    /// before, and one late null can never invalidate a record batch.
    pub fn arrow_field(&self) -> Field {
        let data_type = match &self.kind {
            ColumnKind::Unknown | ColumnKind::Scalar(_) => self.kind.scalar_or_text().data_type(),
            ColumnKind::List(ElemKind::Unknown) => {
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)))
            }
            ColumnKind::List(ElemKind::Scalar(scalar)) => {
                DataType::List(Arc::new(Field::new("item", scalar.data_type(), true)))
            }
            ColumnKind::List(ElemKind::Object(fields)) => {
                let struct_fields: Fields = fields
                    .iter()
                    .map(|name| Field::new(name, DataType::Utf8, true))
                    .collect();
                DataType::List(Arc::new(Field::new(
                    "item",
                    DataType::Struct(struct_fields),
                    true,
                )))
            }
        };
        Field::new(&self.name, data_type, true)
    }

    pub fn nullable(&self, rows: u64) -> bool {
        self.nulls > 0 || self.seen < rows
    }

    pub fn label(&self) -> String {
        self.kind.label()
    }
}

/// A corpus family.  The preset pins the columns the Python loaders require so
/// they are validated and ordered first; structural inference does the rest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preset {
    Pretrain,
    Sft,
    Dpo,
    Agent,
    Generic,
}

impl Preset {
    pub fn parse(value: &str) -> Result<Preset> {
        match value.to_ascii_lowercase().as_str() {
            "auto" | "generic" => Ok(Preset::Generic),
            "pretrain" | "cpt" => Ok(Preset::Pretrain),
            "sft" | "lora" => Ok(Preset::Sft),
            "dpo" => Ok(Preset::Dpo),
            "agent" | "agentic" => Ok(Preset::Agent),
            other => bail!(
                "unknown format {other:?}; expected auto, pretrain, sft, dpo, agent or generic"
            ),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Preset::Pretrain => "pretrain",
            Preset::Sft => "sft",
            Preset::Dpo => "dpo",
            Preset::Agent => "agent",
            Preset::Generic => "generic",
        }
    }

    /// Columns that must exist, in the order the loaders expect them.
    pub fn required(self, text_key: &str) -> Vec<String> {
        match self {
            Preset::Pretrain => vec![text_key.to_string()],
            Preset::Sft => vec!["conversations".to_string()],
            Preset::Dpo => vec!["chosen".to_string(), "rejected".to_string()],
            Preset::Agent => vec!["conversations".to_string(), "gt".to_string()],
            Preset::Generic => Vec::new(),
        }
    }
}

/// Detect the corpus family from one parsed row; ``--format auto`` uses this so
/// validation and the report name the trainer that will read the file.
pub fn detect_preset(row: &Value, text_key: &str) -> Preset {
    let Some(object) = row.as_object() else {
        return Preset::Generic;
    };
    if object.contains_key(text_key) {
        return Preset::Pretrain;
    }
    if object.contains_key("chosen") && object.contains_key("rejected") {
        return Preset::Dpo;
    }
    if object.contains_key("conversations") {
        return if object.contains_key("gt") {
            Preset::Agent
        } else {
            Preset::Sft
        };
    }
    Preset::Generic
}

/// Ordered, immutable result of the pre-scan.
#[derive(Debug, Clone)]
pub struct SchemaPlan {
    pub columns: Vec<Column>,
    pub rows: u64,
    pub preset: Preset,
    /// Shape conflicts that forced a JSON-text fallback, one line each.
    pub notes: Vec<String>,
}

impl SchemaPlan {
    /// Wrap an already known schema; verification reads one back from a footer.
    pub fn from_columns(columns: Vec<Column>, preset: Preset) -> SchemaPlan {
        SchemaPlan {
            columns,
            rows: 0,
            preset,
            notes: Vec::new(),
        }
    }

    pub fn arrow_schema(&self) -> SchemaRef {
        Arc::new(Schema::new(
            self.columns
                .iter()
                .map(Column::arrow_field)
                .collect::<Vec<_>>(),
        ))
    }

    #[cfg(test)]
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|column| column.name == name)
    }

    pub fn describe(&self) -> Vec<(String, String, bool)> {
        self.columns
            .iter()
            .map(|column| {
                (
                    column.name.clone(),
                    column.label(),
                    column.nullable(self.rows),
                )
            })
            .collect()
    }
}

/// Accumulates observations until the whole file has been seen.
#[derive(Debug, Clone)]
pub struct SchemaBuilder {
    preset: Preset,
    text_key: String,
    columns: Vec<Column>,
    notes: Vec<String>,
    rows: u64,
}

impl SchemaBuilder {
    pub fn new(preset: Preset, text_key: &str) -> SchemaBuilder {
        let mut builder = SchemaBuilder {
            preset,
            text_key: text_key.to_string(),
            columns: Vec::new(),
            notes: Vec::new(),
            rows: 0,
        };
        // Required columns lead the schema so a reader sees the loader's fields
        // first even when a corpus writes them after metadata.
        for name in preset.required(text_key) {
            builder.push_column(&name);
        }
        builder
    }

    fn push_column(&mut self, name: &str) {
        self.columns.push(Column {
            name: name.to_string(),
            kind: ColumnKind::Unknown,
            seen: 0,
            nulls: 0,
        });
    }

    fn column_mut(&mut self, name: &str) -> &mut Column {
        if let Some(index) = self.columns.iter().position(|column| column.name == name) {
            return &mut self.columns[index];
        }
        self.push_column(name);
        self.columns.last_mut().expect("column was just pushed")
    }

    pub fn observe(&mut self, row: &Value) -> Result<()> {
        let object = match row {
            Value::Object(object) => object,
            other => bail!("expected a JSON object per line, found {}", json_kind(other)),
        };
        self.rows += 1;
        for (name, value) in object {
            let observed = observe_kind(value);
            // The column borrow ends before the note is recorded; both touch
            // ``self`` but never at the same time.
            let fell_back = {
                let column = self.column_mut(name);
                column.seen += 1;
                if value.is_null() {
                    column.nulls += 1;
                }
                let (widened, fell_back) = column.kind.widen(&observed);
                column.kind = widened;
                fell_back
            };
            if fell_back {
                self.notes.push(format!(
                    "column {name:?} mixes irreconcilable shapes; storing JSON text"
                ));
            }
        }
        Ok(())
    }

    /// Merge a builder produced from an earlier range of the same file.
    pub fn merge(&mut self, other: &SchemaBuilder) {
        self.rows += other.rows;
        for column in &other.columns {
            let fell_back = {
                let target = self.column_mut(&column.name);
                let (widened, fell_back) = target.kind.widen(&column.kind);
                target.kind = widened;
                target.seen += column.seen;
                target.nulls += column.nulls;
                fell_back
            };
            if fell_back {
                self.notes.push(format!(
                    "column {:?} mixes irreconcilable shapes; storing JSON text",
                    column.name
                ));
            }
        }
        self.notes.extend(other.notes.iter().cloned());
    }

    /// Freeze the observation into a plan, checking the preset's requirements.
    pub fn finish(mut self) -> Result<SchemaPlan> {
        for column in &mut self.columns {
            if let ColumnKind::List(ElemKind::Object(fields)) = &column.kind {
                column.kind = ColumnKind::List(ElemKind::Object(order_message_fields(fields)));
            }
        }
        for name in self.preset.required(&self.text_key) {
            match self.columns.iter().find(|column| column.name == name) {
                Some(column) if !matches!(column.kind, ColumnKind::Unknown) => {}
                Some(_) => bail!(
                    "required column {name:?} is null in every row; compile the file that \
                     holds the data"
                ),
                None => bail!(
                    "required column {name:?} is missing; every row must carry it for \
                     --format {}",
                    self.preset.name()
                ),
            }
        }
        if self.columns.is_empty() {
            bail!("no columns were observed; the source has no rows");
        }
        let mut notes = std::mem::take(&mut self.notes);
        notes.dedup();
        let preset = self.preset;
        Ok(SchemaPlan {
            columns: self.columns,
            rows: self.rows,
            preset,
            notes,
        })
    }
}

/// Canonical chat fields first (in loader order), then any extras as seen.
fn order_message_fields(fields: &[String]) -> Vec<String> {
    let mut ordered: Vec<String> = CANONICAL_MESSAGE_FIELDS
        .iter()
        .filter(|name| fields.iter().any(|field| field == *name))
        .map(|name| name.to_string())
        .collect();
    for field in fields {
        if !ordered.contains(field) {
            ordered.push(field.clone());
        }
    }
    ordered
}

fn observe_kind(value: &Value) -> ColumnKind {
    match value {
        Value::Array(items) => {
            let mut element = ElemKind::Unknown;
            for item in items {
                let observed = match item {
                    Value::Object(object) => {
                        let mut fields = match &element {
                            ElemKind::Object(fields) => fields.clone(),
                            _ => Vec::new(),
                        };
                        for name in object.keys() {
                            if !fields.contains(name) {
                                fields.push(name.clone());
                            }
                        }
                        ElemKind::Object(fields)
                    }
                    other => ElemKind::Scalar(ScalarType::of(other).unwrap_or(ScalarType::Str)),
                };
                let merged = match (&element, &observed) {
                    (ElemKind::Unknown, kind) | (kind, ElemKind::Unknown) => kind.clone(),
                    (ElemKind::Scalar(left), ElemKind::Scalar(right)) => {
                        ElemKind::Scalar(left.widen(*right))
                    }
                    (ElemKind::Object(left), ElemKind::Object(right)) => {
                        let mut fields = left.clone();
                        for name in right {
                            if !fields.contains(name) {
                                fields.push(name.clone());
                            }
                        }
                        ElemKind::Object(fields)
                    }
                    // A mixed list degrades to text elements so the column type
                    // stays stable across rows.
                    _ => ElemKind::Scalar(ScalarType::Str),
                };
                element = merged;
            }
            ColumnKind::List(element)
        }
        // Objects outside a list are JSON payloads (``tools``, ``gt`` maps);
        // the loaders expect them as text.
        Value::Object(_) => ColumnKind::Scalar(ScalarType::Str),
        other => match ScalarType::of(other) {
            Some(scalar) => ColumnKind::Scalar(scalar),
            None => ColumnKind::Unknown,
        },
    }
}

pub fn json_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn plan_of(rows: &[Value], preset: Preset) -> SchemaPlan {
        let mut builder = SchemaBuilder::new(preset, "text");
        for row in rows {
            builder.observe(row).unwrap();
        }
        builder.finish().unwrap()
    }

    #[test]
    fn absent_keys_become_nullable_columns() {
        let plan = plan_of(
            &[json!({"text": "a", "token_count": 3}), json!({"text": "b"})],
            Preset::Pretrain,
        );
        let index = plan.column_index("token_count").unwrap();
        assert!(plan.columns[index].nullable(plan.rows));
        assert_eq!(plan.columns[index].kind, ColumnKind::Scalar(ScalarType::Int));
    }

    #[test]
    fn mixed_numbers_widen_without_text_fallback() {
        assert_eq!(ScalarType::Int.widen(ScalarType::Float), ScalarType::Float);
        assert_eq!(ScalarType::Int.widen(ScalarType::Bool), ScalarType::Str);
    }

    #[test]
    fn message_fields_are_ordered_canonically() {
        let plan = plan_of(
            &[json!({"conversations": [
                {"content": "hi", "role": "user", "custom": "x"},
                {"role": "assistant", "content": "yo", "tools": {"a": 1}},
            ]})],
            Preset::Sft,
        );
        let ColumnKind::List(ElemKind::Object(fields)) = &plan.columns[0].kind else {
            panic!("expected a message list, got {}", plan.columns[0].label());
        };
        assert_eq!(
            fields,
            &vec![
                "role".to_string(),
                "content".to_string(),
                "tools".to_string(),
                "custom".to_string()
            ]
        );
    }

    #[test]
    fn empty_message_lists_do_not_decide_the_element_type() {
        let plan = plan_of(
            &[
                json!({"conversations": []}),
                json!({"conversations": [{"role": "user", "content": "hi"}]}),
            ],
            Preset::Sft,
        );
        let ColumnKind::List(ElemKind::Object(fields)) = &plan.columns[0].kind else {
            panic!("expected a message list, got {}", plan.columns[0].label());
        };
        assert_eq!(fields, &vec!["role".to_string(), "content".to_string()]);
    }

    #[test]
    fn list_of_scalars_stays_typed() {
        let plan = plan_of(
            &[json!({"conversations": [], "gt": ["1", "2"]})],
            Preset::Agent,
        );
        let gt = plan.column_index("gt").unwrap();
        assert_eq!(
            plan.columns[gt].kind,
            ColumnKind::List(ElemKind::Scalar(ScalarType::Str))
        );
    }

    #[test]
    fn mixed_list_and_scalar_falls_back_to_text() {
        let plan = plan_of(
            &[json!({"text": "a", "meta": [1, 2]}), json!({"text": "b", "meta": "x"})],
            Preset::Pretrain,
        );
        let index = plan.column_index("meta").unwrap();
        assert_eq!(plan.columns[index].kind, ColumnKind::Scalar(ScalarType::Str));
        assert!(plan.notes.iter().any(|note| note.contains("\"meta\"")));
    }

    #[test]
    fn missing_required_column_is_rejected() {
        let mut builder = SchemaBuilder::new(Preset::Pretrain, "text");
        builder.observe(&json!({"body": "oops"})).unwrap();
        assert!(builder.finish().is_err());
    }

    #[test]
    fn detect_preset_names_the_trainer_family() {
        assert_eq!(
            detect_preset(&json!({"text": "x"}), "text"),
            Preset::Pretrain
        );
        assert_eq!(
            detect_preset(&json!({"conversations": []}), "text"),
            Preset::Sft
        );
        assert_eq!(
            detect_preset(&json!({"conversations": [], "gt": []}), "text"),
            Preset::Agent
        );
        assert_eq!(
            detect_preset(&json!({"chosen": [], "rejected": []}), "text"),
            Preset::Dpo
        );
    }
}
