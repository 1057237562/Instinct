//! Reading Parquet back: the inspect path and the verification oracle.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use anyhow::{bail, Context, Result};
use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef};
use arrow_schema::{DataType, SchemaRef};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::metadata::ParquetMetaData;
use serde::Serialize;
use serde_json::{Map, Value};

use crate::report::normalize;
use crate::schema::{Column, ColumnKind, ElemKind, Preset, ScalarType, SchemaPlan};

#[derive(Debug, Clone, Serialize)]
pub struct ColumnChunkSummary {
    pub rows: i64,
    pub bytes: i64,
    pub uncompressed_bytes: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ParquetSummary {
    pub path: String,
    pub bytes: u64,
    pub rows: i64,
    pub row_groups: usize,
    pub row_group_rows: Vec<i64>,
    pub columns: Vec<ColumnChunkSummary>,
    pub created_by: Option<String>,
    pub arrow_schema: String,
}

/// Footer-only summary: no data pages are read.
pub fn summarize(path: &Path) -> Result<ParquetSummary> {
    let file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let bytes = file.metadata().map(|meta| meta.len()).unwrap_or(0);
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("{} is not a readable parquet file", path.display()))?;
    let metadata: &ParquetMetaData = &builder.metadata();
    let mut row_group_rows = Vec::with_capacity(metadata.num_row_groups());
    let mut columns = Vec::with_capacity(metadata.num_row_groups());
    for index in 0..metadata.num_row_groups() {
        let group = metadata.row_group(index);
        row_group_rows.push(group.num_rows());
        let mut rows = group.num_rows();
        let mut compressed = 0i64;
        let mut uncompressed = 0i64;
        for column in 0..group.num_columns() {
            let chunk = group.column(column);
            rows = chunk.num_values().max(rows);
            compressed += chunk.compressed_size();
            uncompressed += chunk.uncompressed_size();
        }
        columns.push(ColumnChunkSummary {
            rows,
            bytes: compressed,
            uncompressed_bytes: uncompressed,
        });
    }
    Ok(ParquetSummary {
        path: normalize(path),
        bytes,
        rows: metadata.file_metadata().num_rows(),
        row_groups: metadata.num_row_groups(),
        row_group_rows,
        columns,
        created_by: metadata.file_metadata().created_by().map(str::to_string),
        arrow_schema: format!("{:#}", builder.schema()),
    })
}

/// Read up to ``limit`` rows starting at ``offset`` as JSON values.
pub fn read_values(
    path: &Path,
    offset: u64,
    limit: Option<u64>,
) -> Result<(SchemaRef, Vec<Value>)> {
    let file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("{} is not a readable parquet file", path.display()))?;
    let schema = builder.schema().clone();
    let mut builder = builder.with_batch_size(1024);
    if offset > 0 {
        builder = builder.with_offset(offset as usize);
    }
    if let Some(limit) = limit {
        builder = builder.with_limit(limit as usize);
    }
    let reader = builder.build().context("cannot start a parquet reader")?;
    let mut rows = Vec::new();
    for batch in reader {
        let batch = batch.context("cannot read a parquet record batch")?;
        for index in 0..batch.num_rows() {
            rows.push(row_to_value(&batch, index));
        }
    }
    Ok((schema, rows))
}

/// Map a compiled parquet schema back onto the compiler's column model, which
/// also checks that every column has a shape the Python loaders can consume.
pub fn plan_from_arrow(schema: &SchemaRef, preset: Preset) -> Result<SchemaPlan> {
    let columns = schema
        .fields()
        .iter()
        .map(|field| {
            Ok(Column::new(
                field.name().clone(),
                column_kind(field.data_type(), field.name())?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(SchemaPlan::from_columns(columns, preset))
}

fn column_kind(data_type: &DataType, name: &str) -> Result<ColumnKind> {
    Ok(match data_type {
        DataType::Null => ColumnKind::Unknown,
        DataType::List(child) | DataType::LargeList(child) => match child.data_type() {
            DataType::Struct(fields) => ColumnKind::List(ElemKind::Object(
                fields.iter().map(|field| field.name().clone()).collect(),
            )),
            other => ColumnKind::List(ElemKind::Scalar(
                ScalarType::from_data_type(other).ok_or_else(|| {
                    anyhow::anyhow!("column {name:?} is a list of an unsupported arrow type")
                })?,
            )),
        },
        other => match ScalarType::from_data_type(other) {
            Some(scalar) => ColumnKind::Scalar(scalar),
            None => bail!(
                "column {name:?} has arrow type {other}, which the Instinct loaders cannot read"
            ),
        },
    })
}

pub fn row_to_value(batch: &arrow_array::RecordBatch, row: usize) -> Value {
    let mut object = Map::new();
    for (index, field) in batch.schema().fields().iter().enumerate() {
        object.insert(field.name().clone(), value_at(batch.column(index), row));
    }
    Value::Object(object)
}

fn value_at(array: &ArrayRef, row: usize) -> Value {
    if array.is_null(row) {
        return Value::Null;
    }
    match array.data_type() {
        DataType::Utf8 => Value::String(array.as_string::<i32>().value(row).to_string()),
        DataType::LargeUtf8 => Value::String(array.as_string::<i64>().value(row).to_string()),
        DataType::Int64 => Value::Number(array.as_primitive::<Int64Type>().value(row).into()),
        DataType::Int32 => Value::Number(array.as_primitive::<Int32Type>().value(row).into()),
        DataType::Float64 => match serde_json::Number::from_f64(
            array.as_primitive::<Float64Type>().value(row),
        ) {
            Some(number) => Value::Number(number),
            None => Value::String("nan".to_string()),
        },
        DataType::Boolean => Value::Bool(array.as_boolean().value(row)),
        DataType::List(_) => {
            let list = array.as_list::<i32>();
            let values = list.value(row);
            Value::Array((0..values.len()).map(|index| value_at(&values, index)).collect())
        }
        DataType::Struct(fields) => {
            let structure = array.as_struct();
            let mut object = Map::new();
            for index in 0..fields.len() {
                object.insert(
                    fields[index].name().clone(),
                    value_at(structure.column(index), row),
                );
            }
            Value::Object(object)
        }
        other => Value::String(format!("<unsupported arrow type {other}>")),
    }
}

/// Drop null-valued keys so an omitted key and an explicit null compare equal.
pub fn canonical(value: &Value) -> Value {
    match value {
        Value::Object(object) => Value::Object(
            object
                .iter()
                .filter(|(_, value)| !value.is_null())
                .map(|(key, value)| (key.clone(), canonical(value)))
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.iter().map(canonical).collect()),
        other => other.clone(),
    }
}

/// Footer key-value metadata, used to read back the alignment contract.
pub fn footer_metadata(path: &Path) -> Result<BTreeMap<String, String>> {
    let file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("{} is not a readable parquet file", path.display()))?;
    let mut values = BTreeMap::new();
    if let Some(pairs) = builder.metadata().file_metadata().key_value_metadata() {
        for pair in pairs {
            if let Some(value) = &pair.value {
                values.insert(pair.key.clone(), value.clone());
            }
        }
    }
    Ok(values)
}

/// Reject a table whose shape no loader can consume for this preset.
pub fn require_columns(schema: &SchemaRef, preset: Preset, text_key: &str) -> Result<()> {
    for name in preset.required(text_key) {
        if schema.field_with_name(&name).is_err() {
            bail!(
                "parquet file has no {name:?} column, which --format {} requires",
                preset.name()
            );
        }
    }
    Ok(())
}
