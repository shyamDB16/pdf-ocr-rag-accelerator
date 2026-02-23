#!/usr/bin/env python3
"""
PDF RAG pipeline entrypoint for external or job-based runs.

Loads config from config/config.yml (or env) and runs:
  1. Ingest & parse PDFs -> Delta parsed_table
  2. Chunk -> Delta chunked_table, then create/update Vector Search index

Requires a Spark session (e.g. Databricks notebook, or Spark Connect to a
Databricks cluster). For Databricks, prefer running the notebooks directly
or as workflow tasks; use this script when driving from CI or a local driver
with Spark Connect.

Usage:
  # From repo root, with config/config.yml present:
  python scripts/run_pipeline.py [--ingest-only | --chunk-only]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Repo root = parent of scripts/
REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config():
    """Load config from config/config.yml if present."""
    config_path = REPO_ROOT / "config" / "config.yml"
    if not config_path.exists():
        return None
    try:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: could not load {config_path}: {e}", file=sys.stderr)
    return None


def run_ingest_parse(spark, config: dict) -> None:
    """Run ingest + parse step (notebook 01 logic)."""
    catalog = config.get("catalog", "main")
    schema = config.get("schema", "default")
    volume_name = config.get("volume_name", "pdf_docs")
    parsed_table = config.get("parsed_table") or f"{catalog}.{schema}.rag_parsed_docs"
    volume_path = f"/Volumes/{catalog}/{schema}/{volume_name}"

    import fitz
    import pymupdf4llm
    from urllib.parse import urlparse
    from pyspark.sql.types import StructType, StructField, StringType, TimestampType
    import pyspark.sql.functions as F

    def parse_pdf(content: bytes, path: str, modification_time, doc_length: int) -> tuple:
        try:
            doc = fitz.Document(stream=content, filetype="pdf")
            md_text = pymupdf4llm.to_markdown(doc)
            doc_uri = urlparse(path).path or path
            return (md_text.strip(), "SUCCESS", doc_uri, modification_time)
        except Exception as e:
            return ("", f"ERROR: {e}", path, modification_time)

    schema_out = StructType([
        StructField("content", StringType()),
        StructField("parser_status", StringType()),
        StructField("doc_uri", StringType()),
        StructField("last_modified", TimestampType()),
    ])
    parse_udf = F.udf(parse_pdf, schema_out)

    raw_df = (
        spark.read.format("binaryFile")
        .option("recursiveFileLookup", "true")
        .load(volume_path)
    )
    raw_df = raw_df.filter(F.col("path").rlike(r"\.pdf$"))
    if raw_df.count() == 0:
        raise ValueError(f"No PDF files found under {volume_path}")

    parsed_df = raw_df.withColumn(
        "parsed",
        parse_udf("content", "path", "modificationTime", "length")
    ).select(
        F.col("parsed.content").alias("content"),
        F.col("parsed.parser_status").alias("parser_status"),
        F.col("parsed.doc_uri").alias("doc_uri"),
        F.col("parsed.last_modified").alias("last_modified"),
    )
    parsed_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(parsed_table)
    print(f"Wrote {parsed_df.count()} rows to {parsed_table}")


def run_chunk_index(spark, config: dict) -> None:
    """Run chunk + Vector Search index step (notebook 02 logic)."""
    parsed_table = config.get("parsed_table", "main.default.rag_parsed_docs")
    chunked_table = config.get("chunked_table", "main.default.rag_docs_chunked")
    vector_index_name = config.get("vector_index_name", "main.default.rag_docs_chunked_index")
    vector_search_endpoint = config.get("vector_search_endpoint", "one-env-shared-endpoint")
    embedding_endpoint = config.get("embedding_endpoint", "databricks-bge-m3")
    chunk_size = config.get("chunk_size", 500)
    chunk_overlap = config.get("chunk_overlap", 50)

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from pyspark.sql import functions as F
    from pyspark.sql.types import ArrayType, StringType

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    def chunk_text(doc: str) -> list:
        if not doc or not doc.strip():
            return []
        return splitter.split_text(doc)

    chunk_udf = F.udf(chunk_text, ArrayType(StringType()), useArrow=True)
    parsed_df = spark.table(parsed_table)
    propagate_columns = ["doc_uri", "parser_status", "last_modified"]
    chunked_df = (
        parsed_df.withColumn("content_chunked", chunk_udf(F.col("content")))
        .select(*propagate_columns, F.explode("content_chunked").alias("content_chunked"))
        .withColumn("chunk_id", F.md5(F.col("content_chunked")))
        .select("chunk_id", "content_chunked", *propagate_columns)
    )
    chunked_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(chunked_table)
    spark.sql(f"ALTER TABLE {chunked_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

    from databricks.sdk.service.vectorsearch import (
        DeltaSyncVectorIndexSpecRequest,
        EmbeddingSourceColumn,
        PipelineType,
        VectorIndexType,
    )
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.errors.platform import ResourceDoesNotExist, BadRequest
    import time

    w = WorkspaceClient()
    vsc = w.vector_search_indexes

    def find_index(name):
        try:
            return vsc.get_index(index_name=name)
        except ResourceDoesNotExist:
            return None

    existing = find_index(vector_index_name)
    if existing:
        while not getattr(existing, "status", None) or not getattr(existing.status, "ready", True):
            print("Index not ready, waiting 30s...")
            time.sleep(30)
            existing = find_index(vector_index_name)
        try:
            vsc.sync_index(index_name=vector_index_name)
            print(f"Sync started for {vector_index_name}")
        except BadRequest as e:
            print(f"Sync in progress or error: {e}")
    else:
        delta_sync_spec = DeltaSyncVectorIndexSpecRequest(
            source_table=chunked_table,
            pipeline_type=PipelineType.TRIGGERED,
            embedding_source_columns=[
                EmbeddingSourceColumn(
                    name="content_chunked",
                    embedding_model_endpoint_name=embedding_endpoint,
                )
            ],
        )
        vsc.create_index(
            name=vector_index_name,
            endpoint_name=vector_search_endpoint,
            primary_key="chunk_id",
            index_type=VectorIndexType.DELTA_SYNC,
            delta_sync_index_spec=delta_sync_spec,
        )
        print(f"Created index {vector_index_name}")
    print(f"Chunked table: {chunked_table}, count: {spark.table(chunked_table).count()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PDF RAG pipeline (ingest+parse, chunk+index)")
    parser.add_argument("--ingest-only", action="store_true", help="Only run ingest & parse")
    parser.add_argument("--chunk-only", action="store_true", help="Only run chunk & index")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yml")
    args = parser.parse_args()

    config_path = args.config or (REPO_ROOT / "config" / "config.yml")
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = load_config() or {}
        if not config:
            print("No config found; using defaults. Copy config/config.yml.example to config/config.yml.", file=sys.stderr)

    try:
        spark = __get_spark()
    except NameError:
        print("No Spark session found. Run this script from a Databricks notebook or with Spark Connect.", file=sys.stderr)
        sys.exit(1)

    if args.chunk_only:
        run_chunk_index(spark, config)
    elif args.ingest_only:
        run_ingest_parse(spark, config)
    else:
        run_ingest_parse(spark, config)
        run_chunk_index(spark, config)
    print("Done.")


def __get_spark():
    """Get Spark session (Databricks notebook or Spark Connect)."""
    try:
        from pyspark.sql import SparkSession
        return SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    except Exception:
        raise NameError("No Spark session available")


if __name__ == "__main__":
    main()
