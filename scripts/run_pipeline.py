#!/usr/bin/env python3
"""
Chunk + index step for the PDF OCR RAG pipeline.

The full pipeline is:
  1. OCR + adapter (Ray, vLLM, GPU) → notebook 01-ocr-and-adapter.py or Databricks Job
  2. Chunk + Vector Search index → this script or notebook 02-chunk-index.py

This script runs only step 2. It reads the parsed-docs table produced by 01
(content, parser_status, doc_uri, last_modified), chunks with RecursiveCharacterTextSplitter,
writes the chunked Delta table, enables CDC, and creates/updates the Vector Search index.

Requires a Spark session (Databricks notebook or Spark Connect) and config (parsed_table,
chunked_table, vector index, endpoints). Run step 1 via the GPU notebook/Job first.

Usage:
  python scripts/run_pipeline.py [--config path/to/config.yml]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Repo root = parent of scripts/
REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: Path | None = None) -> dict:
    """Load config from config/config.yml or given path."""
    path = config_path or REPO_ROOT / "config" / "config.yml"
    if not path.exists():
        return {}
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: could not load {path}: {e}", file=sys.stderr)
    return {}


def run_chunk_index(spark, config: dict) -> None:
    """Run chunk + Vector Search index (same logic as 02-chunk-index.py)."""
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
    parser = argparse.ArgumentParser(
        description="Run chunk + index step (parsed-docs → chunked table + Vector Search). OCR step: use 01-ocr-and-adapter.py or Job."
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yml")
    args = parser.parse_args()

    config = load_config(args.config)
    if not config:
        print("No config found; using defaults. Copy config/config.yml.example to config/config.yml.", file=sys.stderr)

    try:
        spark = _get_spark()
    except NameError:
        print("No Spark session found. Run from a Databricks notebook or with Spark Connect.", file=sys.stderr)
        sys.exit(1)

    run_chunk_index(spark, config)
    print("Done.")


def _get_spark():
    """Get Spark session (Databricks notebook or Spark Connect)."""
    try:
        from pyspark.sql import SparkSession
        return SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    except Exception:
        raise NameError("No Spark session available")


if __name__ == "__main__":
    main()
