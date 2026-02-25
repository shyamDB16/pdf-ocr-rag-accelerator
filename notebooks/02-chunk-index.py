# Databricks notebook source
# MAGIC %md
# MAGIC # 2. Chunk and index
# MAGIC
# MAGIC Reads the parsed-docs Delta table from **01** (OCR+adapter), chunks each document with a character-based splitter, writes a chunked Delta table (with CDC enabled), and creates a Vector Search index using Delta Sync.
# MAGIC
# MAGIC **Prerequisite:** Run `01-ocr-and-adapter.py`, then set catalog/schema/parsed_table_suffix below to match 01's output table.
# MAGIC
# MAGIC **Output:** A chunked Delta table (`chunk_id`, `content_chunked`, `doc_uri`, …) and a Vector Search index on the same table.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config
# MAGIC
# MAGIC Use the same catalog/schema as 01. Set the parsed table name to the output of 01. Configure chunked table name, vector search endpoint, and embedding model endpoint.

# COMMAND ----------

dbutils.widgets.text("catalog", "main", "UC Catalog")
dbutils.widgets.text("schema", "default", "UC Schema")
dbutils.widgets.text("parsed_table_suffix", "rag_parsed_docs", "Parsed table name (from 01)")
dbutils.widgets.text("vector_search_endpoint", "one-env-shared-endpoint", "Vector Search Endpoint")
dbutils.widgets.text("embedding_endpoint", "databricks-bge-m3", "Embedding Endpoint")
dbutils.widgets.text("chunk_size", "500", "Chunk Size")
dbutils.widgets.text("chunk_overlap", "50", "Chunk Overlap")

# COMMAND ----------

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
parsed_suffix = dbutils.widgets.get("parsed_table_suffix").strip() or "rag_parsed_docs"

PARSED_TABLE = f"{CATALOG}.{SCHEMA}.{parsed_suffix}"
CHUNKED_TABLE = f"{CATALOG}.{SCHEMA}.rag_docs_chunked"
VECTOR_INDEX_NAME = f"{CATALOG}.{SCHEMA}.rag_docs_chunked_index"

VECTOR_SEARCH_ENDPOINT = dbutils.widgets.get("vector_search_endpoint")
EMBEDDING_ENDPOINT = dbutils.widgets.get("embedding_endpoint")
CHUNK_SIZE = int(dbutils.widgets.get("chunk_size"))
CHUNK_OVERLAP = int(dbutils.widgets.get("chunk_overlap"))

print(f"Parsed table: {PARSED_TABLE}")
print(f"Chunked table: {CHUNKED_TABLE}")
print(f"Vector index: {VECTOR_INDEX_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install chunking dependency
# MAGIC
# MAGIC We use character-based chunking (no tokenizer) so no embedding-model dependency is required for this step.

# COMMAND ----------

# MAGIC %pip install langchain-text-splitters --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

def chunk_text(doc: str) -> list:
    if not doc or not doc.strip():
        return []
    return splitter.split_text(doc)

chunk_udf = F.udf(chunk_text, ArrayType(StringType()), useArrow=True)

parsed_df = spark.table(PARSED_TABLE)
propagate_columns = ["doc_uri", "parser_status", "last_modified"]
doc_column = "content"

chunked_df = (
    parsed_df.withColumn("content_chunked", chunk_udf(F.col(doc_column)))
    .select(*propagate_columns, F.explode("content_chunked").alias("content_chunked"))
    .withColumn("chunk_id", F.md5(F.col("content_chunked")))
    .select("chunk_id", "content_chunked", *propagate_columns)
)

chunked_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(CHUNKED_TABLE)
n_chunks = spark.table(CHUNKED_TABLE).count()
print(f"Wrote {n_chunks} chunks to {CHUNKED_TABLE}")

spark.sql(f"ALTER TABLE {CHUNKED_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print("CDC enabled for Vector Search sync.")
display(spark.table(CHUNKED_TABLE))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Vector Search index (Delta Sync)
# MAGIC
# MAGIC Creates a vector index on the chunked table. Embeddings are computed by the specified embedding endpoint. Index name and endpoint must already exist or be created in the workspace.

# COMMAND ----------

from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,
    PipelineType,
    VectorIndexType,
)
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import ResourceDoesNotExist, BadRequest
import time

def build_retriever_index(
    vector_search_endpoint: str,
    chunked_docs_table_name: str,
    vector_search_index_name: str,
    embedding_endpoint_name: str,
    force_delete_index_before_create: bool = False,
    primary_key: str = "chunk_id",
    embedding_source_column: str = "content_chunked",
) -> tuple:
    w = WorkspaceClient()
    vsc = w.vector_search_indexes

    def find_index(name):
        try:
            return vsc.get_index(index_name=name)
        except ResourceDoesNotExist:
            return None

    existing = find_index(vector_search_index_name)
    if existing:
        if force_delete_index_before_create:
            vsc.delete_index(index_name=vector_search_index_name)
            while find_index(vector_search_index_name):
                time.sleep(30)
            create_index = True
        else:
            while not existing.status.ready:
                print("Index not ready, waiting 30s...")
                time.sleep(30)
                existing = find_index(vector_search_index_name)
            try:
                vsc.sync_index(index_name=vector_search_index_name)
                return (False, f"Sync started for {vector_search_index_name}")
            except BadRequest as e:
                return (True, f"Sync already in progress or error: {e}")
    else:
        create_index = True

    if create_index:
        delta_sync_spec = DeltaSyncVectorIndexSpecRequest(
            source_table=chunked_docs_table_name,
            pipeline_type=PipelineType.TRIGGERED,
            embedding_source_columns=[
                EmbeddingSourceColumn(
                    name=embedding_source_column,
                    embedding_model_endpoint_name=embedding_endpoint_name,
                )
            ],
        )
        vsc.create_index(
            name=vector_search_index_name,
            endpoint_name=vector_search_endpoint,
            primary_key=primary_key,
            index_type=VectorIndexType.DELTA_SYNC,
            delta_sync_index_spec=delta_sync_spec,
        )
        return (False, f"Created index {vector_search_index_name}")
    return (False, "OK")

is_error, msg = build_retriever_index(
    vector_search_endpoint=VECTOR_SEARCH_ENDPOINT,
    chunked_docs_table_name=CHUNKED_TABLE,
    vector_search_index_name=VECTOR_INDEX_NAME,
    embedding_endpoint_name=EMBEDDING_ENDPOINT,
    force_delete_index_before_create=False,
)
if is_error:
    raise Exception(msg)
print(msg)
print("Note: Index may still be syncing. Check the Vector Search UI for status.")