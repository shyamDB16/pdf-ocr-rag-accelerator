# Databricks notebook source
# MAGIC %md
# MAGIC # 3. RAG Chain — Retrieve & Generate
# MAGIC
# MAGIC <div style="background:#e7f3fe;border-left:6px solid #2196F3;padding:12px;margin:12px 0;border-radius:4px">
# MAGIC <b>What this notebook does</b><br/>
# MAGIC Builds a RAG chain on top of the Vector Search index created in <b>02-chunk-index</b>:
# MAGIC <ol>
# MAGIC   <li>Retrieves relevant chunks from the Vector Search index</li>
# MAGIC   <li>Generates answers using a Foundation Model API endpoint</li>
# MAGIC   <li>Traces every call with MLflow for observability</li>
# MAGIC   <li>Logs the chain as an MLflow model (ready for Model Serving)</li>
# MAGIC </ol>
# MAGIC </div>
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Notebook **02-chunk-index** has run successfully (Vector Search index exists and is synced)
# MAGIC - A Foundation Model API endpoint is available (e.g. `databricks-meta-llama-3-3-70b-instruct`)
# MAGIC
# MAGIC **Pipeline context:**
# MAGIC ```
# MAGIC 01-ocr-and-adapter  →  02-chunk-index  →  03-rag-chain (this notebook)
# MAGIC       (OCR)              (index)            (retrieve + generate)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install dependencies

# COMMAND ----------

%pip install databricks-langchain langchain-core mlflow --quiet
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC <div style="background:#fff3cd;border-left:6px solid #ffc107;padding:12px;margin:12px 0;border-radius:4px">
# MAGIC <b>Endpoints</b><br/>
# MAGIC <code>vector_search_endpoint</code> and <code>vector_index_name</code> must match what was created in notebook 02.<br/>
# MAGIC <code>llm_endpoint</code> is a Foundation Model API pay-per-token endpoint or a provisioned-throughput serving endpoint.
# MAGIC </div>

# COMMAND ----------

dbutils.widgets.text("catalog", "main", "UC Catalog")
dbutils.widgets.text("schema", "default", "UC Schema")
dbutils.widgets.text("vector_search_endpoint", "one-env-shared-endpoint", "Vector Search Endpoint")
dbutils.widgets.text("llm_endpoint", "databricks-meta-llama-3-3-70b-instruct", "LLM Endpoint")
dbutils.widgets.text("num_results", "5", "Number of chunks to retrieve")

# COMMAND ----------

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VECTOR_INDEX_NAME = f"{CATALOG}.{SCHEMA}.rag_docs_chunked_index"
VECTOR_SEARCH_ENDPOINT = dbutils.widgets.get("vector_search_endpoint")
LLM_ENDPOINT = dbutils.widgets.get("llm_endpoint")
NUM_RESULTS = int(dbutils.widgets.get("num_results"))

print(f"Vector index:    {VECTOR_INDEX_NAME}")
print(f"VS endpoint:     {VECTOR_SEARCH_ENDPOINT}")
print(f"LLM endpoint:    {LLM_ENDPOINT}")
print(f"Top-k:           {NUM_RESULTS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up retriever
# MAGIC
# MAGIC Uses the Databricks Vector Search index created in notebook 02. The retriever returns the top-k most similar chunks for a given query.

# COMMAND ----------

from databricks_langchain import DatabricksVectorSearch

vector_store = DatabricksVectorSearch(
    endpoint=VECTOR_SEARCH_ENDPOINT,
    index_name=VECTOR_INDEX_NAME,
    text_column="content_chunked",
    columns=["chunk_id", "content_chunked", "doc_uri"],
)

retriever = vector_store.as_retriever(search_kwargs={"k": NUM_RESULTS})

# Quick test
test_docs = retriever.invoke("test query")
print(f"✓ Retriever connected — returned {len(test_docs)} chunk(s) for test query")
if test_docs:
    print(f"  Sample doc_uri: {test_docs[0].metadata.get('doc_uri', 'N/A')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up LLM
# MAGIC
# MAGIC Uses the Databricks Foundation Model API. Swap `llm_endpoint` to use any served model (DBRX, Llama, Mixtral, or your own fine-tuned model).

# COMMAND ----------

from databricks_langchain import ChatDatabricks

llm = ChatDatabricks(
    endpoint=LLM_ENDPOINT,
    temperature=0.1,
    max_tokens=1024,
)

# Quick test
test_response = llm.invoke("Say hello in one word.")
print(f"✓ LLM connected — response: {test_response.content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build the RAG chain
# MAGIC
# MAGIC <div style="background:#d4edda;border-left:6px solid #28a745;padding:12px;margin:12px 0;border-radius:4px">
# MAGIC <b>Chain architecture</b><br/>
# MAGIC <code>question → retriever → format context → prompt → LLM → answer</code><br/>
# MAGIC Uses LangChain Expression Language (LCEL) for composability.
# MAGIC </div>

# COMMAND ----------

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Prompt template ---
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that answers questions based on the provided document context.

Instructions:
- Use ONLY the information in the context below to answer the question.
- If the context does not contain enough information, say "I don't have enough information in the provided documents to answer this question."
- Cite the source document when possible (use the doc_uri).
- Be concise and direct."""),
    ("human", """Context from retrieved documents:

{context}

---

Question: {question}"""),
])


def format_docs(docs):
    """Format retrieved documents into a single context string with source attribution."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("doc_uri", "unknown")
        formatted.append(f"[{i}] (Source: {source})\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


# --- Assemble the chain ---
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)

print("✓ RAG chain assembled")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Enable MLflow tracing
# MAGIC
# MAGIC Every chain invocation is traced — inputs, retrieved chunks, LLM calls, outputs, and latencies are all logged automatically.

# COMMAND ----------

import mlflow

mlflow.langchain.autolog()
print("✓ MLflow tracing enabled — traces will appear in the MLflow Experiment UI")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the chain
# MAGIC
# MAGIC Run a few queries to verify end-to-end retrieval and generation.

# COMMAND ----------

# --- Replace with questions relevant to YOUR documents ---
test_questions = [
    "What are the key findings in the documents?",
    "Summarize the main topics covered.",
]

for q in test_questions:
    print(f"\n{'='*60}")
    print(f"Q: {q}")
    print(f"{'='*60}")
    answer = rag_chain.invoke(q)
    print(f"\nA: {answer}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrieval with sources
# MAGIC
# MAGIC Helper to return both the answer and the source documents used.

# COMMAND ----------

from langchain_core.runnables import RunnableParallel

rag_chain_with_sources = RunnableParallel(
    {
        "answer": rag_chain,
        "source_documents": retriever,
    }
)

# Example usage
result = rag_chain_with_sources.invoke("What are the key findings?")
print(f"Answer: {result['answer']}\n")
print("Sources:")
for doc in result["source_documents"]:
    print(f"  - {doc.metadata.get('doc_uri', 'unknown')} (chunk: {doc.metadata.get('chunk_id', 'N/A')[:8]}...)")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Advanced features (commented — uncomment to enable)
# MAGIC
# MAGIC <div style="background:#e7f3fe;border-left:6px solid #2196F3;padding:12px;margin:12px 0;border-radius:4px">
# MAGIC <b>Available enhancements</b>
# MAGIC <ul>
# MAGIC   <li><b>Reranking</b> — re-score retrieved chunks with a cross-encoder for better precision</li>
# MAGIC   <li><b>Hybrid search</b> — combine semantic (vector) and keyword (BM25) retrieval</li>
# MAGIC   <li><b>Chunk enrichment</b> — add parent/sibling context around each chunk</li>
# MAGIC </ul>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reranking
# MAGIC
# MAGIC Re-scores the top-k retrieved chunks using a cross-encoder model to improve precision.
# MAGIC Requires a reranker model endpoint (e.g. `bge-reranker-v2-m3` served on Model Serving).

# COMMAND ----------

# # --- Reranking with a served cross-encoder ---
# # Uncomment and set your reranker endpoint name
#
# RERANKER_ENDPOINT = "your-reranker-endpoint"
# RETRIEVAL_TOP_K = 20  # retrieve more, then rerank down to NUM_RESULTS
#
# from databricks.sdk import WorkspaceClient
# from langchain_core.documents import Document
#
# w = WorkspaceClient()
#
# def rerank_docs(query_and_docs: dict) -> list:
#     """Rerank documents using a cross-encoder model served on Databricks."""
#     query = query_and_docs["question"]
#     docs = query_and_docs["documents"]
#
#     passages = [doc.page_content for doc in docs]
#     pairs = [{"query": query, "passage": p} for p in passages]
#
#     response = w.serving_endpoints.query(
#         name=RERANKER_ENDPOINT,
#         dataframe_records=pairs,
#     )
#
#     scores = [r["score"] for r in response.predictions]
#     ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
#     return [doc for _, doc in ranked[:NUM_RESULTS]]
#
#
# wide_retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_TOP_K})
#
# rag_chain_with_reranking = (
#     {
#         "question": RunnablePassthrough(),
#         "documents": wide_retriever,
#     }
#     | RunnablePassthrough.assign(context=lambda x: format_docs(rerank_docs(x)))
#     | RAG_PROMPT
#     | llm
#     | StrOutputParser()
# )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Hybrid search
# MAGIC
# MAGIC Combines vector similarity with keyword matching for better recall on exact terms,
# MAGIC acronyms, and proper nouns that embeddings may not capture well.

# COMMAND ----------

# # --- Hybrid search: vector + keyword ---
# # Uncomment to enable. Requires the chunked Delta table for keyword search.
#
# from pyspark.sql import functions as F
#
# CHUNKED_TABLE = f"{CATALOG}.{SCHEMA}.rag_docs_chunked"
#
# def keyword_search(query: str, top_k: int = 10) -> list:
#     """Simple keyword search using Spark SQL LIKE on the chunked table."""
#     keywords = query.lower().split()
#     conditions = " OR ".join(
#         [f"lower(content_chunked) LIKE '%{kw}%'" for kw in keywords]
#     )
#     results = (
#         spark.table(CHUNKED_TABLE)
#         .filter(conditions)
#         .limit(top_k)
#         .collect()
#     )
#     from langchain_core.documents import Document
#     return [
#         Document(
#             page_content=row["content_chunked"],
#             metadata={"doc_uri": row["doc_uri"], "chunk_id": row["chunk_id"], "search": "keyword"},
#         )
#         for row in results
#     ]
#
#
# def hybrid_search(query: str) -> list:
#     """Merge vector and keyword results, deduplicate by chunk_id."""
#     vector_docs = retriever.invoke(query)
#     kw_docs = keyword_search(query, top_k=NUM_RESULTS)
#
#     seen = set()
#     merged = []
#     for doc in vector_docs + kw_docs:
#         cid = doc.metadata.get("chunk_id", doc.page_content[:50])
#         if cid not in seen:
#             seen.add(cid)
#             merged.append(doc)
#     return merged[:NUM_RESULTS]
#
#
# rag_chain_hybrid = (
#     {
#         "context": (lambda q: format_docs(hybrid_search(q))),
#         "question": RunnablePassthrough(),
#     }
#     | RAG_PROMPT
#     | llm
#     | StrOutputParser()
# )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Chunk enrichment
# MAGIC
# MAGIC Adds surrounding context (previous/next chunks from the same document) to each retrieved chunk,
# MAGIC giving the LLM more context without increasing the retrieval window.

# COMMAND ----------

# # --- Chunk enrichment: add sibling context ---
# # Uncomment to enable. Reads adjacent chunks from the chunked Delta table.
#
# CHUNKED_TABLE = f"{CATALOG}.{SCHEMA}.rag_docs_chunked"
#
# from pyspark.sql import Window
#
# def enrich_chunks(docs: list) -> list:
#     """For each retrieved chunk, fetch previous and next chunks from the same document."""
#     if not docs:
#         return docs
#
#     chunk_ids = [doc.metadata.get("chunk_id") for doc in docs]
#
#     # Build a window over each document's chunks ordered by chunk_id
#     chunked_df = spark.table(CHUNKED_TABLE)
#     w = Window.partitionBy("doc_uri").orderBy("chunk_id")
#
#     enriched_df = (
#         chunked_df
#         .withColumn("prev_chunk", F.lag("content_chunked", 1).over(w))
#         .withColumn("next_chunk", F.lead("content_chunked", 1).over(w))
#         .filter(F.col("chunk_id").isin(chunk_ids))
#         .collect()
#     )
#
#     enriched_map = {
#         row["chunk_id"]: {
#             "prev": row["prev_chunk"] or "",
#             "content": row["content_chunked"],
#             "next": row["next_chunk"] or "",
#         }
#         for row in enriched_df
#     }
#
#     from langchain_core.documents import Document
#     enriched_docs = []
#     for doc in docs:
#         cid = doc.metadata.get("chunk_id")
#         if cid in enriched_map:
#             ctx = enriched_map[cid]
#             full_text = "\n\n".join(filter(None, [ctx["prev"], ctx["content"], ctx["next"]]))
#             enriched_docs.append(
#                 Document(page_content=full_text, metadata=doc.metadata)
#             )
#         else:
#             enriched_docs.append(doc)
#     return enriched_docs
#
#
# enriched_retriever = retriever | enrich_chunks
#
# rag_chain_enriched = (
#     {
#         "context": enriched_retriever | format_docs,
#         "question": RunnablePassthrough(),
#     }
#     | RAG_PROMPT
#     | llm
#     | StrOutputParser()
# )

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Log chain as MLflow model
# MAGIC
# MAGIC <div style="background:#d4edda;border-left:6px solid #28a745;padding:12px;margin:12px 0;border-radius:4px">
# MAGIC <b>Why log the chain?</b><br/>
# MAGIC Logging the RAG chain as an MLflow model enables:
# MAGIC <ul>
# MAGIC   <li>One-click deployment to <b>Model Serving</b></li>
# MAGIC   <li>Versioned model registry in Unity Catalog</li>
# MAGIC   <li>Evaluation with <code>mlflow.evaluate()</code> (notebook 04)</li>
# MAGIC   <li>Inference tables for production monitoring</li>
# MAGIC </ul>
# MAGIC </div>

# COMMAND ----------

import mlflow
from mlflow.models import infer_signature

mlflow.set_registry_uri("databricks-uc")

MODEL_NAME = f"{CATALOG}.{SCHEMA}.pdf_ocr_rag_chain"

# Infer signature from a sample
sample_input = "What are the key findings?"
sample_output = rag_chain.invoke(sample_input)
signature = infer_signature(sample_input, sample_output)

with mlflow.start_run(run_name="pdf_ocr_rag_chain") as run:
    model_info = mlflow.langchain.log_model(
        lc_model=rag_chain,
        artifact_path="rag_chain",
        registered_model_name=MODEL_NAME,
        input_example=sample_input,
        signature=signature,
    )
    print(f"✓ Model logged: {model_info.model_uri}")
    print(f"  Run ID: {run.info.run_id}")

print(f"\n✓ Registered as: {MODEL_NAME}")
print("  → Deploy to Model Serving from the Unity Catalog Models UI")
print("  → Or use: mlflow.deployments.create_deployment(...)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quick test — load from MLflow and query
# MAGIC
# MAGIC Verifies the logged model works when loaded back.

# COMMAND ----------

loaded_chain = mlflow.langchain.load_model(model_info.model_uri)

answer = loaded_chain.invoke("Summarize the main topics covered in the documents.")
print(f"Answer from loaded model:\n{answer}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Next steps
# MAGIC
# MAGIC <div style="background:#e7f3fe;border-left:6px solid #2196F3;padding:12px;margin:12px 0;border-radius:4px">
# MAGIC <ol>
# MAGIC   <li><b>Evaluate</b> — run <code>04-evaluate.py</code> to measure retrieval and answer quality with <code>mlflow.evaluate()</code></li>
# MAGIC   <li><b>Deploy</b> — run <code>05-deploy-and-monitor.py</code> to create a Model Serving endpoint with inference tables</li>
# MAGIC   <li><b>Iterate</b> — enable reranking, hybrid search, or chunk enrichment above and re-log the model</li>
# MAGIC </ol>
# MAGIC </div>
