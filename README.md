# PDF OCR RAG Accelerator

A production-ready pipeline for turning PDFs into a queryable RAG system on **Databricks** — using open-source OCR models, distributed GPU inference, and Foundation Model APIs.

**What makes this accelerator useful:**
- **Open-source OCR** — HunyuanOCR or DeepSeek-OCR, no vendor lock-in
- **Streaming GPU pipeline** — PDFs render in memory and stream directly to OCR via Ray, no intermediate files
- **End-to-end in 3 notebooks** — OCR, index, and RAG chain with MLflow model logging
- **Classic or Serverless GPU** — same pipeline on managed clusters or Serverless GPU Compute (SGC)
- **Plug-and-play deployment** — Databricks Asset Bundles: clone, configure, `bundle deploy`, done
- **Production-ready** — MLflow tracing, Vector Search with CDC auto-sync, model registry

<!-- Add a screenshot of the pipeline running or the RAG chain output here -->
<!-- <img width="1200" alt="Pipeline overview" src="..."> -->

---

## Pipeline overview

```
┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│  01  OCR + Adapter   │     │  02  Chunk & Index   │     │  03  RAG Chain       │
│                      │     │                      │     │                      │
│  PDFs (UC Volume)    │     │  Parsed-docs table   │     │  Vector Search       │
│    ↓ Ray streaming   │     │    ↓ LangChain       │     │    ↓ Retriever       │
│  Render pages (CPU)  │────▶│  Character chunking  │────▶│  Foundation Model    │
│    ↓ vLLM (GPU)      │     │    ↓ Databricks SDK  │     │    ↓ MLflow tracing  │
│  OCR inference       │     │  Vector Search index │     │  Logged ML model     │
│    ↓ Adapter         │     │  (Delta Sync + CDC)  │     │  (ready to serve)    │
│  Parsed-docs table   │     │                      │     │                      │
└──────────────────────┘     └──────────────────────┘     └──────────────────────┘
  GPU cluster / SGC              CPU cluster                  CPU cluster
```

> **Two compute options for step 1:** Use `01-ocr-and-adapter.py` on a **classic GPU cluster**, or `01b-ocr-serverless.py` on **Serverless GPU Compute** — no cluster management needed. Steps 2 and 3 are the same either way.

---

## Quick start

Get the full pipeline running in a few commands:

```bash
git clone <this-repo>
cd pdf-ocr-rag-accelerator

# Edit databricks.yml — set your catalog, schema, pdf_dir, endpoints
databricks bundle validate -t dev
databricks bundle deploy -t dev

# Classic GPU cluster
databricks bundle run pdf_ocr_rag_pipeline -t dev

# OR — Serverless GPU Compute (no cluster to configure)
databricks bundle run pdf_ocr_rag_pipeline_sgc -t dev
```

**What you get:**
- Distributed OCR across all available GPUs — processes hundreds of PDFs
- Chunked documents indexed in Vector Search with automatic sync
- RAG chain logged as an MLflow model — one-click deploy to Model Serving
- Full tracing of every retrieval and LLM call in the MLflow Experiment UI

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Databricks workspace** | Unity Catalog enabled |
| **GPU compute** (pick one) | **Classic:** GPU cluster (e.g. `g5.xlarge`, 1+ workers, DBR 14.3 LTS ML) **or** **SGC:** Serverless GPU with A10 accelerator |
| **Vector Search endpoint** | Create in your workspace (e.g. `one-env-shared-endpoint`) |
| **Embedding endpoint** | e.g. `databricks-bge-m3` (pay-per-token or provisioned) |
| **LLM endpoint** | e.g. `databricks-meta-llama-3-3-70b-instruct` |
| **PDFs** | Uploaded to a UC Volume (e.g. `/Volumes/main/default/ocr_data/pdfs/`) |

---

## Tech stack

**OCR & Inference**
- **[vLLM](https://github.com/vllm-project/vllm)** — high-throughput LLM serving for multimodal OCR
- **[Ray Data](https://docs.ray.io/en/latest/data/data.html)** — distributed streaming with backpressure across GPUs
- **[PyMuPDF](https://pymupdf.readthedocs.io/)** — fast PDF page rendering (no poppler dependency)

**RAG & Search**
- **[Databricks Vector Search](https://docs.databricks.com/en/generative-ai/vector-search.html)** — Delta Sync index with CDC for automatic updates
- **[LangChain](https://python.langchain.com/)** + **[databricks-langchain](https://python.langchain.com/docs/integrations/providers/databricks/)** — retriever, LLM, and chain composition
- **[Foundation Model APIs](https://docs.databricks.com/en/machine-learning/model-serving/score-foundation-models.html)** — pay-per-token access to Llama, DBRX, Mixtral, etc.

**Observability & Deployment**
- **[MLflow](https://mlflow.org/)** — tracing, experiment tracking, model registry
- **[Databricks Asset Bundles](https://docs.databricks.com/en/dev-tools/bundles/)** — CI/CD-ready job deployment

---

## Usage

### Option A: Run notebooks interactively

1. **Clone** this repo into your Databricks workspace (Repos)
2. **Upload PDFs** to a UC Volume
3. **Pick your compute** and run notebooks in order:

<table>
<tr><th>Step</th><th>Classic GPU cluster</th><th>Serverless GPU (SGC)</th><th>Key config</th></tr>
<tr>
  <td><b>1</b></td>
  <td><code>01-ocr-and-adapter.py</code><br/>Attach to GPU cluster</td>
  <td><code>01b-ocr-serverless.py</code><br/>Connect → Serverless GPU → A10</td>
  <td><code>pdf_dir</code>, <code>ocr_model</code>, <code>uc_catalog</code>/<code>uc_schema</code></td>
</tr>
<tr><td><b>2</b></td><td colspan="2"><code>02-chunk-index.py</code> (CPU — same for both)</td><td><code>vector_search_endpoint</code>, <code>embedding_endpoint</code></td></tr>
<tr><td><b>3</b></td><td colspan="2"><code>03-rag-chain.py</code> (CPU — same for both)</td><td><code>llm_endpoint</code>, <code>num_results</code></td></tr>
</table>

### Option B: Deploy with Asset Bundles

```bash
# 1. Configure
#    Edit databricks.yml — set catalog, schema, pdf_dir, endpoints

# 2. Deploy (uploads notebooks, creates both jobs)
databricks bundle deploy -t dev

# 3. Run — pick one:
databricks bundle run pdf_ocr_rag_pipeline -t dev       # Classic GPU cluster
databricks bundle run pdf_ocr_rag_pipeline_sgc -t dev   # Serverless GPU
```

Both jobs run three tasks in sequence: OCR → Chunk & Index → RAG Chain. All parameters are passed from bundle variables — no widget configuration needed in the UI.

---

## Classic vs Serverless GPU

| | Classic (`01-ocr-and-adapter.py`) | Serverless (`01b-ocr-serverless.py`) |
|---|---|---|
| **Compute** | You configure GPU cluster (node type, workers, DBR) | Auto-provisioned A10 GPUs — no cluster management |
| **Ray init** | `ray.init()` connects to cluster's Ray instance | `@ray_launch(gpus=N, gpu_type='a10', remote=True)` provisions remote GPUs |
| **GPU count** | Auto-detected from cluster resources | Explicit `num_gpus` widget (e.g. 5) |
| **Execution** | Direct function call | `.distributed()` on decorated function |
| **Startup** | Fast (cluster already running) | A few minutes (GPU provisioning) |
| **Output** | Identical `rag_parsed_docs` table | Identical `rag_parsed_docs` table |

Both notebooks produce the same output — notebooks 02 and 03 work unchanged regardless of which you use.

---

## Input modes (notebook 01 / 01b)

| Mode | Input | How it works | When to use |
|---|---|---|---|
| **`pdf`** (default) | UC Volume of `.pdf` files | Ray streams PDFs, renders pages in memory, feeds directly to GPU OCR. No intermediate files. Optional: `save_images=true` to checkpoint PNGs. | Fresh pipeline, large PDF corpora |
| **`images`** | UC Volume of `.png` files named `{doc}_page_{N}.png` | Reads pre-existing PNGs, normalizes to same schema, runs OCR. | Already have rendered images, or re-running OCR with different model |

### How streaming works (PDF mode)

```
ray.data.read_binary_files(pdf_dir)
    ↓  .flat_map(render_pages)        ← CPU workers: PDF → PNG bytes in memory
    ↓  .map_batches(OCRPredictor)     ← GPU workers: vLLM inference (1 GPU each)
    ↓  .write_parquet(...)            ← results to UC Volume
```

Ray Data's streaming executor applies **backpressure** — pages are rendered only as fast as the GPU can consume them. Memory stays bounded regardless of corpus size.

---

## RAG chain (notebook 03)

```
User question
    ↓  DatabricksVectorSearch retriever (top-k chunks)
    ↓  Format context with source attribution
    ↓  ChatDatabricks LLM (Foundation Model API)
    ↓  Answer with document citations
```

**Built-in enhancements** (commented out, ready to enable):
- **Reranking** — re-score chunks with a cross-encoder for better precision
- **Hybrid search** — combine vector + keyword search for acronyms and proper nouns
- **Chunk enrichment** — add sibling context from adjacent chunks

The chain is logged as an **MLflow model** in Unity Catalog — deploy to Model Serving from the UI or CLI.

---

## Project structure

```
pdf-ocr-rag-accelerator/
├── notebooks/
│   ├── 01-ocr-and-adapter.py      # OCR pipeline — classic GPU cluster
│   ├── 01b-ocr-serverless.py      # OCR pipeline — Serverless GPU (SGC)
│   ├── 02-chunk-index.py          # Chunking + Vector Search index
│   └── 03-rag-chain.py            # RAG chain + MLflow model logging
│
├── databricks.yml                 # DAB root config (variables, targets)
├── resources/
│   ├── pdf_ocr_rag_job.yml        # DAB job — classic GPU cluster
│   └── pdf_ocr_rag_job_sgc.yml   # DAB job — Serverless GPU (SGC)
│
├── config/
│   └── config.yml.example         # Config template for standalone script
├── scripts/
│   └── run_pipeline.py            # Chunk + index only (standalone)
├── workflows/
│   ├── pdf_rag_job.json           # Job JSON (manual import alternative)
│   └── README.md
│
├── requirements.txt
└── README.md
```

---

## Configuration reference

### Notebook 01 / 01b — OCR + Adapter

| Parameter | Description | Default |
|---|---|---|
| `input_mode` | `pdf` or `images` | `pdf` |
| `pdf_dir` | UC Volume path to PDF files | `/Volumes/main/default/ocr_data/pdfs` |
| `image_dir` | UC Volume path to PNGs (image mode, or checkpoint dir) | `/Volumes/main/default/ocr_data/images` |
| `save_images` | Save rendered PNGs as checkpoint | `false` |
| `render_dpi` | DPI for PDF page rendering | `300` |
| `ocr_model` | `tencent/HunyuanOCR` or `deepseek-ai/DeepSeek-OCR` | `deepseek-ai/DeepSeek-OCR` |
| `batch_size` | vLLM inference batch size | `4` |
| `num_gpus` | Number of A10 GPUs to provision (**01b only**) | `5` |
| `uc_catalog` / `uc_schema` | Unity Catalog location | `main` / `default` |
| `parsed_table` | Adapter output table name | `rag_parsed_docs` |

### Notebook 02 — Chunk & Index

| Parameter | Description | Default |
|---|---|---|
| `catalog` / `schema` | Must match notebook 01 | `main` / `default` |
| `parsed_table_suffix` | Parsed table from 01 | `rag_parsed_docs` |
| `vector_search_endpoint` | Databricks Vector Search endpoint | `one-env-shared-endpoint` |
| `embedding_endpoint` | Embedding model endpoint | `databricks-bge-m3` |
| `chunk_size` / `chunk_overlap` | Chunking parameters | `500` / `50` |

### Notebook 03 — RAG Chain

| Parameter | Description | Default |
|---|---|---|
| `catalog` / `schema` | Must match notebook 02 | `main` / `default` |
| `vector_search_endpoint` | Same as notebook 02 | `one-env-shared-endpoint` |
| `llm_endpoint` | Foundation Model API endpoint | `databricks-meta-llama-3-3-70b-instruct` |
| `num_results` | Chunks to retrieve (top-k) | `5` |

---

## Adapter (01 → 02)

The OCR table has **one row per page** (`doc_uri`, `page_num`, `ocr_text`). The adapter in notebook 01 groups by `doc_uri`, concatenates `ocr_text` in page order, and writes a table with **one row per document**: `content`, `parser_status`, `doc_uri`, `last_modified`. That table is the input to notebook 02.

---

## Troubleshooting

**OCR is slow or OOM on large PDFs**
- Reduce `render_dpi` from 300 to 200 — smaller images, faster inference, lower memory
- Increase `batch_size` if GPU memory allows, decrease if OOM
- Add more GPU workers to the cluster — Ray auto-distributes

**Vector Search index not syncing**
- Check that CDC is enabled: `ALTER TABLE ... SET TBLPROPERTIES (delta.enableChangeDataFeed = true)`
- Verify the Vector Search endpoint is running in the workspace UI
- Manually trigger sync: notebook 02 does this automatically, but you can also use the SDK

**RAG chain returns "I don't have enough information"**
- Check that the Vector Search index has finished syncing (check VS UI)
- Increase `num_results` to retrieve more chunks
- Verify your query relates to the indexed documents
- Enable hybrid search (uncomment in notebook 03) for better recall on exact terms

**Ray cluster issues (classic)**
- Ensure DBR 14.3 LTS ML or later (Ray is pre-installed)
- Check `ray.cluster_resources()` shows expected GPU count
- If Ray init fails, try `ray.shutdown()` first, then re-init

**Serverless GPU (SGC) issues**
- Ensure you selected **Serverless GPU** → **A10** in the notebook's Connect menu
- `@ray_launch` provisioning can take 2-5 minutes on first run — this is normal
- If provisioning fails, check your workspace has SGC enabled and GPU quota available
- Reduce `num_gpus` if you hit quota limits

**HuggingFace gated model access**
- Set `hf_secret_scope` and `hf_secret_key` widgets to your Databricks secret containing the HF token
- Ensure your HF account has accepted the model's license terms

---

## Next steps

Once the pipeline is running:

1. **Enable advanced RAG features** — uncomment reranking, hybrid search, or chunk enrichment in notebook 03
2. **Deploy to Model Serving** — the logged MLflow model can be served as an endpoint for production queries
3. **Add evaluation** — use `mlflow.evaluate()` to measure retrieval and answer quality
4. **Schedule the pipeline** — add a `schedule:` block in `resources/pdf_ocr_rag_job.yml` for automated re-indexing
5. **Monitor in production** — enable inference tables on the serving endpoint for observability

---

## License and attribution

OCR models: [HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR) (Tencent), [DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) (DeepSeek). See their HuggingFace model cards for license terms.

| Library | Purpose | License |
|---|---|---|
| vLLM | LLM inference engine | Apache 2.0 |
| Ray | Distributed computing | Apache 2.0 |
| PyMuPDF | PDF rendering | AGPL 3.0 |
| LangChain | RAG chain composition | MIT |
| MLflow | Tracing & model registry | Apache 2.0 |
| Databricks SDK | Vector Search & platform APIs | Apache 2.0 |
