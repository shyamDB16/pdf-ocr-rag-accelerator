# PDF OCR + Vector Search Pipeline

Turn a volume of PDFs into a searchable Vector Search index on Databricks, using open-source OCR models on GPUs.

The pipeline has two tasks:

1. **OCR** — Ray Data streams PDFs through PyMuPDF (page rendering) and vLLM (GPU inference) to produce a parsed-docs Delta table. Runs on a classic GPU cluster (`01-ocr-and-adapter.py`) or Serverless GPU Compute (`01b-ocr-serverless.py`).
2. **Chunk + Index** — Splits documents into chunks and creates a Databricks Vector Search index with Delta Sync for automatic updates (`02-chunk-index.py`).

Supported OCR models: **DeepSeek-OCR** and **HunyuanOCR** (set via the `ocr_model` variable).

A third notebook (`03-rag-chain.py`) is included for building a RAG chain on top of the index, but it is not part of the core pipeline.

## Quick start

```bash
git clone <this-repo>
cd pdf-ocr-rag-accelerator

# 1. Edit databricks.yml — set catalog, schema, pdf_dir, and endpoint names
databricks bundle validate -t dev
databricks bundle deploy -t dev

# 2. Run the pipeline (pick one):
databricks bundle run pdf_ocr_rag_pipeline -t dev       # Classic GPU cluster
databricks bundle run pdf_ocr_rag_pipeline_sgc -t dev   # Serverless GPU Compute
```

Both jobs run OCR then Chunk+Index in sequence. All parameters flow from bundle variables — no widget configuration needed.

## Prerequisites

- Databricks workspace with **Unity Catalog** enabled
- **GPU compute**: a classic GPU cluster (e.g. `g5.xlarge`, DBR 14.3 LTS ML) _or_ Serverless GPU with A10 accelerator
- A **Vector Search endpoint** in your workspace
- An **embedding model endpoint** (e.g. `databricks-qwen3-embedding-0-6b`)
- PDFs uploaded to a **UC Volume**

## Configuration

All variables live in `databricks.yml`. The key ones:

| Variable | What it controls | Default |
|---|---|---|
| `catalog` / `schema` | Unity Catalog location for tables and index | `shyam_catalog_ss` / `llm_download` |
| `pdf_dir` | UC Volume path to input PDFs | `/Volumes/.../ocr_data/pdf` |
| `ocr_model` | HuggingFace model ID | `tencent/HunyuanOCR` |
| `input_mode` | `pdf` (stream from PDFs) or `images` (pre-rendered PNGs) | `pdf` |
| `save_images` | Checkpoint rendered PNGs to `image_dir` | `false` |
| `sgc_num_gpus` | Number of A10 GPUs (Serverless job only) | `1` |
| `vector_search_endpoint` | Vector Search endpoint name | `one-env-shared-endpoint` |
| `embedding_endpoint` | Embedding model for the index | `databricks-qwen3-embedding-0-6b` |
| `hf_secret_scope` / `hf_secret_key` | Databricks secret for gated HuggingFace models | — |

Override per-target under `targets.dev.variables` or `targets.prod.variables`.

## Project structure

```
pdf-ocr-rag-accelerator/
├── databricks.yml                  # Bundle config — variables, targets
├── resources/
│   ├── pdf_ocr_rag_job.yml         # Job definition (classic GPU)
│   └── pdf_ocr_rag_job_sgc.yml    # Job definition (Serverless GPU)
├── notebooks/
│   ├── 01-ocr-and-adapter.py       # Task 1: OCR — classic GPU cluster
│   ├── 01b-ocr-serverless.py       # Task 1: OCR — Serverless GPU Compute
│   ├── 02-chunk-index.py           # Task 2: chunk + Vector Search index
│   └── 03-rag-chain.py            # (Optional) RAG chain + MLflow logging
└── requirements.txt
```

## How OCR streaming works

In `pdf` mode, Ray Data builds a streaming pipeline with backpressure:

```
read_binary_files(pdf_dir)
  → flat_map(render_pages)       # CPU: PyMuPDF renders each page to PNG bytes
  → map_batches(OCRPredictor)    # GPU: vLLM runs the OCR model
  → write to Delta table         # one row per document (pages concatenated)
```

Pages are rendered only as fast as the GPU can consume them, so memory stays bounded regardless of corpus size.

## Tech stack

**Ray Data** for distributed streaming across GPUs, **vLLM** for high-throughput multimodal inference, **PyMuPDF** for PDF rendering, **Databricks Vector Search** (Delta Sync + CDC) for the search index, and **Databricks Asset Bundles** for deployment.
