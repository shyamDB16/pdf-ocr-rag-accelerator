# PDF OCR RAG Accelerator

Plug-and-play pipeline: **OCR document images** (HunyuanOCR or DeepSeek-OCR on a GPU cluster), **adapter** to parsed-docs shape, then **chunk and build a Vector Search index** for RAG. Works in **Databricks** (notebooks run as `.py` in Repos).

## What it does

| Step | Notebook | Output |
|------|----------|--------|
| **1. OCR + adapter** | `01-ocr-and-adapter.py` | OCR Delta table (one row per page) → **parsed-docs** Delta table (one row per document: `content`, `parser_status`, `doc_uri`, `last_modified`) |
| **2. Chunk & index** | `02-chunk-index.py` | Chunked Delta table + Vector Search index (Delta Sync) |

- **OCR**: Distributed OCR on a **classic GPU cluster** (Ray + vLLM), HunyuanOCR or DeepSeek-OCR. Input: PNG images in a UC volume directory (e.g. from PDF→image conversion). Output: one row per page; then an **adapter** aggregates by document and writes the parsed-docs table expected by 02.
- **Chunking**: Character-based (LangChain `RecursiveCharacterTextSplitter`).
- **Index**: Databricks Vector Search with Delta Sync; embeddings from the configured embedding endpoint.

## Repo layout

```
pdf-ocr-rag-accelerator/
├── README.md
├── databricks.yml              # Databricks Asset Bundle root config
├── resources/
│   └── pdf_ocr_rag_job.yml     # DAB job: ocr_and_adapter (GPU) → chunk_index (CPU)
├── config/
│   └── config.yml.example
├── notebooks/
│   ├── 01-ocr-and-adapter.py   # OCR (Ray+vLLM) + adapter → parsed-docs table
│   └── 02-chunk-index.py       # Chunk + Vector Search index
├── workflows/
│   ├── pdf_rag_job.json        # Job JSON (manual import alternative)
│   └── README.md
├── docs/
│   └── USING_AI_DEV_KIT_FOR_DAB.md   # How to use AI Dev Kit to build this DAB
├── scripts/
│   └── run_pipeline.py         # Chunk + index only (after 01 has produced parsed-docs)
└── requirements.txt
```

## Use in Databricks

1. **Clone or import** this repo into your workspace (Repos).
2. **Input**: Put **PNG images** in a Unity Catalog volume directory (e.g. `/Volumes/<catalog>/<schema>/<volume>/images/`). For PDFs, use the PDF→image helper in 01 (e.g. `pdf_to_images_pymupdf`) in a separate run or script, then point `image_dir` at that directory.
3. **Config**: In **01-ocr-and-adapter.py**, set widgets: `image_dir`, `ocr_model` (HunyuanOCR or DeepSeek-OCR), `uc_catalog`, `uc_schema`, `uc_volume`, `uc_table` (OCR results), `parsed_table` (adapter output). In **02-chunk-index.py**, set `catalog`, `schema`, `parsed_table_suffix` to match 01’s parsed table, plus Vector Search endpoint and embedding endpoint.
4. **Run order**: Run **01-ocr-and-adapter.py** on a **GPU cluster** (Ray + vLLM). Then run **02-chunk-index.py** on a CPU cluster (or same cluster).
5. **Job**: Use the DAB (see below) or `workflows/pdf_rag_job.json` for manual import. See **workflows/README.md**.

## Deploy with Databricks Asset Bundles (DABs)

Deploy and run the pipeline as a single job using the Databricks CLI:

1. **Prerequisites**: [Databricks CLI](https://docs.databricks.com/dev-tools/cli/) installed and configured (`databricks configure --profile DEFAULT`).
2. **Edit** `databricks.yml`: set `workspace.profile` (or `workspace.host`) for each target (`dev` / `prod`) to match your workspace.
3. **Validate**: `databricks bundle validate -t dev`
4. **Deploy** (uploads notebooks and creates/updates the job): `databricks bundle deploy -t dev`
5. **Run**: `databricks bundle run pdf_ocr_rag_pipeline -t dev`

**Using the AI Dev Kit to build or change the DAB:** See [docs/USING_AI_DEV_KIT_FOR_DAB.md](docs/USING_AI_DEV_KIT_FOR_DAB.md) for step-by-step instructions (install the kit from this repo, open in Cursor, ask the AI to create or modify the bundle).

### How the DAB works (plug and play)

| Step | What happens |
|------|-------------------------------|
| **Deploy** | The bundle uploads `notebooks/01-ocr-and-adapter.py` and `02-chunk-index.py` to your workspace and creates or updates the job **PDF OCR RAG Pipeline**. Task parameters (catalog, schema, image_dir, parsed table, Vector Search/embedding endpoints) are taken from **bundle variables** in `databricks.yml`. |
| **Configure once** | Set variables in `databricks.yml` (globally or per target). The ones that usually need changing: **`image_dir`** (where your PNGs live), **`catalog`** and **`schema`** if not `main`/`default`, and **`vector_search_endpoint`** / **`embedding_endpoint`** if your workspace uses different names. |
| **Run** | `databricks bundle run pdf_ocr_rag_pipeline -t dev` starts the job. No need to fill in widget values in the UI; the job passes them from the bundle. |

So **plug and play** means: clone the repo → set `image_dir` (and optionally catalog/schema/endpoints) in `databricks.yml` → `bundle deploy` → `bundle run`. Ensure PNGs are in `image_dir` and that the Vector Search and embedding endpoints exist in the workspace. To add a schedule, add a `schedule:` block under the job in `resources/pdf_ocr_rag_job.yml`.

## Adapter (01 → 02)

The OCR table has **one row per page** (`image_path`, `ocr_text`, `model`, `timestamp`). The **adapter** in 01 derives `doc_uri` from `image_path` (e.g. `mydoc_page_1.png` → base name), concatenates `ocr_text` in page order, and writes a table with one row per document: `content`, `parser_status`, `doc_uri`, `last_modified`. That table is the input to 02.

## Config reference

| Key | Description |
|-----|-------------|
| `image_dir` | Directory of PNG images (e.g. `/Volumes/<catalog>/<schema>/<volume>/images`) |
| `ocr_model` | `tencent/HunyuanOCR` or `deepseek-ai/DeepSeek-OCR` |
| `catalog`, `schema` | Unity Catalog catalog and schema |
| `parsed_table` | Full table name for adapter output (input to 02) |
| `chunked_table`, `vector_index_name` | Chunked table and Vector Search index |
| `vector_search_endpoint`, `embedding_endpoint` | Databricks Vector Search and embedding endpoints |
| `chunk_size`, `chunk_overlap` | Chunking parameters |

## Requirements

- **01**: Classic Databricks **GPU cluster** (e.g. g5.xlarge), Ray, vLLM, PyMuPDF, transformers. Hugging Face token if using gated models.
- **02**: CPU cluster; `langchain-text-splitters`, Databricks SDK. Vector Search and embedding endpoints must exist in the workspace.

## Scripts

`scripts/run_pipeline.py` runs only the **chunk + index** step (reads parsed-docs from 01, chunks, writes chunked table, creates/updates Vector Search index). Run **01-ocr-and-adapter.py** (Ray, vLLM, GPU) first to produce the parsed-docs table.

## Pushing to your Git

Use this folder as a standalone repo (e.g. `pdf-ocr-rag-accelerator` on GitHub):

1. Create a new repo on your Git host. Do not add a README if you already have one here.
2. From this folder:
   ```bash
   cd pdf-ocr-rag-accelerator
   git init
   git add .
   git commit -m "Initial commit: PDF OCR RAG accelerator"
   git remote add origin https://github.com/<your-org>/pdf-ocr-rag-accelerator.git
   git branch -M main
   git push -u origin main
   ```

## License and attribution

See repo root. OCR models: HunyuanOCR (Tencent), DeepSeek-OCR (DeepSeek). See their Hugging Face model cards for terms.
