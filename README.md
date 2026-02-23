# PDF OCR RAG Accelerator

Plug-and-play pipeline for **ingesting many PDFs**, **parsing** (and optionally **OCR**), and building a **retrieval-ready** Delta table + Vector Search index. Works in **Databricks** and **externally** (e.g. Spark Connect, local Python).

## What it does

| Step | Notebook / Script | Output |
|------|-------------------|--------|
| **1. Ingest & parse** | `01-ingest-parse-pdfs.ipynb` | Delta table: `content`, `parser_status`, `doc_uri`, `last_modified` |
| **2. Chunk & index** | `02-chunk-index.ipynb` | Chunked Delta table + Vector Search index (Delta Sync) |

- **Parsing**: Native PDF text extraction (PyMuPDF + `pymupdf4llm` → markdown). For scanned PDFs, use the OCR notebooks in the parent repo (`sgc-vllm-ocr.ipynb` or `classic-cluster-ocr-hunyuanocr.ipynb`) to produce images/text first, then point this pipeline at the resulting content.
- **Chunking**: Character-based (LangChain `RecursiveCharacterTextSplitter`).
- **Index**: Databricks Vector Search with Delta Sync; embeddings from the configured embedding endpoint.

## Repo layout

```
pdf-ocr-rag-accelerator/
├── README.md                 # This file
├── config/
│   └── config.yml.example    # Copy to config.yml and edit
├── notebooks/
│   ├── 01-ingest-parse-pdfs.ipynb
│   └── 02-chunk-index.ipynb
├── workflows/                # Databricks Job (ingest → chunk/index)
│   ├── pdf_rag_job.json      # Job definition: 2 tasks, 1 cluster
│   └── README.md             # How to import or create the job
├── scripts/                  # Optional: run outside Databricks
│   └── run_pipeline.py
└── requirements.txt
```

## Use in Databricks

1. **Clone or import** this repo into your workspace (Repo or folder).
2. **Config**: In each notebook, set the **Config** cell to your catalog, schema, volume, table names, and Vector Search endpoint/embedding model (or load from `config/config.yml` if you add a loader cell).
3. **Run order**:
   - Run **01-ingest-parse-pdfs.ipynb** (ensure volume exists; optionally downloads sample PDFs; parses and writes to `parsed_table`).
   - Run **02-chunk-index.ipynb** (reads `parsed_table`, chunks, writes `chunked_table`, enables CDC, creates Vector Search index).
4. **Job (recommended)**: Use the predefined job so the pipeline runs in one go:
   - **Import**: Edit `workflows/pdf_rag_job.json` (set your repo path to `.../pdf-ocr-rag-accelerator/notebooks/...` and optional cluster spec), then create the job via **Workflows → Create job → JSON** or `databricks jobs create --json-file workflows/pdf_rag_job.json`.
   - **Or create manually**: Workflows → Create job → add two notebook tasks (01 then 02, task 2 depends on task 1), attach a job cluster. See **workflows/README.md** for step-by-step.
   - Configure catalog, schema, volume, and tables in each notebook’s Config cell (or add job parameters and wire them into the notebooks).

**OCR path (scanned PDFs):**  
Use the OCR notebooks in the parent folder to turn PDFs into text/images, write OCR text into a Delta table or volume, then either (a) point notebook 01 at that output (if it’s already “parsed” rows) or (b) add a small adapter notebook that reads OCR output and writes the same schema as 01 (`content`, `parser_status`, `doc_uri`, `last_modified`) so 02 can run unchanged.

## Use externally (e.g. Spark Connect, local)

1. **Config**: Copy `config/config.yml.example` to `config/config.yml` and set your values. For external runs you may set a local path for PDFs and use Spark Connect to run against a Databricks cluster.
2. **Dependencies**: `pip install -r requirements.txt`
3. **Run**:
   - **Option A**: Use **Databricks Repos** or **Sync** so the notebooks live in the workspace and run them via Databricks Jobs (no change from “Use in Databricks”).
   - **Option B**: Use `scripts/run_pipeline.py` with Spark Connect: connect to the cluster, run the same logic (read PDFs from volume or local path, parse, write to Delta; chunk; create index) so the pipeline is driven from your machine or CI.

## Config reference

| Key | Description |
|-----|-------------|
| `catalog`, `schema` | Unity Catalog catalog and schema |
| `volume_name` | UC volume name for source PDFs (path = `/Volumes/{catalog}/{schema}/{volume_name}`) |
| `parsed_table` | Full table name for parsed docs (output of notebook 01) |
| `chunked_table` | Full table name for chunked docs (output of notebook 02) |
| `vector_index_name` | Vector Search index name |
| `vector_search_endpoint` | Databricks Vector Search endpoint name |
| `embedding_endpoint` | Databricks embedding model endpoint name |
| `chunk_size`, `chunk_overlap` | Chunking parameters |

## Requirements

- **Databricks**: Unity Catalog enabled; Vector Search endpoint and embedding endpoint created; cluster with access to the volume and tables.
- **Notebook 01**: `pymupdf4llm`, `PyMuPDF` (fitz).
- **Notebook 02**: `langchain-text-splitters`; Databricks SDK for Vector Search API.

## Pushing to your Git

Use this folder as a **standalone repo** (e.g. named `pdf-ocr-rag-accelerator` on GitHub/GitLab):

1. **Create a new repo** on your Git host (GitHub, GitLab, etc.) named `pdf-ocr-rag-accelerator` (or "PDF OCR RAG Accelerator"). Do not initialize with a README if you already have one here.

2. **From this folder** (`pdf-ocr-rag-accelerator`):

   ```bash
   cd pdf-ocr-rag-accelerator
   git init
   git add .
   git commit -m "Initial commit: PDF OCR RAG accelerator"
   git remote add origin https://github.com/<your-org>/pdf-ocr-rag-accelerator.git
   git branch -M main
   git push -u origin main
   ```

   Replace `<your-org>/pdf-ocr-rag-accelerator` with your actual repo URL (SSH or HTTPS).

3. **If the accelerator lives inside another repo** (e.g. `oss-ocr`) and you want it as its own repo: copy this folder to a new directory, `cd` into it, then run the commands above (no need to `git init` again if you copied without `.git`).

## License and attribution

See repo root. For sample PDFs, the default uses [dbdemos-dataset](https://github.com/databricks-demos/dbdemos-dataset) (Databricks demos).
