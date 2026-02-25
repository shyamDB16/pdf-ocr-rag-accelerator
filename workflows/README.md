# PDF OCR RAG Job

Run the full pipeline (OCR + adapter → chunk → index) as a single Databricks Job.

## Pipeline

1. **ocr_and_adapter** (`01-ocr-and-adapter.py`): Runs on a **GPU cluster**. OCRs images (HunyuanOCR or DeepSeek-OCR via Ray + vLLM), writes OCR results to Delta, then **adapts** to the parsed-docs shape (one row per document: `content`, `parser_status`, `doc_uri`, `last_modified`).
2. **chunk_index** (`02-chunk-index.py`): Runs on a **CPU cluster**. Reads the parsed-docs table, chunks with LangChain, writes chunked Delta table, enables CDC, creates/updates the Vector Search index.

## Option 1: Import job definition

1. **Edit paths** in `pdf_rag_job.json`:
   - Replace `<your-repo-folder>` with your Repos folder path (e.g. `shyam@company.com/pdf-ocr-rag-accelerator`).
   - Notebook paths:
     - `.../pdf-ocr-rag-accelerator/notebooks/01-ocr-and-adapter`
     - `.../pdf-ocr-rag-accelerator/notebooks/02-chunk-index`

2. **Optional**: Adjust job clusters (e.g. `gpu_cluster`: node type, workers; `cpu_cluster`: node type). Task 1 must use a GPU cluster for OCR.

3. **Create the job**: Workflows → Jobs → Create job → JSON, paste the contents of `pdf_rag_job.json`, then Save.

4. **Config**: Set widget values in 01 (image_dir, uc_catalog, uc_schema, parsed_table, etc.) and 02 (catalog, schema, parsed_table_suffix, endpoints) before running, or use job parameters if you wire them.

## Option 2: Create job from UI

1. Create job with two tasks.
2. **Task 1**: Notebook `01-ocr-and-adapter.py`, **GPU cluster** (e.g. g5.xlarge, 1 worker).
3. **Task 2**: Notebook `02-chunk-index.py`, **CPU cluster**; Depends on Task 1.
4. Save and run.

## Schedule (optional)

Add a schedule to refresh the pipeline when new images (or PDFs converted to images) are added to the volume.
