# PDF RAG Job

Run the full pipeline (ingest → parse → chunk → index) as a single Databricks Job.

## Option 1: Import job definition

1. **Edit paths** in `pdf_rag_job.json`:
   - Replace `<your-repo-folder>` with your Repos folder path (e.g. `shyam@company.com/pdf-ocr-rag-accelerator` or the folder name where you cloned the repo).
   - Notebook paths must be the full workspace paths to the notebooks under your repo, e.g.:
     - `/Repos/<your-repo-folder>/pdf-ocr-rag-accelerator/notebooks/01-ingest-parse-pdfs`
     - `/Repos/<your-repo-folder>/pdf-ocr-rag-accelerator/notebooks/02-chunk-index`

2. **Optional**: Adjust `job_clusters` (e.g. `spark_version`, `node_type_id`, `num_workers`) for your workspace.

3. **Create the job**:
   - **UI**: Workflows → Jobs → Create job → **Switch to JSON** (or Import JSON), paste the contents of `pdf_rag_job.json`, then Save.
   - **CLI**: `databricks jobs create --json-file workflows/pdf_rag_job.json`

4. **Config**: Set catalog, schema, volume, and table names in the **Config** cell of each notebook (01 and 02) before running the job, or use job parameters and pass them into the notebooks if you add parameter support.

## Option 2: Create job from UI (no JSON)

1. Workflows → Jobs → Create job.
2. Add **Task 1**: Notebook task → select `pdf-ocr-rag-accelerator/notebooks/01-ingest-parse-pdfs.ipynb` from your Repo.
3. Add **Task 2**: Notebook task → select `pdf-ocr-rag-accelerator/notebooks/02-chunk-index.ipynb`; set **Depends on** → Task 1.
4. Create a **Job cluster** (e.g. single-node Photon, `i3.xlarge`) and assign it to both tasks.
5. Save and run.

## Schedule (optional)

In the job settings, add a **Schedule** (e.g. daily or weekly) to refresh the parsed/chunked tables and Vector Search index from new PDFs in the volume.
