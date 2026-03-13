# Databricks notebook source
# MAGIC %md
# MAGIC # 1b. OCR Pipeline — PDF / Image → Parsed Docs (Serverless GPU)
# MAGIC
# MAGIC <div style="background:#e7f3fe;border-left:6px solid #2196F3;padding:12px;margin:12px 0;border-radius:4px">
# MAGIC <b>Serverless GPU variant</b><br/>
# MAGIC Same pipeline as <code>01-ocr-and-adapter.py</code> but uses <b>Databricks Serverless GPU Compute (SGC)</b> instead of a classic GPU cluster. SGC auto-provisions remote A10 GPUs — no cluster configuration needed.
# MAGIC </div>
# MAGIC
# MAGIC **Differences from classic (01):**
# MAGIC - Uses `@ray_launch` from `serverless_gpu.ray` to provision remote GPUs
# MAGIC - Calls `.distributed()` to execute on provisioned workers
# MAGIC - `num_gpus` is set explicitly (not auto-detected from cluster)
# MAGIC - No `ray.init()` — handled by `@ray_launch`
# MAGIC
# MAGIC **Setup:**
# MAGIC 1. Click **Connect** → **Serverless GPU**
# MAGIC 2. Open **Environment** side panel → set **Accelerator** to **A10**
# MAGIC 3. Click **Apply** and **Confirm**
# MAGIC
# MAGIC **Output:** Same `rag_parsed_docs` table as notebook 01 — notebooks 02 and 03 work unchanged.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install dependencies

# COMMAND ----------

# Pre-compiled Flash Attention for A10s
%pip install --no-cache-dir "torch==2.9.0+cu128" --index-url https://download.pytorch.org/whl/cu128
%pip install -U --no-cache-dir wheel ninja packaging
%pip install --force-reinstall --no-cache-dir --no-build-isolation flash-attn
%pip install einops hf_transfer
%pip install "ray[data]>=2.47.1"

# vLLM with all dependencies
%pip install "vllm==0.13.0"

# PDF and image processing
%pip install pdf2image Pillow "pymupdf>=1.23.0"

%restart_python


# COMMAND ----------

from packaging.version import Version
import torch
import flash_attn
import vllm
import ray
import transformers
from PIL import Image

print(f"PyTorch: {torch.__version__}")
print(f"Flash Attention: {flash_attn.__version__}")
print(f"vLLM: {vllm.__version__}")
print(f"Ray: {ray.__version__}")
print(f"Transformers: {transformers.__version__}")

assert Version(ray.__version__) >= Version("2.47.1"), "Ray version must be at least 2.47.1"
print("\n✓ All version checks passed!")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC <div style="background:#fff3cd;border-left:6px solid #ffc107;padding:12px;margin:12px 0;border-radius:4px">
# MAGIC <b>SGC note</b><br/>
# MAGIC <code>num_gpus</code> sets how many remote A10 GPUs <code>@ray_launch</code> provisions.
# MAGIC The notebook itself runs on a single A10 for orchestration; the heavy OCR work runs on the provisioned workers.
# MAGIC </div>

# COMMAND ----------

# --- Input mode ---
dbutils.widgets.dropdown("input_mode", "pdf", ["pdf", "images"], "Input Mode")
dbutils.widgets.text("pdf_dir", "/Volumes/main/default/ocr_data/pdfs", "PDF Directory")
dbutils.widgets.text("image_dir", "/Volumes/main/default/ocr_data/images", "Image Directory")
dbutils.widgets.dropdown("save_images", "false", ["true", "false"], "Save rendered PNGs (checkpoint)")
dbutils.widgets.text("render_dpi", "300", "Render DPI (PDF mode)")

# --- OCR model ---
dbutils.widgets.dropdown("ocr_model", "deepseek-ai/DeepSeek-OCR",
                         ["tencent/HunyuanOCR", "deepseek-ai/DeepSeek-OCR"],
                         "OCR Model")
dbutils.widgets.text("batch_size", "4", "Batch Size")
dbutils.widgets.text("num_gpus", "5", "Number of A10 GPUs to provision")
dbutils.widgets.text("hf_secret_scope", "", "HF Secret Scope (optional)")
dbutils.widgets.text("hf_secret_key", "", "HF Secret Key (optional)")

# --- Unity Catalog ---
dbutils.widgets.text("uc_catalog", "main", "UC Catalog")
dbutils.widgets.text("uc_schema", "default", "UC Schema")
dbutils.widgets.text("uc_volume", "ocr_data", "UC Volume")
dbutils.widgets.text("uc_table", "ocr_inference_results", "OCR Results Table")
dbutils.widgets.text("parsed_table", "rag_parsed_docs", "Parsed Docs Table (adapter output)")

# COMMAND ----------

import os

INPUT_MODE = dbutils.widgets.get("input_mode")
PDF_DIR = dbutils.widgets.get("pdf_dir")
IMAGE_DIR = dbutils.widgets.get("image_dir")
SAVE_IMAGES = dbutils.widgets.get("save_images") == "true"
RENDER_DPI = int(dbutils.widgets.get("render_dpi"))

OCR_MODEL = dbutils.widgets.get("ocr_model")
BATCH_SIZE = int(dbutils.widgets.get("batch_size"))
NUM_GPUS = int(dbutils.widgets.get("num_gpus"))
HF_SECRET_SCOPE = dbutils.widgets.get("hf_secret_scope")
HF_SECRET_KEY = dbutils.widgets.get("hf_secret_key")

UC_CATALOG = dbutils.widgets.get("uc_catalog")
UC_SCHEMA = dbutils.widgets.get("uc_schema")
UC_VOLUME = dbutils.widgets.get("uc_volume")
UC_TABLE = dbutils.widgets.get("uc_table")
PARSED_TABLE_SUFFIX = dbutils.widgets.get("parsed_table").strip() or "rag_parsed_docs"

UC_VOLUME_PATH = f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/{UC_VOLUME}"
UC_TABLE_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.{UC_TABLE}"
PARSED_TABLE_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.{PARSED_TABLE_SUFFIX}"
PARQUET_OUTPUT_PATH = f"{UC_VOLUME_PATH}/ocr_inference_output"

print(f"Input Mode:       {INPUT_MODE}")
if INPUT_MODE == "pdf":
    print(f"PDF Directory:    {PDF_DIR}")
    print(f"Save Images:      {SAVE_IMAGES}")
    print(f"Render DPI:       {RENDER_DPI}")
else:
    print(f"Image Directory:  {IMAGE_DIR}")
print(f"OCR Model:        {OCR_MODEL}")
print(f"Num GPUs (SGC):   {NUM_GPUS}")
print(f"Batch Size:       {BATCH_SIZE}")
print(f"\nUC Volume:        {UC_VOLUME_PATH}")
print(f"OCR Table:        {UC_TABLE_NAME}")
print(f"Parsed Table:     {PARSED_TABLE_NAME}")
print(f"Parquet Output:   {PARQUET_OUTPUT_PATH}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Model configuration

# COMMAND ----------

MODEL_CONFIGS = {
    "tencent/HunyuanOCR": {
        "llm_kwargs": dict(
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=32768,
            max_num_seqs=6,
            gpu_memory_utilization=0.90,
            mm_processor_cache_gb=0,
            enable_prefix_caching=False,
        ),
        "sampling_kwargs": dict(temperature=0, max_tokens=16384),
        "uses_processor": True,
        "logits_processors": None,
    },
    "deepseek-ai/DeepSeek-OCR": {
        "llm_kwargs": dict(
            dtype="bfloat16",
            max_model_len=8192,
            max_num_seqs=6,
            gpu_memory_utilization=0.90,
            mm_processor_cache_gb=0,
            enable_prefix_caching=False,
        ),
        "sampling_kwargs": dict(
            temperature=0.0,
            max_tokens=32700,
            extra_args=dict(
                ngram_size=30,
                window_size=90,
                whitelist_token_ids={128821, 128822},
            ),
            skip_special_tokens=False,
        ),
        "uses_processor": False,
        "logits_processors": "deepseek",
    },
}
cfg = MODEL_CONFIGS[OCR_MODEL]
print(f"✓ Loaded config for: {OCR_MODEL}")
print(f"  max_model_len={cfg['llm_kwargs']['max_model_len']}, max_tokens={cfg['sampling_kwargs']['max_tokens']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the distributed OCR pipeline (SGC)
# MAGIC
# MAGIC <div style="background:#d4edda;border-left:6px solid #28a745;padding:12px;margin:12px 0;border-radius:4px">
# MAGIC <b>How SGC works</b><br/>
# MAGIC <code>@ray_launch(gpus=N, gpu_type='a10', remote=True)</code> provisions <b>N remote A10 GPUs</b>.
# MAGIC The decorated function runs on the Ray cluster with those GPUs available.
# MAGIC Call <code>.distributed()</code> to launch — initial startup may take a few minutes as GPU nodes are provisioned and models are loaded.
# MAGIC </div>

# COMMAND ----------

from serverless_gpu.ray import ray_launch
import io
import re
import glob
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image
from vllm import LLM, SamplingParams


@ray_launch(gpus=NUM_GPUS, gpu_type='a10', remote=True)
def run_ocr_pipeline():
    """Distributed OCR pipeline on Serverless GPU — streaming PDF or image input."""

    _model_name = OCR_MODEL
    _cfg = MODEL_CONFIGS[_model_name]

    # --- Page renderer (PDF mode) ---
    def make_render_fn(dpi, save_images, image_output_dir):
        import fitz

        zoom = dpi / 72.0
        mat_val = (zoom, zoom)

        def render_pages(row):
            pdf_bytes = row["bytes"]
            pdf_path = row["path"]
            doc_stem = Path(pdf_path).stem
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            mat = fitz.Matrix(*mat_val)
            results = []

            for page_idx in range(len(doc)):
                page = doc[page_idx]
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")

                if save_images:
                    out_path = os.path.join(image_output_dir, f"{doc_stem}_page_{page_idx + 1}.png")
                    pix.save(out_path)

                results.append({
                    "image_bytes": img_bytes,
                    "doc_uri": doc_stem,
                    "page_num": page_idx + 1,
                })

            doc.close()
            return results

        return render_pages

    # --- Image normalizer (image mode) ---
    def normalize_image_row(row):
        path = row["path"]
        stem = Path(path).stem
        match = re.match(r"(.+)_page_(\d+)$", stem)
        doc_uri = match.group(1) if match else stem
        page_num = int(match.group(2)) if match else 1

        with open(path, "rb") as f:
            image_bytes = f.read()

        return {"image_bytes": image_bytes, "doc_uri": doc_uri, "page_num": page_num}

    # --- Post-processing ---
    def _clean_repeated_substrings(text):
        n = len(text)
        if n < 8000:
            return text
        for length in range(2, n // 10 + 1):
            candidate = text[-length:]
            count, i = 0, n - length
            while i >= 0 and text[i : i + length] == candidate:
                count += 1
                i -= length
            if count >= 10:
                return text[: n - length * (count - 1)]
        return text

    def _postprocess(text):
        if _model_name == "tencent/HunyuanOCR":
            return _clean_repeated_substrings(text)
        return text

    # --- OCR predictor ---
    class OCRPredictor:
        def __init__(self):
            llm_kw = dict(model=_model_name, **_cfg["llm_kwargs"])
            if _cfg["logits_processors"] == "deepseek":
                from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
                llm_kw["logits_processors"] = [NGramPerReqLogitsProcessor]
            self.llm = LLM(**llm_kw)
            self.sampling_params = SamplingParams(**_cfg["sampling_kwargs"])
            self.model_name = _model_name
            self.processor = None
            if _cfg["uses_processor"]:
                from transformers import AutoProcessor
                self.processor = AutoProcessor.from_pretrained(_model_name)
            print(f"✓ OCRPredictor initialized with {_model_name}")

        def _build_input(self, image_bytes):
            img = Image.open(io.BytesIO(image_bytes))
            if self.model_name == "tencent/HunyuanOCR":
                user_prompt = (
                    "Extract all information from the main body of the document image "
                    "and represent it in markdown format, ignoring headers and footers. "
                    "Tables should be expressed in HTML format, formulas in the document "
                    "should be represented using LaTeX format, and the parsing should be "
                    "organized according to the reading order."
                )
                messages = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": user_prompt},
                    ]},
                ]
                prompt = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                return {"prompt": prompt, "multi_modal_data": {"image": [img]}}
            else:
                img = img.convert("RGB")
                prompt = "<image>\nConvert the document to markdown. Think Carefully"
                return {"prompt": prompt, "multi_modal_data": {"image": img}}

        def __call__(self, batch):
            image_bytes_list = batch["image_bytes"].tolist()
            doc_uris = batch["doc_uri"].tolist()
            page_nums = batch["page_num"].tolist()

            inputs = [self._build_input(b) for b in image_bytes_list]
            outputs = self.llm.generate(inputs, self.sampling_params)

            ocr_texts = [_postprocess(o.outputs[0].text) for o in outputs]
            ts = datetime.now().isoformat()

            return {
                "doc_uri": doc_uris,
                "page_num": page_nums,
                "ocr_text": ocr_texts,
                "model": [self.model_name] * len(ocr_texts),
                "timestamp": [ts] * len(ocr_texts),
            }

    # ---- Build the Ray Dataset ----
    if INPUT_MODE == "pdf":
        print(f"📄 PDF mode — reading from {PDF_DIR}")
        ds = ray.data.read_binary_files(PDF_DIR, include_paths=True)
        ds = ds.filter(lambda row: row["path"].lower().endswith(".pdf"))
        n_pdfs = ds.count()
        print(f"  Found {n_pdfs} PDF(s)")

        if SAVE_IMAGES:
            os.makedirs(IMAGE_DIR, exist_ok=True)
            print(f"  Image checkpoint enabled → {IMAGE_DIR}")

        render_fn = make_render_fn(
            dpi=RENDER_DPI,
            save_images=SAVE_IMAGES,
            image_output_dir=IMAGE_DIR,
        )
        ds = ds.flat_map(render_fn)
    else:
        print(f"🖼️  Image mode — reading PNGs from {IMAGE_DIR}")
        image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
        if not image_paths:
            raise FileNotFoundError(f"No PNG images found in {IMAGE_DIR}")
        print(f"  Found {len(image_paths)} image(s)")
        ds = ray.data.read_binary_files(
            [p for p in image_paths], include_paths=True
        )
        ds = ds.map(normalize_image_row)

    # ---- OCR inference (GPU) ----
    ds = ds.map_batches(
        OCRPredictor,
        concurrency=(1, NUM_GPUS),
        batch_size=BATCH_SIZE,
        num_gpus=1,
        num_cpus=12,
    )

    # ---- Write results ----
    print(f"\nWriting results to {PARQUET_OUTPUT_PATH}")
    ds.write_parquet(PARQUET_OUTPUT_PATH, mode="overwrite")
    print("✓ Parquet written")

    # ---- Preview ----
    samples = ray.data.read_parquet(PARQUET_OUTPUT_PATH).take(limit=5)
    print("\n" + "=" * 60)
    print("SAMPLE OCR RESULTS")
    print("=" * 60)
    for i, row in enumerate(samples):
        doc = row.get("doc_uri", "?")
        pg = row.get("page_num", "?")
        text = (row.get("ocr_text", "") or "")[:200]
        print(f"[{i+1}] {doc} p.{pg}: {text!r}...")

    return PARQUET_OUTPUT_PATH

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run distributed OCR on Serverless GPU
# MAGIC
# MAGIC <div style="background:#fff3cd;border-left:6px solid #ffc107;padding:12px;margin:12px 0;border-radius:4px">
# MAGIC <b>Startup time</b><br/>
# MAGIC Initial launch may take a few minutes as remote A10 GPU nodes are provisioned and models are loaded. Subsequent runs on a warm cluster are faster.
# MAGIC </div>

# COMMAND ----------

parquet_path = run_ocr_pipeline.distributed()
print(f"\n✓ OCR inference complete → {parquet_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save as Delta table in Unity Catalog

# COMMAND ----------

print(f"Loading Parquet from: {PARQUET_OUTPUT_PATH}")
df_spark = spark.read.parquet(PARQUET_OUTPUT_PATH)

print(f"✓ Loaded {df_spark.count()} rows")
df_spark.printSchema()

df_spark.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(UC_TABLE_NAME)

print(f"✓ Delta table created: {UC_TABLE_NAME}")
display(df_spark.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Adapter: OCR table → parsed-docs shape
# MAGIC
# MAGIC Aggregates page-level OCR rows into one row per document — same output as notebook 01 (classic). Notebooks 02 and 03 work unchanged.

# COMMAND ----------

from pyspark.sql import functions as F

ocr_df = spark.table(UC_TABLE_NAME)
parsed_df = (
    ocr_df.withColumn("page_num", F.col("page_num").cast("int"))
    .groupBy("doc_uri")
    .agg(
        F.expr(
            "concat_ws('\n\n', transform(sort_array(collect_list(struct(page_num, ocr_text))), x -> x.ocr_text))"
        ).alias("content"),
        F.lit("SUCCESS").alias("parser_status"),
        F.max("timestamp").cast("timestamp").alias("last_modified"),
    )
)

parsed_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(PARSED_TABLE_NAME)
n_docs = spark.table(PARSED_TABLE_NAME).count()
print(f"✓ Adapter: wrote {n_docs} document(s) to {PARSED_TABLE_NAME}")
display(spark.table(PARSED_TABLE_NAME))
