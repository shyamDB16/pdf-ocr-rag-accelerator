# Databricks notebook source
# MAGIC %md
# MAGIC # OCR + Adapter for RAG (Solution Accelerator)
# MAGIC
# MAGIC This notebook runs an OCR pipeline on a **classic Databricks GPU cluster**: with OSS models (HunyuanOCR / DeepSeek-OCR), `MODEL_CONFIGS`, vLLM `multi_modal_data` API, AutoProcessor for HunyuanOCR, and post-processing.
# MAGIC
# MAGIC **What you'll learn:**
# MAGIC - Set up Ray and vLLM for distributed OCR with the reference’s model configs and inference path
# MAGIC - Use Ray Data to batch process images across multiple GPUs
# MAGIC - Save results to Parquet in Unity Catalog Volumes and convert to Delta tables
# MAGIC
# MAGIC **Use case:** Batch OCR on classic GPU clusters.
# MAGIC
# MAGIC **Next:** Run `02-chunk-index.ipynb` to chunk the parsed-docs table and build the vector index.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster Setup
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC 1. Create a Databricks cluster with GPU instances (e.g., `g5.xlarge`, `g4dn.xlarge`)
# MAGIC 2. Ensure cluster has multiple worker nodes for distributed processing
# MAGIC 3. Use Databricks Runtime 14.3 LTS or later (Ray support)
# MAGIC
# MAGIC **Cluster Configuration Example:**
# MAGIC - Node Type: `g5.xlarge` (1 GPU per node)
# MAGIC - Workers: 4 nodes (4 GPUs total)
# MAGIC - Auto-scaling: Enabled (min: 1, max: 4)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install dependencies
# MAGIC
# MAGIC Install all required packages for distributed Ray and vLLM inference with multimodal support.
# MAGIC

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
# MAGIC ## Initialize Ray Connection
# MAGIC
# MAGIC Connect to the existing Ray cluster on your Databricks cluster.
# MAGIC

# COMMAND ----------

import os
import json

# Set Ray temp directory
os.environ['RAY_TEMP_DIR'] = '/tmp/ray'

# Initialize Ray connection to existing cluster
if not ray.is_initialized():
    ray.init(
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"RAY_TEMP_DIR": "/tmp/ray"}}
    )
    print("✓ Ray initialized")
else:
    print("✓ Ray already initialized")

# Display cluster resources
def print_ray_resources():
    """Print Ray cluster resources and GPU allocation per node."""
    try:
        cluster_resources = ray.cluster_resources()
        print("\nRay Cluster Resources:")
        print(json.dumps(cluster_resources, indent=2))

        nodes = ray.nodes()
        print(f"\nDetected {len(nodes)} Ray node(s):")
        
        for node in nodes:
            node_id = node.get("NodeID", "N/A")[:8]
            ip_address = node.get("NodeManagerAddress", "N/A")
            resources = node.get("Resources", {})
            num_gpus = int(resources.get("GPU", 0))
            print(f"  • Node {node_id}... | IP: {ip_address} | GPUs: {num_gpus}")
            
            # Show specific GPU IDs if available
            gpu_ids = [k for k in resources.keys() if k.startswith("GPU_ID_")]
            if gpu_ids:
                print(f"    GPU IDs: {', '.join(gpu_ids)}")
    except Exception as e:
        print(f"Error querying Ray cluster: {e}")

print_ray_resources()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC Choose OCR model (HunyuanOCR or DeepSeek-OCR), image directory, and Unity Catalog output paths.
# MAGIC

# COMMAND ----------

# Widget configuration (matches sgc-vllm-ocr.ipynb reference)
dbutils.widgets.dropdown("ocr_model", "deepseek-ai/DeepSeek-OCR",
                         ["tencent/HunyuanOCR", "deepseek-ai/DeepSeek-OCR"],
                         "OCR Model")
dbutils.widgets.text("image_dir", "/Volumes/main/default/ocr_data/images", "Image Directory")
dbutils.widgets.text("hf_secret_scope", "", "HF Secret Scope")
dbutils.widgets.text("hf_secret_key", "", "HF Secret Key")
dbutils.widgets.text("batch_size", "4", "Batch Size")

# Unity Catalog configuration
dbutils.widgets.text("uc_catalog", "main", "UC Catalog")
dbutils.widgets.text("uc_schema", "default", "UC Schema")
dbutils.widgets.text("uc_volume", "ocr_data", "UC Volume")
dbutils.widgets.text("uc_table", "ocr_inference_results", "UC Table")
dbutils.widgets.text("parsed_table", "rag_parsed_docs", "Parsed docs table (adapter output)")



# COMMAND ----------

# Retrieve widget values
OCR_MODEL = dbutils.widgets.get("ocr_model")
IMAGE_DIR = dbutils.widgets.get("image_dir")
HF_SECRET_SCOPE = dbutils.widgets.get("hf_secret_scope")
HF_SECRET_KEY = dbutils.widgets.get("hf_secret_key")
BATCH_SIZE = int(dbutils.widgets.get("batch_size"))

# Unity Catalog paths
UC_CATALOG = dbutils.widgets.get("uc_catalog")
UC_SCHEMA = dbutils.widgets.get("uc_schema")
UC_VOLUME = dbutils.widgets.get("uc_volume")
UC_TABLE = dbutils.widgets.get("uc_table")
PARSED_TABLE_SUFFIX = dbutils.widgets.get("parsed_table").strip() or "rag_parsed_docs"

# Construct paths
UC_VOLUME_PATH = f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/{UC_VOLUME}"
UC_TABLE_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.{UC_TABLE}"
PARSED_TABLE_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.{PARSED_TABLE_SUFFIX}"
PARQUET_OUTPUT_PATH = f"{UC_VOLUME_PATH}/ocr_inference_output"

# Detect available GPUs from Ray cluster
cluster_resources = ray.cluster_resources()
available_gpus = int(cluster_resources.get("GPU", 0))
NUM_GPUS = available_gpus if available_gpus > 0 else 1

print(f"OCR Model:        {OCR_MODEL}")
print(f"Image Directory: {IMAGE_DIR}")
print(f"Available GPUs:   {NUM_GPUS} (auto-detected)")
print(f"Batch Size:      {BATCH_SIZE}")
print(f"\nUnity Catalog:   {UC_VOLUME_PATH}")
print(f"Table Name:      {UC_TABLE_NAME}")
print(f"Parsed Table:    {PARSED_TABLE_NAME}")
print(f"Parquet Output:  {PARQUET_OUTPUT_PATH}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Model configuration 
# MAGIC
# MAGIC Per-model settings for vLLM and sampling. HunyuanOCR uses AutoProcessor for the chat template; DeepSeek-OCR uses a raw `<image>` prompt and an n-gram logits processor.

# COMMAND ----------

# Per-model configuration 
MODEL_CONFIGS = {
    "tencent/HunyuanOCR": {
        "llm_kwargs": dict(
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=32768,
            max_num_seqs=6,
            gpu_memory_utilization=0.80,
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
            gpu_memory_utilization=0.80,
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
# MAGIC ## Define the distributed OCR inference task
# MAGIC
# MAGIC Same pattern as **sgc-vllm-ocr.ipynb**: `OCRPredictor` uses vLLM `prompt` + `multi_modal_data`, model-specific config, and post-processing. Runs directly on the Ray cluster (no `@ray_launch`).
# MAGIC

# COMMAND ----------

from typing import Dict, List
from datetime import datetime
import os
import glob
import numpy as np
from vllm import LLM, SamplingParams
from PIL import Image

def run_distributed_ocr_inference():
    """Run distributed OCR inference (same logic as sgc-vllm-ocr.ipynb)."""
    _model_name = OCR_MODEL
    _cfg = MODEL_CONFIGS[_model_name]

    def _clean_repeated_substrings(text: str) -> str:
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

    def _postprocess(text: str) -> str:
        if _model_name == "tencent/HunyuanOCR":
            return _clean_repeated_substrings(text)
        return text

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

        def _build_input(self, img_path: str) -> dict:
            img = Image.open(img_path)
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
                        {"type": "image", "image": img_path},
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

        def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
            image_paths_batch = batch["image_path"].tolist()
            inputs = [self._build_input(p) for p in image_paths_batch]
            outputs = self.llm.generate(inputs, self.sampling_params)
            result_paths, result_texts, result_models, result_timestamps = [], [], [], []
            for img_path, output in zip(image_paths_batch, outputs):
                raw_text = output.outputs[0].text
                result_paths.append(img_path)
                result_texts.append(_postprocess(raw_text))
                result_models.append(self.model_name)
                result_timestamps.append(datetime.now().isoformat())
            return {
                "image_path": result_paths,
                "ocr_text": result_texts,
                "model": result_models,
                "timestamp": result_timestamps,
            }

    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in {IMAGE_DIR}")

    ds = ray.data.from_items([{"image_path": p} for p in image_paths])
    print(f"✓ Created Ray dataset with {ds.count()} images from {IMAGE_DIR}")

    num_instances = NUM_GPUS
    ds = ds.map_batches(
        OCRPredictor,
        concurrency=(1, num_instances),
        batch_size=BATCH_SIZE,
        num_gpus=1,
        num_cpus=12,
    )

    print(f"\nWriting results to Parquet: {PARQUET_OUTPUT_PATH}")
    ds.write_parquet(PARQUET_OUTPUT_PATH, mode="overwrite")
    print("✓ Parquet files written successfully")

    sample_outputs = ray.data.read_parquet(PARQUET_OUTPUT_PATH).take(limit=5)
    print("\n" + "=" * 60)
    print("SAMPLE OCR RESULTS")
    print("=" * 60 + "\n")
    for i, output in enumerate(sample_outputs):
        img_path = output.get("image_path", "")
        ocr_text = output.get("ocr_text", "")
        display_text = ocr_text[:200] if ocr_text else "N/A"
        print(f"[{i+1}] Image: {os.path.basename(img_path)}")
        print(f"    OCR: {display_text!r}...\n")
    return PARQUET_OUTPUT_PATH


# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# Run distributed OCR inference (direct call, no .distributed())
parquet_path = run_distributed_ocr_inference()
print(f"\n✓ OCR inference complete! Results saved to: {parquet_path}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## PDF to Image Conversion (Helper Functions)
# MAGIC
# MAGIC If you have PDFs, convert them to images first. These helper functions can be used in a separate preprocessing step.
# MAGIC

# COMMAND ----------

from pdf2image import convert_from_path
import fitz  # PyMuPDF
from pathlib import Path
import os
INPUT_IMAGE_PATH = IMAGE_DIR

def pdf_to_images_pymupdf(pdf_path: str, output_dir: str, dpi: int = 300) -> list:
    """Convert PDF to images using PyMuPDF (no external dependencies like poppler)."""
    doc = fitz.open(pdf_path)
    image_paths = []
    
    pdf_name = Path(pdf_path).stem
    zoom = dpi / 72.0  # PyMuPDF uses 72 DPI as base
    mat = fitz.Matrix(zoom, zoom)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=mat)
        image_path = os.path.join(output_dir, f"{pdf_name}_page_{page_num+1}.png")
        pix.save(image_path)
        image_paths.append(image_path)
    
    doc.close()
    return image_paths

# Example usage (set pdf_path and run if you have PDFs to convert):
# pdf_path = "/Volumes/<catalog>/<schema>/<volume>/your_doc.pdf"
# image_paths = pdf_to_images_pymupdf(pdf_path, INPUT_IMAGE_PATH)
# print(f"Converted {len(image_paths)} pages from PDF")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Parquet and convert to Delta table
# MAGIC
# MAGIC Load the Parquet output from Unity Catalog Volumes using Spark and save as a Delta table for efficient querying and governance.
# MAGIC

# COMMAND ----------

# Load Parquet data using Spark
print(f"📖 Loading Parquet from: {PARQUET_OUTPUT_PATH}")
df_spark = spark.read.parquet(PARQUET_OUTPUT_PATH)

# Show schema and row count
print(f"\n✓ Loaded {df_spark.count()} rows")
print("\nSchema:")
df_spark.printSchema()

# Display sample rows
print("\nSample Results:")
display(df_spark.limit(10))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Save as Delta table in Unity Catalog
# MAGIC
# MAGIC Write the OCR results to a Unity Catalog Delta table for governance, performance, and versioning.
# MAGIC

# COMMAND ----------

# Write to Unity Catalog Delta table
print(f"Writing to Delta table: {UC_TABLE_NAME}")

df_spark.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(UC_TABLE_NAME)

print(f"✓ Delta table created successfully: {UC_TABLE_NAME}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Adapter: OCR table → parsed-docs shape
# MAGIC
# MAGIC The OCR Delta table has **one row per page** (`image_path`, `ocr_text`, `model`, `timestamp`). Downstream chunk + vector-index expects **one row per document** with `content`, `parser_status`, `doc_uri`, `last_modified`. This cell aggregates by document: derives `doc_uri` from `image_path` (e.g. `mydoc_page_1.png` → doc base name), concatenates `ocr_text` in page order, and writes the parsed-docs table.

# COMMAND ----------

from pyspark.sql import functions as F

# OCR table: image_path, ocr_text, model, timestamp
# Derive doc_uri by stripping _page_N.png; extract page number to preserve order when concatenating
ocr_df = spark.table(UC_TABLE_NAME)
parsed_df = (
    ocr_df.withColumn("doc_uri", F.regexp_replace(F.col("image_path"), r"_page_\d+\.png$", ""))
    .withColumn("page_num", F.regexp_extract(F.col("image_path"), r"_page_(\d+)\.png$", 1).cast("int"))
    .groupBy("doc_uri")
    .agg(
        F.expr("concat_ws('\n\n', transform(sort_array(collect_list(struct(page_num, ocr_text))), x -> x.ocr_text))").alias("content"),
        F.lit("SUCCESS").alias("parser_status"),
        F.max("timestamp").cast("timestamp").alias("last_modified"),
    )
)

parsed_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(PARSED_TABLE_NAME)
print(f"Adapter: wrote {parsed_df.count()} documents to {PARSED_TABLE_NAME}")
display(parsed_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monitor Ray Dashboard
# MAGIC
# MAGIC Access the Ray dashboard to monitor GPU utilization, task distribution, and performance metrics.
# MAGIC