#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_DIR="${DATASET_DIR:?Set DATASET_DIR to the directory containing domain JSONL files.}"
LLM_MODEL_PATH="${LLM_MODEL_PATH:?Set LLM_MODEL_PATH to a local causal-LM directory.}"
EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:?Set EMBEDDING_MODEL_PATH to a local BGE-M3 directory.}"

cd "$ROOT_DIR"
"$PYTHON_BIN" -m benchmark.run \
  --dataset-dir "$DATASET_DIR" \
  --llm-model-path "$LLM_MODEL_PATH" \
  --embedding-model-path "$EMBEDDING_MODEL_PATH" \
  --output-dir "${OUTPUT_DIR:-${ROOT_DIR}/outputs/smoke}" \
  --domains "${DOMAIN:-mix}" \
  --limit "${LIMIT:-3}" \
  --methods sg_rag \
  --top-k 5 \
  --embedding-backend "${EMBEDDING_BACKEND:-hf}"
