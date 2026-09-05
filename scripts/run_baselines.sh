#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_DIR="${DATASET_DIR:?Set DATASET_DIR to the directory containing domain JSONL files.}"
LLM_MODEL_PATH="${LLM_MODEL_PATH:?Set LLM_MODEL_PATH to a local causal-LM directory.}"
EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:?Set EMBEDDING_MODEL_PATH to a local BGE-M3 directory.}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/baselines_$(date +%Y%m%d_%H%M%S)}"
DOMAINS="${DOMAINS:-mix}"

DOMAIN_ARGS=()
read -r -a DOMAIN_ARGS <<< "$DOMAINS"
LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

cd "$ROOT_DIR"
"$PYTHON_BIN" -m benchmark.run \
  --dataset-dir "$DATASET_DIR" \
  --llm-model-path "$LLM_MODEL_PATH" \
  --embedding-model-path "$EMBEDDING_MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --domains "${DOMAIN_ARGS[@]}" \
  --methods direct full bm25 vector hybrid hyde simple_graph \
  "${LIMIT_ARGS[@]}" \
  --top-k 5 \
  --chunk-words 350 \
  --chunk-overlap 80 \
  --max-new-tokens 128 \
  --llm-max-input-tokens 7900 \
  --embedding-backend "${EMBEDDING_BACKEND:-hf}"

echo "Baseline output: $OUTPUT_DIR"
