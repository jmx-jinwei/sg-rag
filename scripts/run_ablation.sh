#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_DIR="${DATASET_DIR:?Set DATASET_DIR to the directory containing domain JSONL files.}"
LLM_MODEL_PATH="${LLM_MODEL_PATH:?Set LLM_MODEL_PATH to a local causal-LM directory.}"
EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:?Set EMBEDDING_MODEL_PATH to a local BGE-M3 directory.}"
ROOT_OUT="${ROOT_OUT:-${ROOT_DIR}/outputs/ablation_$(date +%Y%m%d_%H%M%S)}"
DOMAINS="${DOMAINS:-all}"

DOMAIN_ARGS=()
read -r -a DOMAIN_ARGS <<< "$DOMAINS"
LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

run_variant() {
  local name="$1"
  shift
  local output_dir="$ROOT_OUT/$name"
  echo "==== SG-RAG ablation: $name ===="
  cd "$ROOT_DIR"
  "$PYTHON_BIN" -m benchmark.run \
    --dataset-dir "$DATASET_DIR" \
    --llm-model-path "$LLM_MODEL_PATH" \
    --embedding-model-path "$EMBEDDING_MODEL_PATH" \
    --output-dir "$output_dir" \
    --domains "${DOMAIN_ARGS[@]}" \
    --methods sg_rag \
    "${LIMIT_ARGS[@]}" \
    --top-k 10 \
    --chunk-words 350 \
    --chunk-overlap 80 \
    --max-new-tokens 128 \
    --llm-max-input-tokens 7900 \
    --embedding-backend "${EMBEDDING_BACKEND:-hf}" \
    --skillgraph-evidence-per-chunk 4 \
    "$@"
}

run_variant full
run_variant no_skill_card_graph --no-skillgraph-use-skill-card-graph
run_variant no_multilayer_retrieval --no-skillgraph-use-multilayer-retrieval
run_variant no_adaptive_answer_detail --no-skillgraph-adaptive-answer-detail

echo "Ablation outputs: $ROOT_OUT"
