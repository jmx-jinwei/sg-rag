# SG-RAG

SG-RAG is a small-model-friendly retrieval-augmented generation method based on per-chunk Skill Cards, a deterministic document-local Skill Graph, multilayer retrieval, and sentence-level evidence backtracking.

This repository contains the SG-RAG implementation and the lightweight baselines used for controlled comparisons in the accompanying paper. It is designed for local inference with a causal language model and a BGE-M3-compatible embedding model.

## Method Overview

SG-RAG avoids one-shot LLM entity and relation extraction. Instead, it builds an evidence-preserving index from each document:

```text
Document -> overlapping chunks -> Skill Cards
                                -> topic and entity-like terms
                                -> event/state sentences
                                -> local lexical edges
                                -> sentence-level evidence records
```

For each question, retrieval combines:

1. BM25 chunk scores.
2. Dense BGE-M3 chunk scores.
3. Multiple query variants with reciprocal-rank fusion.
4. Skill Card node matching and local edge expansion.
5. Event/state compatibility signals.
6. Sentence-level dense evidence reranking and context-window backtracking.

Selected evidence sentences are restored with a bounded neighboring-sentence window and supplied through an evidence-only answer prompt. The answer-detail controller adapts the requested response style to question types such as reason, person, temporal, comparison, and transformation-chain questions. The released implementation preserves the exact fixed English lexical resources and optional rule-based answer-compression step used by the reported configuration. These rules are not learned and are not claimed to transfer unchanged across languages or genres; disable the compression step with `--no-skillgraph-answer-compression` when evaluating that dependency.

## Repository Layout

```text
SG-RAG/
├── sg_rag/
│   ├── method.py       # SG-RAG algorithm
│   ├── models.py       # local causal LM and embedding adapters
│   ├── retrieval.py    # retrieval contracts and BM25 primitive
│   ├── prompts.py      # shared baseline prompt
│   └── utils.py        # tokenization, chunking, scaling, answer helpers
├── baselines/
│   └── methods.py      # direct, full, BM25, vector, hybrid, HyDE, SimpleGraph
├── benchmark/
│   ├── run.py          # command-line benchmark runner
│   ├── dataset.py      # JSONL loading and normalization
│   ├── metrics.py      # answer and retrieval metrics
│   ├── output.py       # wall-clock runtime metadata
├── configs/example.env
├── scripts/
├── tests/
├── requirements.txt
└── run_sg_rag.py
```

The repository intentionally excludes datasets, model weights, generated predictions, paper result tables, API credentials, and machine-specific paths. LightRAG and MiniRAG source trees are also excluded; their official repositories should be used separately when those baselines are needed.

## Requirements

- Python 3.10 or newer.
- A CUDA-capable GPU is recommended for the local Llama-style model and BGE-M3.
- A local causal language model compatible with `transformers.AutoModelForCausalLM`.
- A local BGE-M3 embedding model. The default `hf` backend uses `transformers`; the optional `FlagEmbedding` package enables the native BGE-M3 backend.
- A directory of domain JSONL files in the format described below.

Install the public dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

The project can also be installed as an editable package:

```bash
python -m pip install -e .
```

For the native BGE-M3 backend, install the optional dependency separately:

```bash
python -m pip install FlagEmbedding
```

## Input Data

The runner expects one JSONL file per domain, for example `mix.jsonl` or `agriculture.jsonl`. Each row must contain `input` or `question`, and may contain `answers`, `answer`, or `gold_answer`, plus `context`:

```json
{"_id":"example-1","input":"Who discovered the method?","answers":["Alice"],"context":"Alice discovered the method in 2019."}
```

Additional fields are preserved in the output under `row_extra`.

## Run SG-RAG

The easiest server workflow is to export three local paths and call the script:

```bash
export DATASET_DIR=/path/to/UltraDomain
export LLM_MODEL_PATH=/path/to/Meta-Llama-3-8B-Instruct
export EMBEDDING_MODEL_PATH=/path/to/bge-m3
export DOMAINS="mix agriculture"
export OUTPUT_DIR=outputs/sg_rag_agriculture
bash scripts/run_sg_rag.sh
```

The equivalent direct command is:

```bash
python -m benchmark.run \
  --dataset-dir /path/to/UltraDomain \
  --llm-model-path /path/to/Meta-Llama-3-8B-Instruct \
  --embedding-model-path /path/to/bge-m3 \
  --output-dir outputs/sg_rag \
  --domains mix \
  --methods sg_rag \
  --top-k 10 \
  --chunk-words 350 \
  --chunk-overlap 80 \
  --max-new-tokens 128 \
  --llm-max-input-tokens 7900 \
  --embedding-backend hf \
  --skillgraph-evidence-per-chunk 4
```

For a short smoke run, set `LIMIT=3` and use `bash scripts/run_smoke.sh`.

## Baseline Comparisons

The retained lightweight baselines are selected with `--methods`:

```text
direct          question-only generation
full            full-context generation
bm25            lexical chunk retrieval
vector          dense BGE-M3 chunk retrieval
hybrid          weighted BM25 and dense retrieval
hyde            hypothetical-document dense retrieval
simple_graph    Keyword-Graph RAG: heuristic local graph retrieval with BM25 fallback
sg_rag          SG-RAG (skillgraph is accepted as a compatibility alias)
```

Run all retained baselines with:

```bash
export DATASET_DIR=/path/to/UltraDomain
export LLM_MODEL_PATH=/path/to/Meta-Llama-3-8B-Instruct
export EMBEDDING_MODEL_PATH=/path/to/bge-m3
export DOMAINS=mix
bash scripts/run_baselines.sh
```

The paper comparison uses `top-k=5` for the retained chunk-retrieval baselines. `scripts/run_baselines.sh` applies this setting; SG-RAG uses `top-k=10`.

## Ablation Experiments

The ablation script runs the complete configuration and three component-removal variants. To reproduce the complete 20-domain ablation, use `all`, which selects every JSONL domain file in `DATASET_DIR`:

```bash
export DOMAINS=all
bash scripts/run_ablation.sh
```

The variants disable one component at a time:

```text
full                       complete SG-RAG configuration
no_skill_card_graph        disables Skill Card nodes and local graph signals
no_multilayer_retrieval    disables query variants and multilayer fusion
no_adaptive_answer_detail  uses the legacy short-answer instruction
```

The complete method uses all three components. The script keeps it and each ablation in separate output directories.

## Outputs and Timing

Each run writes:

```text
<output>/run_config.json
<output>/run_metadata.json
<output>/summary_metrics.csv
<output>/<method>/<domain>_predictions.jsonl
```

`run_metadata.json` records `started_at`, `finished_at`, `elapsed_sec`, status, selected methods, domains, and sample limit. `elapsed_sec` is the wall-clock duration of the benchmark process, including model loading and retrieval/generation, but excluding work performed outside the runner. It is separate from the per-sample `timing.query_time_sec` and `timing.generation_time_sec` fields in each JSONL record. Final metric evaluation is performed in-process and is not reported as the model's generation time.

SG-RAG retrieval metadata additionally records `build_time_sec`, Skill Card count, sentence count, node count, edge count, active retrieval layers, query variants, and ablation flags.

## Important Parameters

The paper-oriented defaults are:

```text
top_k                         10
chunk_words                   350
chunk_overlap                 80
max_new_tokens                128
llm_max_input_tokens          7900
embedding_backend             hf
skillgraph_evidence_per_chunk 4
skillgraph_sentence_dense     true
skillgraph_adaptive_answer_detail true
skillgraph_answer_compression true
```

Use `python -m benchmark.run --help` for the full list of model, retrieval, graph, evidence, answer, and quality-gate parameters.

## Reproducibility Notes

- Keep the model paths, GPU type, CUDA version, batch size, context limits, and sample limit fixed when comparing methods.
- Compare `run_metadata.json` for whole-run wall-clock time and the per-record timing fields for average query latency; these answer different questions.
- The default embedding backend is `hf`. Use the same backend for every method in a comparison.
- Full experiments can require substantial GPU memory, especially for long contexts and large local language models.
- The retrieval metrics are answer-token-overlap proxies, not manually annotated passage relevance judgments.

## Development Checks

Run the tests and compile checks before publishing:

```bash
python -m pytest -q
python -m compileall -q sg_rag baselines benchmark run_sg_rag.py
python -m benchmark.run --help
```

No model or dataset download is triggered by the help command.

## License

This project is released under the MIT License. See `LICENSE`.
