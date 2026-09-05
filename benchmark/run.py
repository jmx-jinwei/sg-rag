"""Run SG-RAG and selected lightweight baselines on UltraDomain-compatible data.

This public runner contains only the selected local methods.
Use `python -m benchmark.run --help` for the complete command-line interface.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - allows help and core tests without ML extras
    torch = None

try:
    from transformers import __version__ as transformers_version
except ImportError:  # pragma: no cover - reported in run metadata
    transformers_version = "not-installed"

from baselines.methods import (
    BM25Method,
    DirectMethod,
    FullContextMethod,
    HybridMethod,
    HyDEMethod,
    SimpleGraphMethod,
    VectorMethod,
)
from benchmark.metrics import answer_quality_metrics, retrieval_proxy_metrics
from benchmark.dataset import load_domain_rows, select_domains
from benchmark.output import write_run_metadata
from sg_rag.method import SGRAGMethod
from sg_rag.retrieval import BaseMethod, RetrievalResult
from sg_rag.utils import (
    approx_word_count,
    best_qa_prf_detail,
    postprocess_short_answer,
    stable_hash,
)


def make_method(args: argparse.Namespace, method: str) -> BaseMethod:
    needs_llm = method in {"direct", "full", "bm25", "vector", "hybrid", "hyde", "simple_graph", "sg_rag", "skillgraph"}
    needs_embedder = method in {"vector", "hybrid", "hyde", "sg_rag", "skillgraph"}
    if needs_llm or needs_embedder:
        from sg_rag.models import LocalCausalLLM, LocalEmbeddingModel
    else:
        LocalCausalLLM = None
        LocalEmbeddingModel = None
    llm = (
            LocalCausalLLM(
            args.llm_model_path,
            max_new_tokens=args.max_new_tokens,
            max_input_tokens=args.llm_max_input_tokens,
            dtype=args.llm_dtype,
            device_map=args.device_map,
        )
        if needs_llm
        else None
    )
    embedder = (
            LocalEmbeddingModel(
            args.embedding_model_path,
            batch_size=args.embedding_batch_size,
            max_length=args.embedding_max_length,
            dtype=args.embedding_dtype,
            backend=args.embedding_backend,
        )
        if needs_embedder
        else None
    )
    return {
        "direct": DirectMethod,
        "full": FullContextMethod,
        "bm25": BM25Method,
        "vector": VectorMethod,
        "hybrid": HybridMethod,
        "hyde": HyDEMethod,
        "simple_graph": SimpleGraphMethod,
        "sg_rag": SGRAGMethod,
        "skillgraph": SGRAGMethod,
    }[method](args, llm, embedder)


def row_extra_fields(row: dict[str, Any]) -> dict[str, Any]:
    reserved = {"_id", "label", "input", "answers", "context"}
    return {key: value for key, value in row.items() if key not in reserved}


def evaluate_method_domain(
    args: argparse.Namespace,
    method_name: str,
    domain: str,
    rows: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    method = make_method(args, method_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metric_totals: defaultdict[str, float] = defaultdict(float)
    retrieval_totals: defaultdict[str, float] = defaultdict(float)
    retrieval_counts: defaultdict[str, int] = defaultdict(int)
    total_query_time = 0.0
    total_generation_time = 0.0
    total_context_words = 0
    total_retrieved_chunks = 0
    retrieval_available_count = 0
    error_count = 0
    count = 0
    stopped_early = False
    stop_reason = None

    with output_path.open("w", encoding="utf-8") as handle:
        for sample_index, row in enumerate(rows, start=1):
            question = row["input"]
            context = row.get("context") or ""
            answers = row.get("answers") or []
            if isinstance(answers, str):
                answers = [answers]

            start = time.perf_counter()
            try:
                raw_prediction, retrieved, generation_time = method.answer(question, context)
                error = None
            except Exception as exc:
                raw_prediction = ""
                retrieved = RetrievalResult(chunks=[], meta={})
                generation_time = 0.0
                error = str(exc)
                error_count += 1
            query_time = time.perf_counter() - start
            scored_prediction = (
                postprocess_short_answer(raw_prediction, args.max_answer_words)
                if args.score_postprocessed
                else raw_prediction
            )
            score_detail = best_qa_prf_detail(scored_prediction, answers)
            metrics = {
                "precision": float(score_detail["precision"]),
                "recall": float(score_detail["recall"]),
                "f1": float(score_detail["f1"]),
            }
            metrics.update(answer_quality_metrics(scored_prediction, answers))
            retrieval_metrics = retrieval_proxy_metrics(answers, retrieved.chunks)
            if retrieval_metrics.get("available"):
                retrieval_available_count += 1
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if math.isfinite(float(value)):
                        metric_totals[key] += float(value)
            for key, value in retrieval_metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if math.isfinite(float(value)):
                        retrieval_totals[key] += float(value)
                        retrieval_counts[key] += 1
            count += 1
            total_query_time += query_time
            total_generation_time += generation_time
            total_context_words += len(context.split())
            total_retrieved_chunks += len(retrieved.chunks)
            running_metrics = {key: value / count for key, value in metric_totals.items()}

            record = {
                "_id": row.get("_id"),
                "domain": domain,
                "method": method_name,
                "sample_index": sample_index,
                "label": row.get("label"),
                "input": question,
                "answers": answers,
                "prediction": scored_prediction,
                "raw_prediction": raw_prediction,
                "metrics": metrics,
                "running_metrics": running_metrics,
                "retrieval_metrics": retrieval_metrics,
                "scoring": {
                    "best_answer": score_detail["best_answer"],
                    "best_answer_index": score_detail["best_answer_index"],
                    "pred_token_count": score_detail["pred_token_count"],
                    "gold_token_count": score_detail["gold_token_count"],
                    "overlap_token_count": score_detail["overlap_token_count"],
                },
                "context": {
                    "context_id": stable_hash(context) if context else None,
                    "char_length": len(context),
                    "word_count_approx": approx_word_count(context),
                },
                "retrieval": {
                    **retrieved.meta,
                    "top_k": args.top_k,
                    "retrieved_chunk_count": len(retrieved.chunks),
                    "chunks": [
                        {
                            key: chunk.get(key)
                            for key in [
                                "chunk_id",
                                "rank",
                                "score",
                                "dense_score",
                                "bm25_score",
                                "sentence_dense_score",
                                "source",
                                "word_start",
                                "word_end",
                                "char_length",
                                "skill_nodes",
                                "skill_events",
                                "evidence_sentences",
                            ]
                            if key in chunk
                        }
                        | (
                            {"preview": chunk["text"][: args.save_chunk_preview_chars]}
                            if args.save_chunk_preview_chars > 0
                            else {}
                        )
                        for chunk in retrieved.chunks
                    ],
                },
                "timing": {
                    "query_time_sec": round(query_time, 4),
                    "generation_time_sec": round(generation_time, 4),
                },
                "row_extra": row_extra_fields(row),
                "error": error,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

            print(
                f"[{method_name}/{domain}] {sample_index}/{len(rows)} "
                f"F1={metrics['f1']:.4f} P={metrics['precision']:.4f} "
                f"R={metrics['recall']:.4f} avgF1={running_metrics['f1']:.4f} "
                f"avgP={running_metrics['precision']:.4f} time={query_time:.1f}s"
            )

            if (
                args.enforce_threshold
                and count >= args.gate_after
                and (
                    running_metrics["precision"] < args.min_precision
                    or running_metrics["f1"] < args.min_f1
                )
            ):
                stopped_early = True
                stop_reason = (
                    f"quality gate failed after {count} samples: "
                    f"precision={running_metrics['precision']:.4f} < {args.min_precision:.4f} "
                    f"or f1={running_metrics['f1']:.4f} < {args.min_f1:.4f}"
                )
                print(f"STOP: {stop_reason}")
                break

    if count == 0:
        return {
            "method": method_name,
            "domain": domain,
            "count": 0,
            "error_count": error_count,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "exact_match": 0.0,
            "rouge_l": 0.0,
            "error_rate": 0.0,
            "avg_query_time_sec": 0.0,
            "avg_generation_time_sec": 0.0,
            "avg_context_words": 0.0,
            "avg_retrieved_chunks": 0.0,
            "retrieval_available_count": 0,
            "retrieval_available_rate": 0.0,
            "retrieval_metric_definition": "answer_token_overlap_proxy",
            "throughput_qps": 0.0,
            "threshold_met": False,
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
        }
    averages = {key: value / count for key, value in metric_totals.items()}
    retrieval_averages = {
        f"retrieval_{key}": retrieval_totals[key] / retrieval_counts[key]
        for key in retrieval_totals
        if retrieval_counts[key]
    }
    summary = {
        "method": method_name,
        "domain": domain,
        "count": count,
        "error_count": error_count,
        **averages,
        **retrieval_averages,
        "error_rate": error_count / count,
        "avg_query_time_sec": total_query_time / count,
        "avg_generation_time_sec": total_generation_time / count,
        "avg_context_words": total_context_words / count,
        "avg_retrieved_chunks": total_retrieved_chunks / count,
        "retrieval_available_count": retrieval_available_count,
        "retrieval_available_rate": retrieval_available_count / count,
        "retrieval_metric_definition": "answer_token_overlap_proxy",
        "throughput_qps": count / total_query_time if total_query_time else 0.0,
        "threshold_met": averages["precision"] >= args.min_precision and averages["f1"] >= args.min_f1,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
    }
    return summary

def write_run_config(args: argparse.Namespace, methods: list[str], domains: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "methods": methods,
        "domains": domains,
        "dataset_dir": args.dataset_dir,
        "limit": args.limit,
        "models": {
            "llm_model_path": args.llm_model_path,
            "embedding_model_path": args.embedding_model_path,
            "embedding_backend": args.embedding_backend,
            "device_map": args.device_map,
            "llm_dtype": args.llm_dtype,
            "embedding_dtype": args.embedding_dtype,
            "embedding_max_length": args.embedding_max_length,
            "embedding_batch_size": args.embedding_batch_size,
        },
        "retrieval": {
            "chunk_words": args.chunk_words,
            "chunk_overlap": args.chunk_overlap,
            "top_k": args.top_k,
            "hybrid_alpha": args.hybrid_alpha,
            "graph_extractor": args.graph_extractor,
            "graph_max_terms_per_chunk": args.graph_max_terms_per_chunk,
            "graph_edge_window": args.graph_edge_window,
            "skillgraph": {
                "dense_weight": args.skillgraph_dense_weight,
                "bm25_weight": args.skillgraph_bm25_weight,
                "node_weight": args.skillgraph_node_weight,
                "edge_weight": args.skillgraph_edge_weight,
                "event_weight": args.skillgraph_event_weight,
                "evidence_weight": args.skillgraph_evidence_weight,
                "sentence_dense": args.skillgraph_sentence_dense,
                "sentence_dense_weight": args.skillgraph_sentence_dense_weight,
                "max_node_df_ratio": args.skillgraph_max_node_df_ratio,
                "evidence_window": args.skillgraph_evidence_window,
                "aspect_weight": args.skillgraph_aspect_weight,
                "answer_compression": args.skillgraph_answer_compression,
                "use_skill_card_graph": args.skillgraph_use_skill_card_graph,
                "use_multilayer_retrieval": args.skillgraph_use_multilayer_retrieval,
                "adaptive_answer_detail": args.skillgraph_adaptive_answer_detail,
                "query_variants": args.skillgraph_query_variants,
                "rrf_weight": args.skillgraph_rrf_weight,
                "rrf_k": args.skillgraph_rrf_k,
                "topic_terms": args.skillgraph_topic_terms,
                "entity_terms": args.skillgraph_entity_terms,
                "event_terms": args.skillgraph_event_terms,
                "max_events_per_card": args.skillgraph_max_events_per_card,
                "max_nodes_per_card": args.skillgraph_max_nodes_per_card,
                "evidence_per_chunk": args.skillgraph_evidence_per_chunk,
            },
        },
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "llm_max_input_tokens": args.llm_max_input_tokens,
            "score_postprocessed": args.score_postprocessed,
            "max_answer_words": args.max_answer_words,
        },
        "quality_gate": {
            "min_precision": args.min_precision,
            "min_f1": args.min_f1,
            "gate_after": args.gate_after,
            "enforce_threshold": args.enforce_threshold,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__ if torch is not None else "not-installed",
            "transformers": transformers_version,
            "cuda_available": bool(torch is not None and torch.cuda.is_available()),
            "cuda_device_count": torch.cuda.device_count() if torch is not None and torch.cuda.is_available() else 0,
        },
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

def _run(args: argparse.Namespace) -> None:
    methods = args.methods
    if methods == ["all"]:
        methods = ["direct", "full", "bm25", "vector", "hybrid", "hyde", "simple_graph", "sg_rag"]
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    domains = select_domains(dataset_dir, args.domains)
    write_run_config(args, methods, domains, output_dir)

    summary_rows = []
    for domain in domains:
        data_path = dataset_dir / f"{domain}.jsonl"
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        rows = load_domain_rows(dataset_dir, domain, args.limit)
        print(f"Domain={domain} samples={len(rows)}")
        for method_name in methods:
            method_output = output_dir / method_name / f"{domain}_predictions.jsonl"
            metrics = evaluate_method_domain(args, method_name, domain, rows, method_output)
            summary_rows.append(metrics)
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    if summary_rows:
        summary_path = output_dir / "summary_metrics.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = sorted({key for row in summary_rows for key in row})
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Summary written to {summary_path}")


def run(args: argparse.Namespace) -> None:
    started_at = datetime.now().astimezone()
    timer_start = time.perf_counter()
    status = "failed"
    error = None
    methods = (
        ["direct", "full", "bm25", "vector", "hybrid", "hyde", "simple_graph", "sg_rag"]
        if args.methods == ["all"]
        else args.methods
    )
    try:
        _run(args)
        status = "completed"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_run_metadata(
            args.output_dir,
            started_at,
            datetime.now().astimezone(),
            time.perf_counter() - timer_start,
            status,
            methods,
            args.domains,
            args.limit,
            error,
        )

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--llm-model-path", required=True)
    parser.add_argument("--embedding-model-path", default="")
    parser.add_argument("--output-dir", default="rag_benchmark_results")
    parser.add_argument("--domains", nargs="+", default=["mix"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["sg_rag"],
        choices=["direct", "full", "bm25", "vector", "hybrid", "hyde", "simple_graph", "sg_rag", "skillgraph", "all"],
    )

    parser.add_argument("--chunk-words", type=int, default=350)
    parser.add_argument("--chunk-overlap", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--hybrid-alpha", type=float, default=0.65)

    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--llm-max-input-tokens", type=int, default=7900)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--llm-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--embedding-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--embedding-backend", default="hf", choices=["auto", "flag", "hf"])
    parser.add_argument("--embedding-max-length", type=int, default=8192)
    parser.add_argument("--embedding-batch-size", type=int, default=8)

    parser.add_argument("--hyde-max-new-tokens", type=int, default=128)
    parser.add_argument("--graph-extractor", default="heuristic", choices=["heuristic"])
    parser.add_argument("--graph-max-terms-per-chunk", type=int, default=40)
    parser.add_argument("--graph-edge-window", type=int, default=8)
    parser.add_argument("--skillgraph-dense-weight", type=float, default=0.35)
    parser.add_argument("--skillgraph-bm25-weight", type=float, default=0.30)
    parser.add_argument("--skillgraph-node-weight", type=float, default=0.75)
    parser.add_argument("--skillgraph-edge-weight", type=float, default=0.12)
    parser.add_argument("--skillgraph-event-weight", type=float, default=0.65)
    parser.add_argument("--skillgraph-evidence-weight", type=float, default=0.20)
    parser.add_argument("--skillgraph-sentence-dense", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skillgraph-sentence-dense-weight", type=float, default=1.15)
    parser.add_argument("--skillgraph-max-node-df-ratio", type=float, default=0.75)
    parser.add_argument("--skillgraph-evidence-window", type=int, default=1)
    parser.add_argument("--skillgraph-aspect-weight", type=float, default=0.85)
    parser.add_argument("--skillgraph-answer-compression", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skillgraph-use-skill-card-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skillgraph-use-multilayer-retrieval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skillgraph-adaptive-answer-detail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skillgraph-query-variants", type=int, default=5)
    parser.add_argument("--skillgraph-rrf-weight", type=float, default=0.20)
    parser.add_argument("--skillgraph-rrf-k", type=int, default=60)
    parser.add_argument("--skillgraph-topic-terms", type=int, default=12)
    parser.add_argument("--skillgraph-entity-terms", type=int, default=20)
    parser.add_argument("--skillgraph-event-terms", type=int, default=40)
    parser.add_argument("--skillgraph-max-events-per-card", type=int, default=12)
    parser.add_argument("--skillgraph-max-nodes-per-card", type=int, default=36)
    parser.add_argument("--skillgraph-evidence-per-chunk", type=int, default=4)

    parser.add_argument("--score-postprocessed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-answer-words", type=int, default=40)
    parser.add_argument("--save-chunk-preview-chars", type=int, default=500)

    parser.add_argument("--min-precision", type=float, default=0.7)
    parser.add_argument("--min-f1", type=float, default=0.6)
    parser.add_argument("--gate-after", type=int, default=3)
    parser.add_argument("--enforce-threshold", action="store_true")

    return parser

def main() -> None:
    args = build_arg_parser().parse_args()
    if any(method in args.methods or args.methods == ["all"] for method in ["vector", "hybrid", "hyde", "sg_rag", "skillgraph"]):
        if not args.embedding_model_path:
            raise ValueError("--embedding-model-path is required for vector/hybrid/hyde/skillgraph methods")
    run(args)


if __name__ == "__main__":
    main()
