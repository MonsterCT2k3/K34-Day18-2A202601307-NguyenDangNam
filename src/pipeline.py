from __future__ import annotations

"""Production RAG Pipeline — Bài tập NHÓM: ghép M1+M2+M3+M4."""

import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({time.time()-t0:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    print(f"  ✓ Indexed ({time.time()-t0:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:3]]

    from config import get_llm_client, MODEL_NAME
    client = get_llm_client()
    if client and contexts:
        try:
            context_str = "\n\n".join(contexts)
            resp = client.chat.completions.create(model=MODEL_NAME, messages=[
                {"role": "system", "content": "Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'"},
                {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
            ])
            answer = resp.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker, latency_stats: dict | None = None):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    t_gen_start = time.time()
    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)
    t_gen = time.time() - t_gen_start

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    t_eval = time.time() - t0
    print(f"  ✓ RAGAS done ({t_eval:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    if latency_stats is not None:
        latency_stats["generation_s"] = t_gen
        latency_stats["eval_s"] = t_eval
        print("\n" + "=" * 60)
        print("LATENCY BREAKDOWN REPORT (+2 Bonus)")
        print("=" * 60)
        print(f"  {'Stage':<30} {'Latency':>10}")
        print("-" * 42)
        print(f"  {'1. M1 Hierarchical Chunking':<30} {latency_stats.get('chunking_s', 0):>9.2f}s")
        print(f"  {'2. M5 Context Enrichment':<30} {latency_stats.get('enrichment_s', 0):>9.2f}s")
        print(f"  {'3. M2 Hybrid Indexing (Qdrant)':<30} {latency_stats.get('indexing_s', 0):>9.2f}s")
        print(f"  {'4. M3 Cross-Encoder Reranker':<30} {latency_stats.get('reranker_s', 0):>9.2f}s")
        print(f"  {'5. Query Retrieval & LLM Gen':<30} {latency_stats.get('generation_s', 0):>9.2f}s")
        print(f"  {'6. M4 RAGAS Evaluation':<30} {latency_stats.get('eval_s', 0):>9.2f}s")
        total_time = sum(latency_stats.values())
        print("-" * 42)
        print(f"  {'Total Pipeline Time':<30} {total_time:>9.2f}s")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    return results


def main_pipeline():
    latency_stats = {}
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    latency_stats["chunking_s"] = time.time() - t0
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({latency_stats['chunking_s']:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        latency_stats["enrichment_s"] = time.time() - t0
        print(f"  ✓ Enriched {len(enriched)} chunks ({latency_stats['enrichment_s']:.1f}s)", flush=True)
    else:
        latency_stats["enrichment_s"] = time.time() - t0
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    latency_stats["indexing_s"] = time.time() - t0
    print(f"  ✓ Indexed ({latency_stats['indexing_s']:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    latency_stats["reranker_s"] = time.time() - t0
    print(f"  ✓ Reranker ready ({latency_stats['reranker_s']:.1f}s)", flush=True)

    return evaluate_pipeline(search, reranker, latency_stats=latency_stats)


if __name__ == "__main__":
    start = time.time()
    main_pipeline()
    print(f"\nTotal: {time.time() - start:.1f}s")

