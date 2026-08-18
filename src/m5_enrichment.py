from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if len(text.strip()) < 80:
        return text.strip()
    from config import get_llm_client, MODEL_NAME
    client = get_llm_client()
    if client:
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Tóm tắt đoạn văn sau trong 1 câu ngắn gọn súc tích bằng tiếng Việt (không dài hơn văn bản gốc)."},
                    {"role": "user", "content": text},
                ],
                max_tokens=100,
            )
            content = resp.choices[0].message.content
            if content:
                res = content.strip()
                if len(res) <= len(text) * 2:
                    return res
        except Exception as e:
            print(f"  ⚠️  LLM summarize failed: {e}")

    # Extractive fallback (không cần API):
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    return ". ".join(sentences[:2]) + "." if sentences else text


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    from config import get_llm_client, MODEL_NAME
    client = get_llm_client()
    if client:
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. Trả về mỗi câu hỏi trên 1 dòng, kết thúc bằng dấu hỏi '?'."},
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
            )
            content = resp.choices[0].message.content
            if content:
                questions = content.strip().split("\n")
                cleaned = [q.strip().lstrip("0123456789.-) ") for q in questions if q.strip()]
                if cleaned:
                    return cleaned[:n_questions]
        except Exception as e:
            print(f"  ⚠️  LLM HyQA failed: {e}")

    # Extractive fallback:
    import re
    sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if len(s.strip()) > 10]
    return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    from config import get_llm_client, MODEL_NAME
    client = get_llm_client()
    if client:
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về 1 câu duy nhất."},
                    {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"},
                ],
                max_tokens=80,
            )
            content = resp.choices[0].message.content
            if content:
                context = content.strip()
                return f"{context}\n\n{text}"
        except Exception as e:
            print(f"  ⚠️  LLM contextual failed: {e}")

    # Simple fallback:
    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    import json as _json
    from config import get_llm_client, MODEL_NAME
    client = get_llm_client()
    if client:
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": 'Trích xuất metadata từ đoạn văn. Trả về đúng 1 JSON object: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}'},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            content = resp.choices[0].message.content
            if content:
                clean_json = content.strip()
                if "```json" in clean_json:
                    clean_json = clean_json.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_json:
                    clean_json = clean_json.split("```")[1].split("```")[0].strip()
                return _json.loads(clean_json)
        except Exception as e:
            print(f"  ⚠️  LLM metadata failed: {e}")

    return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}


# ─── Combined Single-Call Mode ───────────────────────────


_ENRICH_CACHE = {}


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    global _ENRICH_CACHE
    cache_key = (text[:100], source)
    if cache_key in _ENRICH_CACHE:
        return _ENRICH_CACHE[cache_key]

    import json as _json
    import re
    from config import get_llm_client, MODEL_NAME
    client = get_llm_client()
    if client:
        try:
            prompt_user = (
                f"Tài liệu: {source}\n\nĐoạn văn:\n{text[:1200]}\n\n"
                "Hãy phân tích đoạn văn và trả về đúng 1 JSON object dạng:\n"
                '{"summary": "tóm tắt ngắn gọn", "questions": ["câu hỏi 1?", "câu hỏi 2?", "câu hỏi 3?"], '
                '"context": "mô tả ngữ cảnh", "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi"}}'
            )
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Bạn là chuyên gia RAG. Chỉ trả về duy nhất 1 JSON object hợp lệ, không thêm bất kỳ văn bản giải thích nào khác."},
                    {"role": "user", "content": prompt_user},
                ],
                max_tokens=800,
                temperature=0.0,
            )
            content = resp.choices[0].message.content
            if content:
                clean_json = content.strip()
                if "```json" in clean_json:
                    clean_json = clean_json.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_json:
                    clean_json = clean_json.split("```")[1].split("```")[0].strip()
                match = re.search(r'\{.*\}', clean_json, re.DOTALL)
                if match:
                    clean_json = match.group(0)
                parsed = _json.loads(clean_json)
                if isinstance(parsed, dict) and "summary" in parsed:
                    _ENRICH_CACHE[cache_key] = parsed
                    return parsed
        except Exception:
            pass

    # Fallback nếu không có API key hoặc lỗi:
    import re as _re
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    summary = ". ".join(sentences[:2]) + "." if sentences else text
    questions = [f"{s.rstrip('.')}?" for s in _re.split(r'[.!?\n]', text) if len(s.strip()) > 10][:3]
    context = f"Trích từ tài liệu {source}." if source else "Tài liệu nội bộ doanh nghiệp."
    fallback_res = {
        "summary": summary,
        "questions": questions,
        "context": context,
        "metadata": {"topic": "general", "entities": [], "category": "policy", "language": "vi"},
    }
    _ENRICH_CACHE[cache_key] = fallback_res
    return fallback_res


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    import time
    from config import OPENAI_API_KEY

    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        # Nghỉ nhẹ giữa các request nếu gọi API để tránh rate limit 429
        if OPENAI_API_KEY and len(chunks) > 5:
            time.sleep(0.6)

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
