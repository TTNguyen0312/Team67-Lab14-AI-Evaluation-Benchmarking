"""
Golden Dataset Generator
Reads chunks directly from ChromaDB's SQLite file, then calls OpenAI to generate
50 QA pairs (factual, inferential, adversarial) with ground-truth chunk IDs.
Output: data/golden_set.jsonl
"""

import json
import asyncio
import os
import sqlite3
from typing import List, Dict
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db", "chroma.sqlite3")
OUTPUT_PATH     = os.path.join(os.path.dirname(__file__), "golden_set.jsonl")
TARGET_TOTAL    = 50


# 1. Read all chunks from ChromaDB SQLite

def load_chunks_from_chroma(db_path: str) -> List[Dict]:
    """Return list of {chunk_id, document, doc_id, effective_date}."""
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    # Pivot the key-value metadata table into one row per embedding
    cur.execute("SELECT DISTINCT embedding_id FROM embeddings")
    chunk_ids = [r[0] for r in cur.fetchall()]

    chunks = []
    for cid in chunk_ids:
        cur.execute(
            "SELECT key, string_value FROM embedding_metadata "
            "JOIN embeddings ON embedding_metadata.id = embeddings.id "
            "WHERE embeddings.embedding_id = ?",
            (cid,),
        )
        row_meta = {k: v for k, v in cur.fetchall()}
        chunks.append(
            {
                "chunk_id":       cid,
                "document":       row_meta.get("chroma:document", ""),
                "doc_id":         row_meta.get("doc_id", ""),
                "effective_date": row_meta.get("effective_date", ""),
            }
        )

    conn.close()
    return chunks


# 2. LLM helpers

SYSTEM_PROMPT = """\
Bạn là chuyên gia tạo bộ dữ liệu QA. Bạn sẽ được cung cấp một hoặc nhiều đoạn văn \
từ cơ sở tri thức. Nhiệm vụ của bạn là tạo ra các cặp câu hỏi-câu trả lời thực tế, \
đa dạng mà nhân viên hoặc nhân viên hỗ trợ có thể hỏi.

Quy tắc:
- Toàn bộ nội dung (câu hỏi, câu trả lời) PHẢI được viết bằng tiếng Việt.
- Mỗi cặp PHẢI có thể trả lời CHỈ DỰA TRÊN đoạn văn được cung cấp — không dùng kiến thức ngoài.
- Đa dạng độ khó: ít nhất 1 câu hỏi đánh lừa/khó (adversarial) mỗi batch \
  (ví dụ: giả định sai có vẻ hợp lý, yêu cầu ngoài phạm vi, hoặc câu hỏi cần đọc kỹ \
  một con số/ngày tháng).
- expected_answer phải là một câu hoàn chỉnh, súc tích, trả lời trực tiếp câu hỏi bằng tiếng Việt.
- Luôn trả về JSON object chứa key "pairs" có giá trị là mảng.

Output schema:
{
  "pairs": [
    {
      "question": "<string>",
      "expected_answer": "<string>",
      "difficulty": "easy" | "medium" | "hard",
      "type": "factual" | "inferential" | "adversarial" | "out-of-scope"
    }
  ]
}
"""


def _build_chunk_block(chunks: List[Dict]) -> str:
    parts = []
    for c in chunks:
        parts.append(
            f"[chunk_id: {c['chunk_id']}]\n"
            f"doc_id: {c['doc_id']} | effective_date: {c['effective_date']}\n"
            f"content: {c['document']}"
        )
    return "\n\n".join(parts)


async def generate_pairs_for_chunks(
    client: AsyncOpenAI,
    chunks: List[Dict],
    n: int,
) -> List[Dict]:
    """Ask the LLM to produce `n` QA pairs for the given chunk(s)."""
    chunk_block = _build_chunk_block(chunks)
    user_msg = (
        f"Generate exactly {n} question-answer pairs from the following chunk(s).\n\n"
        f"{chunk_block}"
    )

    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.8,
        response_format={"type": "json_object"},
    )

    raw = resp.choices[0].message.content or "{}"

    parsed = json.loads(raw)

    # Normalise: model may return {"pairs":[...]}, {"questions":[...]}, a bare
    # list (shouldn't happen with json_object mode), or a single object.
    if isinstance(parsed, list):
        pairs = parsed
    elif isinstance(parsed, dict):
        # prefer "pairs", then any list-valued key, then treat dict as single item
        list_val = next((v for v in parsed.values() if isinstance(v, list)), None)
        if list_val is not None:
            pairs = list_val
        elif "question" in parsed:
            pairs = [parsed]   # single QA object returned directly
        else:
            pairs = []

    # Attach ground-truth IDs from the chunks used
    ground_truth_ids = [c["chunk_id"] for c in chunks]
    combined_context = " | ".join(c["document"] for c in chunks)

    result = []
    for p in pairs[:n]:
        result.append(
            {
                "question":            p.get("question", ""),
                "expected_answer":     p.get("expected_answer", ""),
                "ground_truth_ids":    ground_truth_ids,
                "context":             combined_context,
                "metadata": {
                    "difficulty": p.get("difficulty", "medium"),
                    "type":       p.get("type", "factual"),
                    "doc_ids":    [c["doc_id"] for c in chunks],
                },
            }
        )
    return result


# 3. Orchestration — distribute questions across chunks dynamically

def _build_cross_pairs(chunks: List[Dict], n_pairs: int) -> List[List[Dict]]:
    """
    Return `n_pairs` pairs of chunks drawn from *different* doc_ids.
    Cycles through all distinct doc_id combinations so every document
    appears at least once in a cross-pair when possible.
    """
    import itertools, random

    # Group chunk indices by doc_id
    by_doc: Dict[str, List[Dict]] = {}
    for c in chunks:
        by_doc.setdefault(c["doc_id"], []).append(c)

    doc_ids = list(by_doc.keys())
    if len(doc_ids) < 2:
        return []

    # All unique cross-doc combinations, then cycle to fill n_pairs
    combos = list(itertools.combinations(doc_ids, 2))
    random.seed(42)
    random.shuffle(combos)

    pairs = []
    for i in range(n_pairs):
        doc_a, doc_b = combos[i % len(combos)]
        chunk_a = random.choice(by_doc[doc_a])
        chunk_b = random.choice(by_doc[doc_b])
        pairs.append([chunk_a, chunk_b])
    return pairs


async def build_golden_dataset(chunks: List[Dict], total: int) -> List[Dict]:
    """
    Strategy (scales to any number of chunks):
    - Each chunk gets at least 2 questions  →  30 chunks × 2 = 60  (buffer over 50)
    - 5 cross-doc pairs get 2 questions each  →  10 more
    - Trim the combined list to exactly `total` at the end.
    Requesting ≥2 per call avoids the model returning a bare single-object
    instead of a list, which was silently dropping most results.
    """
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    tasks = []

    # --- single-chunk batches: 2 per chunk ---
    for chunk in chunks:
        tasks.append(generate_pairs_for_chunks(client, [chunk], 2))

    # --- cross-doc batches: 2 per pair, pairs from different doc_ids ---
    n_cross     = 5
    cross_pairs = _build_cross_pairs(chunks, n_cross)
    for pair in cross_pairs:
        tasks.append(generate_pairs_for_chunks(client, pair, 2))

    print(f"Dispatching {len(tasks)} batches to OpenAI "
          f"({len(chunks)} single-chunk + {len(cross_pairs)} cross-doc) …")
    batch_results = await asyncio.gather(*tasks)

    all_pairs: List[Dict] = []
    for batch in batch_results:
        all_pairs.extend(batch)

    if len(all_pairs) < total:
        print(f"WARNING: only {len(all_pairs)} pairs generated (expected ≥{total}). "
              "Check API responses above.")

    return all_pairs[:total]


# 4. Entry point

async def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Add it to your .env file.")
        return

    if not os.path.exists(CHROMA_DB_PATH):
        print(f"ERROR: ChromaDB not found at {CHROMA_DB_PATH}")
        return

    print(f"Loading chunks from {CHROMA_DB_PATH} …")
    chunks = load_chunks_from_chroma(CHROMA_DB_PATH)
    print(f"Found {len(chunks)} chunks: {[c['chunk_id'] for c in chunks]}")

    if not chunks:
        print("ERROR: No chunks found. Populate your ChromaDB first.")
        return

    dataset = await build_golden_dataset(chunks, TARGET_TOTAL)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nDone! {len(dataset)} cases saved to {OUTPUT_PATH}")
    print("\nSample entry:")
    print(json.dumps(dataset[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
