from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from pypdf import PdfReader


@dataclass(frozen=True)
class QAExample:
    instruction: str
    input: str
    output: str


def _safe_decode(b: bytes) -> str:
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.decode("latin-1", errors="replace")


def extract_texts_from_uploads(uploaded_files: Iterable) -> list[dict]:
    """
    Returns list of dicts:
      - {"source": <filename>, "kind": "text", "text": <str>}
      - {"source": <filename>, "kind": "qa", "qa": list[QAExample]}
    """
    out: list[dict] = []
    for uf in uploaded_files:
        name = getattr(uf, "name", "uploaded")
        ext = os.path.splitext(name)[1].lower()
        raw = uf.getvalue() if hasattr(uf, "getvalue") else uf.read()

        if ext == ".pdf":
            reader = PdfReader(io.BytesIO(raw))
            pages = []
            for p in reader.pages:
                t = p.extract_text() or ""
                if t.strip():
                    pages.append(t)
            out.append({"source": name, "kind": "text", "text": "\n\n".join(pages)})
            continue

        if ext in {".txt", ".md", ".log"}:
            out.append({"source": name, "kind": "text", "text": _safe_decode(raw)})
            continue

        if ext == ".csv":
            df = pd.read_csv(io.BytesIO(raw))
            cols = {c.strip().lower(): c for c in df.columns}
            if "question" in cols and "answer" in cols:
                q_col, a_col = cols["question"], cols["answer"]
                examples: list[QAExample] = []
                for _, row in df.iterrows():
                    q = str(row.get(q_col, "")).strip()
                    a = str(row.get(a_col, "")).strip()
                    if q and a:
                        examples.append(
                            QAExample(
                                instruction="Answer the user's healthcare question accurately and concisely.",
                                input=q,
                                output=a,
                            )
                        )
                out.append({"source": name, "kind": "qa", "qa": examples})
            else:
                # fallback: serialize rows into text
                lines = []
                head = df.head(200)
                for _, row in head.iterrows():
                    pairs = []
                    for c in df.columns:
                        v = row.get(c, "")
                        s = str(v).strip()
                        if s and s.lower() != "nan":
                            pairs.append(f"{c}: {s}")
                    if pairs:
                        lines.append(" | ".join(pairs))
                out.append({"source": name, "kind": "text", "text": "\n".join(lines)})
            continue

        # Unknown file types treated as text if decodable
        out.append({"source": name, "kind": "text", "text": _safe_decode(raw)})

    return out


_ICD_RE = re.compile(r"\b([A-TV-Z][0-9][0-9](?:\.[0-9A-TV-Z]{1,4})?)\b")
_CPT_RE = re.compile(r"\b(99[0-9]{3}|[0-9]{5})\b")


def _split_into_chunks(text: str, target_chars: int = 1200) -> list[str]:
    blocks = [b.strip() for b in re.split(r"\n\s*\n+", text) if b.strip()]
    chunks: list[str] = []
    buf = ""
    for b in blocks:
        if not buf:
            buf = b
            continue
        if len(buf) + 2 + len(b) <= target_chars:
            buf = f"{buf}\n\n{b}"
        else:
            chunks.append(buf)
            buf = b
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def _make_question(chunk: str) -> str:
    lowered = chunk.lower()
    icds = _ICD_RE.findall(chunk)
    if "icd" in lowered or icds:
        code = icds[0] if icds else "an ICD code"
        return f"What does the dataset say about ICD code {code} and when it applies?"
    if "cpt" in lowered or "procedure" in lowered:
        cpts = _CPT_RE.findall(chunk)
        code = cpts[0] if cpts else "a procedure code"
        return f"Does the policy cover procedure/code {code}? List any conditions or exclusions."
    if "claim" in lowered:
        return "What does the dataset say about claim submission requirements or denial reasons?"
    if "coverage" in lowered or "covered" in lowered or "benefit" in lowered:
        return "What does the policy cover, and what are the conditions/exclusions?"
    if "prior authorization" in lowered or "preauthorization" in lowered:
        return "When is prior authorization required, and what documentation is needed?"
    return "Summarize the key healthcare policy rules and any important exceptions."


def build_qa_from_text_sources(
    items: list[dict],
    *,
    max_examples: int = 200,
    chunk_chars: int = 1200,
) -> list[QAExample]:
    """
    Converts extracted healthcare text into a small instruct dataset.

    Notes:
    - This is fine-tuning style "learning" (LoRA): we provide policy/claims text during training.
    - For best 'WOW', users can upload a Q&A CSV (question, answer).
    """
    examples: list[QAExample] = []

    # Q&A CSV takes precedence (best signal).
    for it in items:
        if it.get("kind") == "qa":
            examples.extend(it["qa"])

    for it in items:
        if it.get("kind") != "text":
            continue
        text = (it.get("text") or "").strip()
        if not text:
            continue

        for chunk in _split_into_chunks(text, target_chars=chunk_chars):
            if len(examples) >= max_examples:
                break
            q = _make_question(chunk)
            # "Memorization" style: teach the model a canonical answer for likely questions.
            a = chunk.strip()
            examples.append(
                QAExample(
                    instruction=(
                        "You are a healthcare policy assistant. Answer the user's question using the dataset you "
                        "were trained on. If the dataset does not contain the answer, say you don't know."
                    ),
                    input=q,
                    output=a,
                )
            )
        if len(examples) >= max_examples:
            break

    return examples[:max_examples]

