# run_pipeline.py
# Usage: python run_pipeline.py config_training.yaml
#
# What it does:
# - Ingest PDFs/TXT (glob paths)
# - Normalize + (optional) boilerplate line removal
# - Filter low-quality docs
# - Config-driven PII masking (regex patterns defined in YAML)
# - Segment into overlapping chunks (char-based)
# - Exact dedup
# - Export:
#   out/<run_name>/corpus.jsonl  ({"text":..., "meta":...})
#   out/<run_name>/diagnostics.json
#   optional MLX data: data/mlx/<run_name>/{train,valid}.jsonl ({"text":...})

import sys
import json
import re
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Tuple

import yaml
from pypdf import PdfReader


# Schema
@dataclass
class Document:
    doc_id: str
    source: str
    doc_type: str
    path: str
    text: str
    meta: Dict[str, Any]


@dataclass
class Segment:
    segment_id: str
    doc_id: str
    text: str
    char_count: int
    word_count: int
    meta: Dict[str, Any]


# Helpers
def stable_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:12]


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: List[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def normalize_text(text: str, collapse_whitespace: bool = True) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    # de-hyphenate line breaks: "orga-\nnisations" -> "organisations"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if collapse_whitespace:
        text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def remove_obvious_boilerplate_lines(text: str) -> str:
    lines_out: List[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            lines_out.append("")
            continue
        if re.fullmatch(r"page\s*\d+(\s*of\s*\d+)?", s, re.IGNORECASE):
            continue
        if re.fullmatch(r"\d{1,4}", s):  # lone numbers
            continue
        lines_out.append(ln)
    return "\n".join(lines_out).strip()


def nonalpha_ratio(text: str) -> float:
    if not text:
        return 1.0
    nonalpha = sum(1 for c in text if not (c.isalpha() or c.isspace()))
    return nonalpha / max(1, len(text))


def chunk_text(text: str, chunk_chars: int, overlap_chars: int) -> List[str]:
    if chunk_chars <= 0:
        t = text.strip()
        return [t] if t else []
    out: List[str] = []
    i, n = 0, len(text)
    step = max(1, chunk_chars - max(0, overlap_chars))
    while i < n:
        piece = text[i : i + chunk_chars].strip()
        if piece:
            out.append(piece)
        i += step
    return out


# Config-driven PII masking
def compile_pii_patterns(pii_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    pii_cfg example:
      enabled: true
      replacement_format: "[{name}]"
      patterns:
        email:
          enabled: true
          regex: "..."
          ignore_case: true
        phone:
          enabled: true
          regex: "..."
          line_hint_regex: "..."
    """
    compiled: Dict[str, Dict[str, Any]] = {}
    patterns = pii_cfg.get("patterns") or {}

    for name, spec in patterns.items():
        if not isinstance(spec, dict):
            continue
        regex = spec.get("regex")
        if not regex:
            continue

        enabled = bool(spec.get("enabled", True))
        flags = 0
        if spec.get("ignore_case", True):
            flags |= re.IGNORECASE

        compiled[name] = {
            "enabled": enabled,
            "regex": re.compile(regex, flags),
            "line_hint_regex": (
                re.compile(spec["line_hint_regex"], re.IGNORECASE)
                if spec.get("line_hint_regex")
                else None
            ),
        }

    return compiled


def mask_pii_configurable(
    text: str,
    compiled_patterns: Dict[str, Dict[str, Any]],
    replacement_format: str = "[{name}]",
) -> Tuple[str, Dict[str, int]]:
    """
    Applies config-defined patterns. Counts are dynamic: one counter per pattern name.
    If a pattern has line_hint_regex, masking applies only to lines matching that hint.
    """
    counts: Dict[str, int] = {name: 0 for name in compiled_patterns.keys()}

    for name, info in compiled_patterns.items():
        if not info.get("enabled", True):
            continue

        rx = info["regex"]
        line_hint = info.get("line_hint_regex")
        repl = replacement_format.format(name=name.upper())

        if line_hint is None:
            text, n = rx.subn(repl, text)
            counts[name] += n
        else:
            lines = text.splitlines()
            out_lines = []
            for ln in lines:
                if line_hint.search(ln):
                    ln, n = rx.subn(repl, ln)
                    counts[name] += n
                out_lines.append(ln)
            text = "\n".join(out_lines)

    return text, counts


# Pipeline
def load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def expand_inputs(inputs_cfg: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []
    for item in inputs_cfg:
        patterns = item.get("paths", [])
        if isinstance(patterns, str):
            patterns = [patterns]
        resolved: List[Path] = []
        for pat in patterns:
            resolved.extend(sorted(Path(".").glob(pat)))
        for p in resolved:
            expanded.append({**item, "path": str(p)})
    return expanded


def ingest_documents(
    inputs: List[Dict[str, Any]], diag: Dict[str, Any]
) -> List[Document]:
    docs: List[Document] = []
    failures: List[Dict[str, str]] = []

    for inp in inputs:
        path = Path(inp["path"])
        source = inp.get("source", "unknown")
        doc_type = inp.get("doc_type", "unknown")
        ftype = inp.get("type", "auto")

        if ftype == "auto":
            ftype = "pdf" if path.suffix.lower() == ".pdf" else "txt"

        try:
            if ftype == "pdf":
                raw = extract_pdf_text(path)
            else:
                raw = read_text_file(path)
        except Exception as e:
            failures.append({"path": str(path), "error": repr(e)})
            continue

        doc_id = stable_id(f"{source}:{doc_type}:{path.name}")
        docs.append(
            Document(
                doc_id=doc_id,
                source=source,
                doc_type=doc_type,
                path=str(path),
                text=raw,
                meta={"filename": path.name},
            )
        )

    diag["counts"]["documents_ingested"] = len(docs)
    if failures:
        diag["ingest_failures"] = failures
    return docs


def run_pipeline(config_path: Path) -> None:
    cfg = load_config(config_path)

    run_name = cfg.get("run_name", config_path.stem)
    out_dir = Path(cfg.get("output_dir", "out")) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    diag: Dict[str, Any] = {
        "run_name": run_name,
        "counts": {},
        "lengths": {},
        "dedup": {"removed": 0},
        "pii_masks": {},
        "examples": {"before_after": []},
        "config_path": str(config_path),
    }

    # Ingest
    inputs = expand_inputs(cfg["inputs"])
    docs = ingest_documents(inputs, diag)

    # Normalize
    norm_cfg = cfg.get("normalize", {})
    for d in docs:
        d.text = normalize_text(
            d.text, collapse_whitespace=norm_cfg.get("collapse_whitespace", True)
        )
        if norm_cfg.get("remove_boilerplate_lines", True):
            d.text = remove_obvious_boilerplate_lines(d.text)

    diag["counts"]["documents_after_normalize"] = len(docs)

    # Clean / filter documents
    clean_cfg = cfg.get("clean", {})
    min_doc_chars = int(clean_cfg.get("min_doc_chars", 200))
    max_doc_chars = int(clean_cfg.get("max_doc_chars", 500000))
    max_nonalpha = float(clean_cfg.get("max_nonalpha_ratio", 0.60))

    kept_docs: List[Document] = []
    for d in docs:
        t = d.text.strip()
        if len(t) < min_doc_chars or len(t) > max_doc_chars:
            continue
        if nonalpha_ratio(t) > max_nonalpha:
            continue
        kept_docs.append(d)
    docs = kept_docs
    diag["counts"]["documents_after_clean"] = len(docs)

    # PII mask (config driven)
    pii_cfg = cfg.get("pii_mask", {}) or {}
    pii_enabled = bool(pii_cfg.get("enabled", False))
    example_limit = int(cfg.get("example_pairs", 3))

    compiled_pii = compile_pii_patterns(pii_cfg) if pii_enabled else {}
    replacement_format = str(pii_cfg.get("replacement_format", "[{name}]"))

    if pii_enabled and compiled_pii:
        for d in docs:
            before = d.text[:700]
            d.text, counts = mask_pii_configurable(
                d.text, compiled_pii, replacement_format=replacement_format
            )
            after = d.text[:700]

            # aggregate counts dynamically
            for k, v in counts.items():
                diag["pii_masks"][k] = diag["pii_masks"].get(k, 0) + v

            # store a few examples
            if len(diag["examples"]["before_after"]) < example_limit:
                diag["examples"]["before_after"].append(
                    {
                        "doc_id": d.doc_id,
                        "doc_type": d.doc_type,
                        "source": d.source,
                        "before": before,
                        "after": after,
                    }
                )

    # Segment
    seg_cfg = cfg.get("segment", {}) or {}
    chunk_chars = int(seg_cfg.get("chunk_chars", 2000))
    overlap_chars = int(seg_cfg.get("overlap_chars", 200))
    seg_min_chars = int(seg_cfg.get("min_segment_chars", 200))
    seg_max_chars = int(seg_cfg.get("max_segment_chars", 6000))

    segments: List[Segment] = []
    for d in docs:
        chunks = chunk_text(
            d.text, chunk_chars=chunk_chars, overlap_chars=overlap_chars
        )
        for idx, ch in enumerate(chunks):
            ch = ch.strip()
            if len(ch) < seg_min_chars or len(ch) > seg_max_chars:
                continue
            seg_id = stable_id(f"{d.doc_id}:{idx}:{ch[:64]}")
            segments.append(
                Segment(
                    segment_id=seg_id,
                    doc_id=d.doc_id,
                    text=ch,
                    char_count=len(ch),
                    word_count=len(ch.split()),
                    meta={
                        "source": d.source,
                        "doc_type": d.doc_type,
                        "filename": d.meta.get("filename"),
                        "pii_masked": pii_enabled,
                        "pii_patterns": {
                            k: bool(v.get("enabled", True))
                            for k, v in compiled_pii.items()
                        },
                    },
                )
            )

    diag["counts"]["segments_before_dedup"] = len(segments)

    # Dedup (exact)
    dedup_cfg = cfg.get("dedup", {}) or {}
    if bool(dedup_cfg.get("enabled", True)):
        seen = set()
        uniq: List[Segment] = []
        for s in segments:
            key = hashlib.sha1(normalize_text(s.text).encode("utf-8")).hexdigest()
            if key in seen:
                diag["dedup"]["removed"] += 1
                continue
            seen.add(key)
            uniq.append(s)
        segments = uniq

    diag["counts"]["segments_after_dedup"] = len(segments)

    # Length stats
    if segments:
        chars = [s.char_count for s in segments]
        words = [s.word_count for s in segments]
        diag["lengths"] = {
            "segment_char_min": min(chars),
            "segment_char_max": max(chars),
            "segment_char_avg": sum(chars) / len(chars),
            "segment_word_min": min(words),
            "segment_word_max": max(words),
            "segment_word_avg": sum(words) / len(words),
        }

    # Export corpus.jsonl
    corpus_path = out_dir / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as f:
        for s in segments:
            f.write(
                json.dumps({"text": s.text, "meta": s.meta}, ensure_ascii=False) + "\n"
            )

    # Export diagnostics.json
    diag_path = out_dir / "diagnostics.json"
    diag_path.write_text(
        json.dumps(diag, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Optional: MLX dataset export (text only)
    mlx_cfg = cfg.get("mlx_export", {}) or {}
    if bool(mlx_cfg.get("enabled", False)):
        mlx_dir = Path(mlx_cfg.get("dir", "data/mlx")) / run_name
        mlx_dir.mkdir(parents=True, exist_ok=True)

        rows = [{"text": s.text} for s in segments]
        random.seed(int(mlx_cfg.get("seed", 42)))
        random.shuffle(rows)

        n = len(rows)
        train_frac = float(mlx_cfg.get("train_frac", 0.9))
        n_train = max(1, int(n * train_frac))

        train = rows[:n_train]
        valid = rows[n_train:] or rows[:1]

        (mlx_dir / "train.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n",
            encoding="utf-8",
        )
        (mlx_dir / "valid.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in valid) + "\n",
            encoding="utf-8",
        )

    print(f"✅ Run '{run_name}' complete.")
    print(f"   Corpus:      {corpus_path}")
    print(f"   Diagnostics: {diag_path}")
    if bool(mlx_cfg.get("enabled", False)):
        print(f"   MLX data:    {mlx_cfg.get('dir','data/mlx')}/{run_name}/")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_pipeline.py <config.yaml>")
        raise SystemExit(1)
    run_pipeline(Path(sys.argv[1]))
