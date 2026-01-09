from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any


def _text(elem: ET.Element | None) -> str:
    """Extract and normalize text from an XML element."""
    if elem is None:
        return ""
    return "".join(elem.itertext()).strip()


def extract_basic_metadata(root: ET.Element) -> dict[str, Any]:
    """
    Extract minimal bibliographic metadata (journal, DOI, date) from the XML root.
    """
    journal = _text(root.find(".//journal-title")) or None
    doi = _text(root.find(".//article-id[@pub-id-type='doi']")) or None

    pub_date = root.find(".//pub-date[@pub-type='epub']")
    if pub_date is None:
        pub_date = root.find(".//pub-date")
    year = _text(pub_date.find("year")) if pub_date is not None else ""
    month = _text(pub_date.find("month")) if pub_date is not None else ""

    return {
        "journal": journal,
        "doi": doi,
        "year": year or None,
        "month": month or None,
    }


def extract_sections_from_pmc_xml(xml_text: str | bytes) -> list[dict[str, str]]:
    """
    Parse PMC XML and extract sections.
    Returns a list of dicts:
      {"section_title": str, "text": str, "section_type": str}
    """
    if isinstance(xml_text, bytes):
        root = ET.fromstring(xml_text)
    else:
        root = ET.fromstring(xml_text)

    sections: list[dict[str, str]] = []

    # 1. Title + Abstract
    title = _text(root.find(".//article-title"))
    abstracts = root.findall(".//abstract")
    abstract_texts = [_text(a) for a in abstracts]
    abstract_text = "\n\n".join([t for t in abstract_texts if t])

    ta_parts: list[str] = []
    if title:
        ta_parts.append(f"TITLE: {title}")
    if abstract_text:
        ta_parts.append(f"ABSTRACT: {abstract_text}")

    if ta_parts:
        sections.append(
            {
                "section_title": "TITLE_ABSTRACT",
                "text": "\n\n".join(ta_parts).strip(),
                "section_type": "TITLE_ABSTRACT",
            }
        )

    # 2. Body Sections
    body = root.find(".//body")
    if body is not None:
        top_secs = body.findall("./sec")
        if top_secs:
            for sec in top_secs:
                sec_title = _text(sec.find("title")) or "SECTION"
                paras = []
                for p in sec.findall(".//p"):
                    t = _text(p)
                    if t:
                        paras.append(t)
                sec_text = "\n".join(paras).strip()
                if sec_text:
                    sections.append(
                        {"section_title": sec_title, "text": sec_text, "section_type": "BODY_SEC"}
                    )
        else:
            # Fallback: body without top-level <sec>
            paras = []
            for p in body.findall(".//p"):
                t = _text(p)
                if t:
                    paras.append(t)
            body_text = "\n".join(paras).strip()
            if body_text:
                sections.append(
                    {"section_title": "BODY", "text": body_text, "section_type": "BODY_FALLBACK"}
                )

    return sections


def chunk_text_words(
    text: str, chunk_size_words: int, overlap_words: int, min_words: int = 1
) -> list[str]:
    """
    Split text into overlapping word chunks.
    """
    words = text.split()
    if len(words) < min_words:
        return []

    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be > 0")
    if overlap_words < 0 or overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must satisfy 0 <= overlap < chunk_size")

    if len(words) <= chunk_size_words:
        return [" ".join(words)]

    chunks: list[str] = []
    step = chunk_size_words - overlap_words
    start = 0
    while start < len(words):
        end = start + chunk_size_words
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start += step

    # Filter by min_words again on the generated chunks to be safe/consistent
    final_chunks = [c for c in chunks if len(c.split()) >= min_words]
    return final_chunks


def build_chunk_records_for_article(
    xml_path: Path,
    chunk_size_words: int,
    overlap_words: int,
    min_words: int,
    pmid_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """
    Parse one XML file and return a list of chunk records (dicts).
    """
    pmid_map = pmid_map or {}
    pmcid = xml_path.stem  # e.g., "PMC123456"
    pmid = pmid_map.get(pmcid)

    try:
        xml_bytes = xml_path.read_bytes()
        # Parse for metadata first (requires root element)
        root = ET.fromstring(xml_bytes)
        base_md = extract_basic_metadata(root)

        # Parse for sections (re-parses or we could refactor to pass root,
        # but extract_sections takes str/bytes per spec. Let's stick to spec but optimize if needed.
        # Actually, let's just pass the bytes to extract_sections_from_pmc_xml)
        sections = extract_sections_from_pmc_xml(xml_bytes)
    except Exception as e:
        print(f"Failed parse {xml_path.name}: {type(e).__name__}: {e}")
        return []

    records = []
    # This is per-article here, but script did global ID.
    # The script used a global `chunk_id`.
    # The function signature returns a list, so we can assign global IDs later
    # OR we just return the records and let the caller assign/fix global IDs?
    # The plan says "Produces the same JSON schema".
    # The script had "chunk_id" which was global.
    # I will omit "chunk_id" here and let the orchestrator add it,
    # or I'll add a placeholder.
    # Actually, better to yield records and let orchestrator count.

    # We will return records without global 'chunk_id' here,
    # or with a local one if the caller wants to re-map.
    # Let's look at the script: "chunk_id" is monotonic across the whole dataset.
    # So `build_chunk_records_for_article` cannot know the global `chunk_id` start.
    # I will omit `chunk_id` in this function's output and add it in `build_chunks_dataset`.

    for sec_idx, section in enumerate(sections):
        sec_title = section["section_title"]
        sec_text = section["text"]

        chunks = chunk_text_words(
            sec_text,
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
            min_words=min_words,
        )

        for i, chunk_text in enumerate(chunks):
            rec = {
                # "chunk_id": ... handled by caller
                "pmcid": pmcid,
                "pmid": pmid,
                "section_index": sec_idx,
                "section_title": sec_title,
                "chunk_index_in_section": i,
                "text": chunk_text,
                **base_md,
                "source_xml": str(xml_path),
            }
            records.append(rec)

    return records


def build_chunks_dataset(
    raw_dir: Path,
    out_dir: Path,
    chunk_size_words: int,
    overlap_words: int,
    min_words: int,
    manifest_path: Path | None = None,
) -> tuple[Path, Path]:
    """
    Iterates over raw_dir/PMC*.xml, chunks them, and writes to out_dir.
    Returns (chunks_path, meta_path).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chunks.jsonl"

    pmid_map = {}
    if manifest_path and manifest_path.exists():
        # Load pmid map logic here or reuse utility?
        # I'll implement a simple loader here to be self-contained or add a helper.
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("type") == "article" and rec.get("ok"):
                        p_pmcid = rec.get("pmcid")
                        p_pmid = rec.get("pmid")
                        if p_pmcid and p_pmid:
                            pmid_map[p_pmcid] = p_pmid
                except Exception:
                    pass

    xml_files = sorted(raw_dir.glob("PMC*.xml"))

    run_meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "raw_dir": str(raw_dir),
        "out": str(out_path),
        "num_xml_files": len(xml_files),
        "chunk_size": chunk_size_words,
        "overlap": overlap_words,
        "min_words": min_words,
        "manifest_used": str(manifest_path) if manifest_path and manifest_path.exists() else None,
    }

    global_chunk_id = 0
    written_count = 0

    with out_path.open("w", encoding="utf-8") as out_f:
        for xml_path in xml_files:
            records = build_chunk_records_for_article(
                xml_path, chunk_size_words, overlap_words, min_words, pmid_map
            )

            for rec in records:
                rec["chunk_id"] = global_chunk_id
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                global_chunk_id += 1
                written_count += 1

    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    return out_path, meta_path
