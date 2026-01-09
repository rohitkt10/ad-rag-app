import json

from ad_rag_pipeline import chunking


def test_build_chunks_dataset_integration(tmp_path):
    # Setup directories
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "chunks"

    # Create fake XMLs
    xml1 = raw_dir / "PMC1.xml"
    xml1.write_text("<article><body><p>Doc 1 content here.</p></body></article>")

    xml2 = raw_dir / "PMC2.xml"
    xml2.write_text("<article><body><p>Doc 2 content here.</p></body></article>")

    # Create manifest
    manifest_path = raw_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        f.write(json.dumps({"type": "article", "ok": True, "pmcid": "PMC1", "pmid": "111"}) + "\n")
        f.write(json.dumps({"type": "article", "ok": True, "pmcid": "PMC2", "pmid": "222"}) + "\n")

    # Run dataset builder
    chunks_path, meta_path = chunking.build_chunks_dataset(
        raw_dir=raw_dir,
        out_dir=out_dir,
        chunk_size_words=50,
        overlap_words=10,
        min_words=1,
        manifest_path=manifest_path,
    )

    assert chunks_path.exists()
    assert meta_path.exists()

    # Check chunks content
    lines = []
    with open(chunks_path) as f:
        for line in f:
            lines.append(json.loads(line))

    assert len(lines) == 2

    # Check global chunk ids
    assert lines[0]["chunk_id"] == 0
    assert lines[1]["chunk_id"] == 1

    # Check PMIDs mapped correctly
    pmc1_rec = next(r for r in lines if r["pmcid"] == "PMC1")
    assert pmc1_rec["pmid"] == "111"
    assert pmc1_rec["text"] == "Doc 1 content here."

    pmc2_rec = next(r for r in lines if r["pmcid"] == "PMC2")
    assert pmc2_rec["pmid"] == "222"

    # Check metadata
    meta = json.loads(meta_path.read_text())
    assert meta["num_xml_files"] == 2
    assert meta["chunk_size"] == 50
