import json

import faiss
import numpy as np
import pytest

from ad_rag_service.indexing import IndexStore


def test_index_store_load_success(tmp_path):
    # Setup artifacts
    index_path = tmp_path / "faiss.index"
    lookup_path = tmp_path / "lookup.jsonl"
    meta_path = tmp_path / "index.meta.json"

    # 1. Create FAISS index
    d = 4
    n = 3
    index = faiss.IndexFlatL2(d)
    index.add(np.random.rand(n, d).astype(np.float32))
    faiss.write_index(index, str(index_path))

    # 2. Create Lookup
    records = [
        {
            "row_id": 0,
            "text": "t1",
            "pmcid": "P1",
            "section_title": "S1",
            "chunk_index_in_section": 0,
            "source_xml": "x",
        },
        {
            "row_id": 1,
            "text": "t2",
            "pmcid": "P2",
            "section_title": "S2",
            "chunk_index_in_section": 0,
            "source_xml": "x",
        },
        {
            "row_id": 2,
            "text": "t3",
            "pmcid": "P3",
            "section_title": "S3",
            "chunk_index_in_section": 0,
            "source_xml": "x",
        },
    ]
    with open(lookup_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # 3. Create Meta
    with open(meta_path, "w") as f:
        json.dump({"embedding_dim": d, "num_chunks": n}, f)

    # Test
    store = IndexStore(index_path, lookup_path, meta_path)
    store.load()

    assert store.index is not None
    assert store.index.ntotal == n
    assert len(store.lookup) == n
    assert store.lookup[0].text == "t1"
    assert store.meta["embedding_dim"] == d


def test_index_store_missing_files(tmp_path):
    store = IndexStore(tmp_path / "i", tmp_path / "l", tmp_path / "m")
    with pytest.raises(FileNotFoundError):
        store.load()


def test_index_store_consistency_mismatch(tmp_path):
    # Setup valid index (n=1) but empty lookup
    index_path = tmp_path / "faiss.index"
    lookup_path = tmp_path / "lookup.jsonl"
    meta_path = tmp_path / "index.meta.json"

    index = faiss.IndexFlatL2(4)
    index.add(np.random.rand(1, 4).astype(np.float32))
    faiss.write_index(index, str(index_path))

    lookup_path.write_text("")
    meta_path.write_text("{}")

    store = IndexStore(index_path, lookup_path, meta_path)
    with pytest.raises(ValueError, match="Consistency error"):
        store.load()


def test_index_store_dimension_mismatch(tmp_path):
    # Meta says dim=10, index has dim=4
    index_path = tmp_path / "faiss.index"
    lookup_path = tmp_path / "lookup.jsonl"
    meta_path = tmp_path / "index.meta.json"

    index = faiss.IndexFlatL2(4)
    index.add(np.zeros((1, 4), dtype=np.float32))
    faiss.write_index(index, str(index_path))

    # Lookup needs 1 record
    with open(lookup_path, "w") as f:
        f.write(
            json.dumps(
                {
                    "row_id": 0,
                    "text": "t",
                    "pmcid": "p",
                    "section_title": "s",
                    "chunk_index_in_section": 0,
                    "source_xml": "x",
                }
            )
            + "\n"
        )

    with open(meta_path, "w") as f:
        json.dump({"embedding_dim": 10}, f)

    store = IndexStore(index_path, lookup_path, meta_path)
    with pytest.raises(ValueError, match="Dimension mismatch"):
        store.load()
