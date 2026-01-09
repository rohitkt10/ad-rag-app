import json

import faiss
import numpy as np
import pytest

from ad_rag_pipeline import indexing


def test_load_chunks(tmp_path):
    chunks_file = tmp_path / "chunks.jsonl"

    # Create test data
    valid_records = [
        {"text": "chunk 1", "pmcid": "PMC1", "meta": "data1"},
        {"text": "chunk 2", "pmcid": "PMC2", "meta": "data2"},
        {"text": "chunk 3", "pmcid": "PMC3", "meta": "data3"},
    ]

    # Write to file with some noise
    with open(chunks_file, "w") as f:
        # Valid records
        for rec in valid_records:
            f.write(json.dumps(rec) + "\n")

        # Invalid JSON line (should be skipped)
        f.write("This is not JSON\n")

    # Test loading
    texts, metas = indexing.load_chunks(chunks_file)

    assert len(texts) == 3
    assert len(metas) == 3

    for i in range(3):
        assert texts[i] == valid_records[i]["text"]
        assert metas[i]["pmcid"] == valid_records[i]["pmcid"]
        assert metas[i]["meta"] == valid_records[i]["meta"]


def test_load_chunks_missing_text_raises_value_error(tmp_path):
    chunks_file = tmp_path / "bad_chunks.jsonl"
    with open(chunks_file, "w") as f:
        f.write(json.dumps({"pmcid": "PMC1"}) + "\n")

    with pytest.raises(ValueError, match="missing required 'text' field"):
        indexing.load_chunks(chunks_file)


def test_build_faiss_index():
    N, d = 4, 3
    embeddings = np.random.rand(N, d).astype(np.float32)

    index = indexing.build_faiss_index(embeddings)

    assert index.ntotal == N
    assert index.d == d
    assert isinstance(index, faiss.IndexFlatIP)


def test_save_artifacts(tmp_path):
    out_dir = tmp_path / "index"

    # Create dummy data
    N, d = 5, 4
    embeddings = np.random.rand(N, d).astype(np.float32)
    index = indexing.build_faiss_index(embeddings)

    metas = [{"id": i, "text": f"text_{i}"} for i in range(N)]
    run_meta = {
        "metric": "cosine",
        "model_id": "test-model",
        "embedding_dim": d,
        "num_chunks": N,
        "source_chunks_path": "/tmp/fake.jsonl",
    }

    # Run save
    faiss_path, lookup_path, meta_path = indexing.save_artifacts(index, metas, out_dir, run_meta)

    # Verify files exist
    assert faiss_path.exists()
    assert lookup_path.exists()
    assert meta_path.exists()

    # Verify FAISS index content
    loaded_index = faiss.read_index(str(faiss_path))
    assert loaded_index.ntotal == N
    assert loaded_index.d == d

    # Verify lookup content
    with open(lookup_path) as f:
        lines = [json.loads(line) for line in f]
        assert len(lines) == N
        for i, line in enumerate(lines):
            assert line["row_id"] == i
            assert line["id"] == i
            assert line["text"] == f"text_{i}"

    # Verify meta content
    with open(meta_path) as f:
        loaded_meta = json.load(f)
        assert loaded_meta["metric"] == "cosine"
        assert loaded_meta["model_id"] == "test-model"
        assert loaded_meta["embedding_dim"] == d
        assert loaded_meta["num_chunks"] == N


def test_build_faiss_index_from_chunks_e2e(tmp_path, monkeypatch):
    chunks_path = tmp_path / "chunks.jsonl"
    out_dir = tmp_path / "output_index"

    N, d = 10, 8

    # Create chunks file
    with open(chunks_path, "w") as f:
        for i in range(N):
            f.write(json.dumps({"text": f"chunk {i}", "id": i}) + "\n")

    # Mock embed_texts to avoid real model usage
    def mock_embed_texts(texts, model_id, batch_size, device, normalize=True):
        assert len(texts) == N
        # Return deterministic embeddings (first N rows of identity matrix)
        # Pad with zeros if d > N, or truncate if N > d (here d=8, N=10, so simple random)
        # To be deterministic and simple:
        rng = np.random.RandomState(42)
        emb = rng.rand(len(texts), d).astype(np.float32)
        if normalize:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / norms
        return emb

    monkeypatch.setattr(indexing, "embed_texts", mock_embed_texts)

    # Run pipeline
    faiss_path, lookup_path, meta_path = indexing.build_faiss_index_from_chunks(
        chunks_path=chunks_path,
        out_dir=out_dir,
        model_id="dummy-model",
        batch_size=2,
        device="cpu",
        metric="cosine",
        force=False,
    )

    # Assertions
    assert faiss_path.exists()
    assert lookup_path.exists()
    assert meta_path.exists()

    loaded_index = faiss.read_index(str(faiss_path))
    assert loaded_index.ntotal == N
    assert loaded_index.d == d

    # Test force=False raises ValueError when index exists
    with pytest.raises(ValueError, match="Index already exists"):
        indexing.build_faiss_index_from_chunks(
            chunks_path=chunks_path,
            out_dir=out_dir,
            model_id="dummy-model",
            batch_size=2,
            device="cpu",
            metric="cosine",
            force=False,
        )

    # Test invalid metric
    with pytest.raises(ValueError, match="Unsupported metric"):
        indexing.build_faiss_index_from_chunks(
            chunks_path=chunks_path,
            out_dir=out_dir,
            model_id="dummy-model",
            batch_size=2,
            device="cpu",
            metric="euclidean",
            force=True,
        )


def test_build_faiss_index_from_chunks_no_chunks(tmp_path):
    chunks_path = tmp_path / "empty.jsonl"
    chunks_path.touch()
    out_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="No chunks found"):
        indexing.build_faiss_index_from_chunks(
            chunks_path=chunks_path, out_dir=out_dir, model_id="dummy", batch_size=1, device="cpu"
        )
