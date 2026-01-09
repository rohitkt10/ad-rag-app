import pytest

from ad_rag_pipeline import embedding


def test_embed_texts_empty_input_raises_value_error():
    with pytest.raises(ValueError, match="The 'texts' list is empty"):
        embedding.embed_texts(texts=[], model_id="dummy", batch_size=1, device="cpu")
