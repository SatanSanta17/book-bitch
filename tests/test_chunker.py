from app.chunker import clean_text, chunk_text


def test_clean_text_removes_extra_whitespace():
    assert clean_text("Hello   world \n") == "Hello world"


def test_chunk_text_respects_overlap():
    text = "A" * 2000
    chunks = chunk_text(text, max_tokens=100, overlap=20)
    assert len(chunks) > 1
    assert chunks[0][-10:] == chunks[1][:10]