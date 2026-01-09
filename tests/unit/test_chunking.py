from ad_rag_pipeline import chunking

# -------------------------------------------------------------------------
# Test chunk_text_words
# -------------------------------------------------------------------------


def test_chunk_text_words_simple():
    text = "one two three four five six"
    # size 3, overlap 1 -> "one two three", "three four five", "five six"
    chunks = chunking.chunk_text_words(text, chunk_size_words=3, overlap_words=1)
    assert len(chunks) == 3
    assert chunks[0] == "one two three"
    assert chunks[1] == "three four five"
    assert chunks[2] == "five six"


def test_chunk_text_words_no_overlap():
    text = "one two three four"
    chunks = chunking.chunk_text_words(text, chunk_size_words=2, overlap_words=0)
    assert len(chunks) == 2
    assert chunks == ["one two", "three four"]


def test_chunk_text_words_min_words_filter():
    text = "one two three"
    # size 2, overlap 0 -> "one two", "three"
    # "three" has 1 word. If min_words=2, it should be dropped.
    chunks = chunking.chunk_text_words(text, chunk_size_words=2, overlap_words=0, min_words=2)
    assert len(chunks) == 1
    assert chunks == ["one two"]


def test_chunk_text_words_empty_or_small():
    assert chunking.chunk_text_words("", 10, 0) == []
    assert chunking.chunk_text_words("hi", 10, 0, min_words=5) == []


# -------------------------------------------------------------------------
# Test extract_sections_from_pmc_xml
# -------------------------------------------------------------------------

SAMPLE_XML = """
<article>
  <front>
    <article-meta>
      <title-group>
        <article-title>Test Title</article-title>
      </title-group>
      <abstract>
        <p>This is the abstract.</p>
      </abstract>
    </article-meta>
  </front>
  <body>
    <sec>
      <title>Introduction</title>
      <p>Intro paragraph 1.</p>
      <p>Intro paragraph 2.</p>
    </sec>
    <sec>
      <title>Methods</title>
      <p>Methods paragraph.</p>
    </sec>
  </body>
</article>
"""


def test_extract_sections():
    sections = chunking.extract_sections_from_pmc_xml(SAMPLE_XML)

    # Expect: TITLE_ABSTRACT, Introduction, Methods
    assert len(sections) == 3

    s0 = sections[0]
    assert s0["section_title"] == "TITLE_ABSTRACT"
    assert "TITLE: Test Title" in s0["text"]
    assert "ABSTRACT: This is the abstract." in s0["text"]

    s1 = sections[1]
    assert s1["section_title"] == "Introduction"
    assert s1["text"] == "Intro paragraph 1.\nIntro paragraph 2."

    s2 = sections[2]
    assert s2["section_title"] == "Methods"
    assert s2["text"] == "Methods paragraph."


def test_extract_sections_fallback_body():
    xml = """<article><body><p>Just a paragraph.</p></body></article>"""
    sections = chunking.extract_sections_from_pmc_xml(xml)
    assert len(sections) == 1
    assert sections[0]["section_title"] == "BODY"
    assert sections[0]["text"] == "Just a paragraph."


# -------------------------------------------------------------------------
# Test build_chunk_records_for_article
# -------------------------------------------------------------------------


def test_build_chunk_records_for_article(tmp_path):
    xml_file = tmp_path / "PMC999.xml"
    xml_file.write_text(SAMPLE_XML, encoding="utf-8")

    records = chunking.build_chunk_records_for_article(
        xml_file, chunk_size_words=10, overlap_words=0, min_words=1
    )

    assert len(records) > 0
    first = records[0]
    assert first["pmcid"] == "PMC999"
    assert "section_title" in first
    assert "text" in first
    assert "source_xml" in first
    # Check that chunk_index_in_section resets

    # "Introduction" has 2 paras, probably small enough to fit in one chunk of size 10?
    # "Intro paragraph 1." -> 3 words. "Intro paragraph 2." -> 3 words. Total 6 words.
    # So 1 chunk.

    intro_recs = [r for r in records if r["section_title"] == "Introduction"]
    assert len(intro_recs) == 1
    assert intro_recs[0]["chunk_index_in_section"] == 0
