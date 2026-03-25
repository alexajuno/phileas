"""Tests for ChromaDB vector store."""

from phileas.vector import VectorStore


def test_add_and_search(chroma_path):
    vs = VectorStore(path=chroma_path)
    vs.add("mem-1", "Giao is building Phileas with Python")
    vs.add("mem-2", "Alice works at a robotics startup")
    results = vs.search("Python programming projects", top_k=2)
    assert len(results) > 0
    # mem-1 should rank higher for Python-related query
    assert results[0][0] == "mem-1"
    vs.close()


def test_dedup_check(chroma_path):
    vs = VectorStore(path=chroma_path)
    vs.add("mem-1", "Giao likes coffee in the morning")
    duplicate = vs.find_duplicate("Giao enjoys morning coffee", threshold=0.85)
    assert duplicate is not None
    assert duplicate == "mem-1"
    vs.close()


def test_no_false_dedup(chroma_path):
    vs = VectorStore(path=chroma_path)
    vs.add("mem-1", "Giao likes coffee in the morning")
    result = vs.find_duplicate("Alice works at a robotics startup", threshold=0.85)
    assert result is None
    vs.close()


def test_delete(chroma_path):
    vs = VectorStore(path=chroma_path)
    vs.add("mem-1", "test memory")
    vs.delete("mem-1")
    results = vs.search("test memory", top_k=5)
    assert len(results) == 0
    vs.close()


def test_count(chroma_path):
    vs = VectorStore(path=chroma_path)
    vs.add("mem-1", "first")
    vs.add("mem-2", "second")
    assert vs.count() == 2
    vs.close()
