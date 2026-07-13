"""RAG: ingest a document, then retrieve the chunks relevant to a question.

IngestionPipeline splits and indexes documents; RagRetriever reads them back. Hermetic:
FakeEmbedder + local SQLite backend, with ScriptedLLM standing in for the optional
query-rewrite model. In production use OpenAIEmbedder() and a real LLMClient.

    uv run python examples/05_rag.py
"""
from agentmaker import IngestionPipeline, RagRetriever, SourceStore
from agentmaker.retrieval import build_sqlite_hybrid
from agentmaker.testing import FakeEmbedder, ScriptedLLM

retriever = build_sqlite_hybrid(FakeEmbedder())
source_store = SourceStore()

pipeline = IngestionPipeline(retriever=retriever, source_store=source_store)
report = pipeline.ingest_text(
    "# Expense Policy\n"
    "## Meals\nThe daily meal allowance is 80, no receipt needed.\n\n"
    "## Lodging\nHotels are capped at 500 per night, receipt required.",
    source="policy.md", fmt="md",
)
print(f"Ingested {report.chunks} chunks.\n")

rag = RagRetriever(retriever, source_store, ScriptedLLM([]))
print("Relevant chunks for 'how much can I spend on meals':")
for chunk in rag.retrieve("how much can I spend on meals", top_k=2):
    print("  -", chunk.content)
