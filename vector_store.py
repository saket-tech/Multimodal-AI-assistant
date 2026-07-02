"""
Vector store for financial document context retrieval.
Stores prior document extractions and billing policies so the assistant
can retrieve supporting context when answering user questions.
"""

import os
from functools import lru_cache

import chromadb
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

DB_PATH = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = "financial_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Built-in billing policy context loaded on startup
_BILLING_POLICIES = [
    "Amazon Web Services (AWS) EC2 charges appear when cloud virtual machines are running. Charges are based on instance type and hours used.",
    "Netflix subscription is a recurring monthly charge for streaming service access, typically $15.99 for standard plan.",
    "Taxes and fees on credit card statements include state sales tax, service fees, and applicable surcharges.",
    "Payment received reduces your outstanding balance. Payments are applied to the oldest charges first.",
    "New charges section lists all purchases made during the billing period before the statement closing date.",
    "Previous balance is the amount owed from the prior billing cycle before any new payments or charges.",
    "Grocery store purchases are retail transactions at supermarkets or food stores.",
    "A billing cycle typically runs 28-31 days. The statement date marks the end of the billing period.",
    "Minimum payment due is the smallest amount you can pay to keep your account in good standing.",
    "Interest charges appear when the full balance is not paid by the due date.",
]


class FinancialVectorStore:
    """ChromaDB-backed vector store for financial document context."""

    def __init__(self) -> None:
        os.makedirs(DB_PATH, exist_ok=True)
        self._model = SentenceTransformer(EMBED_MODEL)
        self._client = chromadb.PersistentClient(path=DB_PATH)
        self._collection: Collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._seed_policies()

    def _seed_policies(self) -> None:
        """Load built-in billing policies if collection is empty."""
        if self._collection.count() == 0:
            self.add_documents(
                _BILLING_POLICIES,
                [{"source": "billing_policy", "type": "policy"} for _ in _BILLING_POLICIES],
            )

    def add_documents(self, texts: list[str], metadatas: list[dict] | None = None) -> None:
        if not texts:
            return
        embeddings = self._model.encode(texts).tolist()
        start = self._collection.count()
        ids = [f"doc_{start + i}" for i in range(len(texts))]
        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas or [{}] * len(texts),
        )

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Return top_k most relevant context chunks for a query."""
        if self._collection.count() == 0:
            return []
        embedding = self._model.encode([query]).tolist()
        results = self._collection.query(
            query_embeddings=embedding,
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        return [
            {
                "content": doc,
                "source": meta.get("source", "unknown"),
                "score": round(1 - dist, 4),
            }
            for doc, meta, dist in zip(docs, metas, dists)
        ]

    def add_extracted_document(self, extracted_text: str, filename: str) -> None:
        """Store a previously analyzed document for future retrieval."""
        self.add_documents(
            [extracted_text],
            [{"source": filename, "type": "extracted_document"}],
        )


@lru_cache(maxsize=1)
def get_vector_store() -> FinancialVectorStore:
    return FinancialVectorStore()
