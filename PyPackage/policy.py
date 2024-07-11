import requests
from bs4 import BeautifulSoup
import numpy as np
import openai
from langchain_core.tools import tool
import config


def convertToText() -> str:
    response = requests.get(
        "https://www.flytoday.ir/faq"
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    questions = soup.select(".faq_accordionNestedTitle__eL1zw")
    return [{"page_content": question.string} for question in questions]


class VectorStoreRetriever:
    def __init__(self, docs: list, vectors: list, oai_client):
        docs = convertToText()
        print(docs)
        self._arr = np.array(vectors)
        self._docs = docs
        self._client = oai_client

    @classmethod
    def from_docs(cls, docs, oai_client):
        embeddings = oai_client.embeddings.create(
            model="text-embedding-3-small", input=[doc["page_content"] for doc in docs]
        )
        vectors = [emb.embedding for emb in embeddings.data]
        return cls(docs, vectors, oai_client)

    def query(self, query: str, k: int = 5) -> list[dict]:
        embed = self._client.embeddings.create(
            model="text-embedding-3-small", input=[query]
        )
        # "@" is just a matrix multiplication in python
        scores = np.array(embed.data[0].embedding) @ self._arr.T
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]
        return [
            {**self._docs[idx], "similarity": scores[idx]} for idx in top_k_idx_sorted
        ]


retriever = None


class Policy:

    def __init__(self) -> None:
        # openai.api_key = config.OPENAI_API_KEY
        docs = convertToText()
        retriever = VectorStoreRetriever.from_docs(docs, openai.Client())

    @tool
    def lookup_policy(query: str) -> str:
        """Consult the company policies to check whether certain options are permitted.
        Use this before making any flight changes performing other 'write' events."""
        docs = retriever.query(query, k=2)
        return "\n\n".join([doc["page_content"] for doc in docs])
