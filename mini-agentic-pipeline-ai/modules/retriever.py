import os
from typing import List, Dict, Any
from dataclasses import dataclass

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    faiss = None  # type: ignore
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore
try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv() -> None:  # type: ignore
        return None
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

@dataclass
class RetrievedDocument:
    content: str
    metadata: Dict[str, Any]
    score: float

class Retriever:
    def __init__(self, data_dir: str, index_dir: str = "vectorstore", chunk_size: int = 600, chunk_overlap: int = 80):
        load_dotenv()
        self.data_dir = data_dir
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index = None
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.client = None
        self.offline_mode = False
        api_key_present = bool(os.getenv("OPENAI_API_KEY"))
        if api_key_present and OpenAI is not None:
            try:
                self.client = OpenAI()
            except Exception:
                self.offline_mode = True
        else:
            self.offline_mode = True
        if faiss is None or np is None:
            # If FAISS not available, force offline simple retrieval
            self.offline_mode = True
        self._offline_tokens: List[set[str]] = []

    def _load_docs(self) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        for fname in sorted(os.listdir(self.data_dir)):
            fpath = os.path.join(self.data_dir, fname)
            if not os.path.isfile(fpath):
                continue
            if not fname.lower().endswith((".txt", ".md")):
                continue
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            docs.append({"content": content, "source": fpath})
        return docs

    def _split_text(self, text: str) -> List[str]:
        chunks: List[str] = []
        if not text:
            return chunks
        i = 0
        stride = self.chunk_size - self.chunk_overlap if self.chunk_size > self.chunk_overlap else self.chunk_size
        while i < len(text):
            chunk = text[i : i + self.chunk_size]
            chunks.append(chunk)
            if stride <= 0:
                break
            i += stride
        return chunks

    def _split_docs(self, docs: List[Dict[str, Any]]) -> None:
        for d in docs:
            for chunk in self._split_text(d["content"]):
                self.texts.append(chunk)
                self.metadatas.append({"source": d["source"]})

    def _embed_texts(self, texts: List[str]):
        assert self.client is not None, "OpenAI client not initialized"
        resp = self.client.embeddings.create(model="text-embedding-3-small", input=texts)
        vectors = [item.embedding for item in resp.data]
        vectors_np = np.array(vectors, dtype="float32") if np is not None else vectors  # type: ignore
        if faiss is not None and np is not None:
            faiss.normalize_L2(vectors_np)
        return vectors_np

    def _embed_query(self, query: str):
        assert self.client is not None, "OpenAI client not initialized"
        resp = self.client.embeddings.create(model="text-embedding-3-small", input=[query])
        q = np.array([resp.data[0].embedding], dtype="float32") if np is not None else [resp.data[0].embedding]  # type: ignore
        if faiss is not None and np is not None:
            faiss.normalize_L2(q)
        return q

    def build_index(self) -> None:
        docs = self._load_docs()
        self._split_docs(docs)
        if not self.texts:
            raise RuntimeError("No documents found to index")
        if self.offline_mode:
            # Precompute tokens for simple lexical retrieval
            self._offline_tokens = [set(t.lower() for t in text.split()) for text in self.texts]
            self.index = "offline"
        else:
            vectors_np = self._embed_texts(self.texts)
            assert faiss is not None and np is not None
            dim = int(vectors_np.shape[1])  # type: ignore[index]
            index = faiss.IndexFlatIP(dim)
            index.add(vectors_np)  # type: ignore[arg-type]
            self.index = index

    def search(self, query: str, k: int = 5) -> List[RetrievedDocument]:
        if self.index is None:
            self.build_index()
        assert self.index is not None
        results: List[RetrievedDocument] = []
        if self.offline_mode or self.index == "offline":
            # Simple lexical overlap scoring
            q_tokens = set(query.lower().split())
            scored = []
            for i, tset in enumerate(self._offline_tokens):
                overlap = len(q_tokens.intersection(tset))
                if overlap > 0:
                    scored.append((float(overlap), i))
            scored.sort(reverse=True)
            for score, idx in scored[:k]:
                results.append(RetrievedDocument(content=self.texts[idx], metadata=self.metadatas[idx], score=score))
            return results
        else:
            q = self._embed_query(query)
            assert faiss is not None
            scores, idxs = self.index.search(q, k)  # type: ignore[arg-type]
            for score, idx in zip(scores[0], idxs[0]):
                if idx == -1:
                    continue
                results.append(RetrievedDocument(content=self.texts[idx], metadata=self.metadatas[idx], score=float(score)))
            return results

