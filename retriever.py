from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from setup import documentos_procesados
from vector_db import vectorstore


# Retriever lexico, maximo 5 cada uno
retriever_bm25 = BM25Retriever.from_documents(documentos_procesados)
retriever_bm25.k = 5

# Retriever vectorial, max 5 cada uno
retriever_vectorial = vectorstore.as_retriever(
    search_kwargs={"k": 5},
)

# Retriever hibrido, max 10
retriever_hibrido = EnsembleRetriever(
    retrievers=[retriever_bm25, retriever_vectorial],
    weights=[0.5, 0.5],
)