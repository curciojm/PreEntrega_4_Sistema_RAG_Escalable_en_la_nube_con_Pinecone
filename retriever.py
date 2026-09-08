from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from vector_db import vectorstore, documentos_procesados


retriever_bm25 = BM25Retriever.from_documents(
    documentos_procesados
)

retriever_bm25.k = 5

# por defecto es similitud
retriever_vectorial = vectorstore.as_retriever(search_type="similarity",
    search_kwargs={"k": 5},
)

retriever_hibrido = EnsembleRetriever(
    retrievers=[retriever_bm25, retriever_vectorial],
    weights=[0.25, 0.75],
)