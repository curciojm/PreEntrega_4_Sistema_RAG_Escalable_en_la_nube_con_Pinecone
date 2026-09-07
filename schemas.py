from typing import List, Dict
from enum import Enum
from pydantic import BaseModel, Field


# Respuesta del modelo
class RespuestaLLM(BaseModel):
    """Lo que el LLM debe generar, parseado directamente de su output."""

    respuesta: str = Field(
        description=(
            "Respuesta a la pregunta del usuario, basada EXCLUSIVAMENTE en el CONTEXTO. "
            "Si la información no está en el contexto, decir explícitamente que no se cuenta con esa información."
        )
    )


# RAG
# Sistema de BÚSQUEDA
class RAGSystem:
    """Encapsula el EnsembleRetriever y expone un método simple para obtener el top-k."""

    def __init__(self, retriever, k: int = 5):
        self.retriever = retriever
        self.k = k

    def obtener_top_k(self, query: str) -> List[Dict]:
        docs = self.retriever.invoke(query)[:self.k]

        return [
            {
                "contenido": d.page_content,
                "fuente": d.metadata.get("fuente", "desconocida"),
                "pagina": d.metadata.get("pagina", "desconocida"),
                "categoria": d.metadata.get("categoria", "desconocida"),
                "chunk_id": d.metadata.get("chunk_id", "desconocida"),
            }
            for d in docs
        ]


# Sistema de RESPUESTA
class RAGResponse:
    def __init__(
        self,
        query: str,
        respuesta: str,
        fuentes: List[str],
        paginas: List[int],
        categorias: List[str],
        chunks_ids: List[int],
        fragmentos_recuperados: List[int],
    ):  
        self.query = query
        self.respuesta = respuesta
        self.fuentes = fuentes
        self.paginas = paginas
        self.categorias = categorias
        self.chunks_ids = chunks_ids
        self.fragmentos_recuperados = fragmentos_recuperados


# Errores posibles
class LLMErrorType(str, Enum):
    """Tipos de errores utilizados para clasificar las excepciones."""

    RATE_LIMIT = "rate_limit"
    UNKNOWN = "unknown"


class LLMError(Exception):
    """Permite clasificar el error y proporcionar un mensaje legible al usuario."""

    def __init__(self, error_type: str, message: str):
        self.error_type = error_type
        self.message = message
        super().__init__(message)