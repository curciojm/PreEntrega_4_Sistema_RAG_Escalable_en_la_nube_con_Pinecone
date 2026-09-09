import pytest

from langchain_core.documents import Document

from schemas import RespuestaLLM, RAGResponse
from rag import formatear_documentos


def test_formatear_documentos():

    docs = [
        Document(
            page_content="La correlación estudia la relación entre dos variables.",
            metadata={
                "fuente": "Pagano 2006",
                "pagina": 10,
                "categoria": "correlacion",
                "chunk_id": 3,
            },
        ),
        Document(
            page_content="La regresión permite realizar predicciones.",
            metadata={
                "fuente": "Sampieri",
                "pagina": 20,
                "categoria": "regresion",
                "chunk_id": 7,
            },
        ),
    ]

    resultado = formatear_documentos(docs)

    assert "[Fuente: Pagano 2006]" in resultado
    assert "[Página: 10]" in resultado
    assert "[Categoría: correlacion]" in resultado
    assert "[Chunk: 3]" in resultado
    assert "La correlación estudia la relación entre dos variables." in resultado

    assert "[Fuente: Sampieri]" in resultado
    assert "[Página: 20]" in resultado
    assert "[Categoría: regresion]" in resultado
    assert "[Chunk: 7]" in resultado
    assert "La regresión permite realizar predicciones." in resultado

    assert "\n\n---\n\n" in resultado


@pytest.mark.asyncio
async def test_get_rag_response(monkeypatch):

    docs = [
        Document(
            page_content="La correlación estudia la relación entre dos variables.",
            metadata={
                "fuente": "Pagano 2006",
                "pagina": 10,
                "categoria": "correlacion",
                "chunk_id": 3,
            },
        ),
        Document(
            page_content="La correlación puede ser positiva o negativa.",
            metadata={
                "fuente": "Pagano 2006",
                "pagina": 11,
                "categoria": "correlacion",
                "chunk_id": 4,
            },
        ),
    ]

    class FakeRetriever:

        async def ainvoke(self, query):
            assert query == "¿Qué es la correlación?"
            return docs

    class FakeParser:

        def get_format_instructions(self):
            return "Formato de prueba"

    class FakeChain:

        async def ainvoke(self, datos):
            assert datos["pregunta"] == "¿Qué es la correlación?"
            assert "La correlación estudia la relación entre dos variables." in datos["contexto"]
            assert datos["formato"] == "Formato de prueba"

            return RespuestaLLM(
                respuesta="La correlación estudia la relación entre dos variables."
            )

    import rag

    monkeypatch.setattr(
        rag,
        "retriever_hibrido",
        FakeRetriever()
    )

    monkeypatch.setattr(
        rag,
        "chain",
        FakeChain()
    )

    monkeypatch.setattr(
        rag,
        "parser_llm",
        FakeParser()
    )

    resultado = await rag.get_rag_response(
        "¿Qué es la correlación?"
    )

    assert isinstance(resultado, RAGResponse)
    assert resultado.query == "¿Qué es la correlación?"
    assert resultado.respuesta == (
        "La correlación estudia la relación entre dos variables."
    )
    assert resultado.fuentes == ["Pagano 2006"]
    assert resultado.paginas == [10, 11]
    assert resultado.categorias == ["correlacion"]
    assert resultado.chunks_ids == [3, 4]
    assert resultado.fragmentos_recuperados == 2