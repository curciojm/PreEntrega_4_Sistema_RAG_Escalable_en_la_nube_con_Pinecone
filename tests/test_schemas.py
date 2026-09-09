from langchain_core.documents import Document

from schemas import RespuestaLLM, RAGSystem, RAGResponse


def test_respuesta_llm():

    respuesta = RespuestaLLM(
        respuesta="La correlación mide la relación entre dos variables."
    )

    assert isinstance(respuesta, RespuestaLLM)
    assert respuesta.respuesta == (
        "La correlación mide la relación entre dos variables."
    )


def test_rag_system_obtener_top_k():

    documentos = [
        Document(
            page_content="Contenido del documento 1",
            metadata={
                "fuente": "Pagano",
                "pagina": 1,
                "categoria": "correlacion",
                "chunk_id": 0,
            },
        ),
        Document(
            page_content="Contenido del documento 2",
            metadata={
                "fuente": "Sampieri",
                "pagina": 2,
                "categoria": "probabilidad",
                "chunk_id": 1,
            },
        ),
        Document(
            page_content="Contenido del documento 3",
            metadata={
                "fuente": "Pagano",
                "pagina": 3,
                "categoria": "regresion",
                "chunk_id": 2,
            },
        ),
    ]

    class FakeRetriever:

        def invoke(self, query):
            assert query == "¿Qué es la correlación?"
            return documentos

    rag_system = RAGSystem(
        retriever=FakeRetriever(),
        k=2
    )

    resultado = rag_system.obtener_top_k(
        "¿Qué es la correlación?"
    )

    assert len(resultado) == 2

    assert resultado[0]["contenido"] == "Contenido del documento 1"
    assert resultado[0]["fuente"] == "Pagano"
    assert resultado[0]["pagina"] == 1
    assert resultado[0]["categoria"] == "correlacion"
    assert resultado[0]["chunk_id"] == 0

    assert resultado[1]["contenido"] == "Contenido del documento 2"


def test_rag_response():

    respuesta = RAGResponse(
        query="¿Qué es la correlación?",
        respuesta="La correlación estudia la relación entre dos variables.",
        fuentes=["Pagano"],
        paginas=[1, 2],
        categorias=["correlacion"],
        chunks_ids=[0, 1],
        fragmentos_recuperados=2,
    )

    assert respuesta.query == "¿Qué es la correlación?"
    assert respuesta.respuesta == (
        "La correlación estudia la relación entre dos variables."
    )
    assert respuesta.fuentes == ["Pagano"]
    assert respuesta.paginas == [1, 2]
    assert respuesta.categorias == ["correlacion"]
    assert respuesta.chunks_ids == [0, 1]
    assert respuesta.fragmentos_recuperados == 2