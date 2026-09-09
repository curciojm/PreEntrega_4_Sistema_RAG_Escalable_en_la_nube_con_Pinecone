from langchain_core.documents import Document

from setup import procesamiento_desde_pdfs


def test_procesamiento_desde_pdfs(monkeypatch):

    documentos_crudos = [
        Document(
            page_content="Texto de prueba",
            metadata={"source": "prueba.pdf", "page": 0}
        )
    ]

    documentos_procesados = [
        Document(
            page_content="Texto procesado",
            metadata={"fuente": "prueba"}
        )
    ]

    class FakeLoader:

        def __init__(self, *args, **kwargs):
            pass

        def load(self):
            return documentos_crudos

    class FakeProcessor:

        def process_document(self, documentos):
            assert documentos == documentos_crudos
            return documentos_procesados

    monkeypatch.setattr(
        "setup.DirectoryLoader",
        FakeLoader
    )

    monkeypatch.setattr(
        "setup.DocumentProcessor",
        FakeProcessor
    )

    resultado = procesamiento_desde_pdfs()

    assert resultado == documentos_procesados


def test_recuperar_documentos_de_pinecone():
    from setup import recuperar_documentos_de_pinecone

    class FakeVector:
        def __init__(self, metadata):
            self.metadata = metadata

    class FakeFetchResult:
        def __init__(self):
            self.vectors = {
                "id-1": FakeVector({
                    "text": "Texto del primer chunk",
                    "fuente": "Pagano 2006",
                    "pagina": 1,
                    "categoria": "correlacion",
                    "chunk_id": 0,
                }),
                "id-2": FakeVector({
                    "text": "Texto del segundo chunk",
                    "fuente": "Sampieri",
                    "pagina": 2,
                    "categoria": "probabilidad",
                    "chunk_id": 1,
                }),
            }

    class FakeIndex:

        def list(self, namespace):
            return [["id-1", "id-2"]]

        def fetch(self, ids, namespace):
            return FakeFetchResult()

    index = FakeIndex()

    documentos = recuperar_documentos_de_pinecone(index)

    assert len(documentos) == 2
    assert documentos[0].page_content == "Texto del primer chunk"
    assert documentos[1].page_content == "Texto del segundo chunk"
    assert documentos[0].metadata["fuente"] == "Pagano 2006"
    assert documentos[1].metadata["categoria"] == "probabilidad"