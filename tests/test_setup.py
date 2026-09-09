
from langchain_core.documents import Document

from setup import DocumentProcessor


def test_process_document_crea_chunks():
    processor = DocumentProcessor()

    documento = Document(
        page_content=(
            "La regresión es una técnica estadística utilizada para realizar "
            "predicciones. La correlación estudia la relación entre dos variables."
        ),
        metadata={
            "source": (
                "Pagano 2006 CAP 6 - "
                "Estadística para las ciencias del comportamiento, correlacion.pdf"
            ),
            "page": 0,
        },
    )

    chunks = processor.process_document([documento])

    # Debe devolver una lista
    assert isinstance(chunks, list)

    # Debe haber generado al menos un chunk
    assert len(chunks) > 0

    # Cada elemento debe ser un Document
    assert all(isinstance(chunk, Document) for chunk in chunks)

    # Ningún chunk debe estar vacío
    assert all(chunk.page_content.strip() for chunk in chunks)


def test_process_document_agrega_metadata():
    processor = DocumentProcessor()

    documento = Document(
        page_content=(
            "La correlación estudia la relación entre dos variables."
        ),
        metadata={
            "source": (
                "Pagano 2006 CAP 6 - "
                "Estadística para las ciencias del comportamiento, correlacion.pdf"
            ),
            "page": 0,
        },
    )

    chunks = processor.process_document([documento])

    assert len(chunks) > 0

    metadata = chunks[0].metadata

    # Fuente obtenida del nombre del archivo
    assert metadata["fuente"] == (
        "Pagano 2006 CAP 6 - "
        "Estadística para las ciencias del comportamiento"
    )

    # Categoría obtenida del nombre del archivo
    assert metadata["categoria"] == "correlacion"

    # PyPDFLoader utiliza páginas comenzando desde 0.
    # Nuestro pipeline las convierte a numeración humana comenzando desde 1.
    assert metadata["pagina"] == 1

    # El primer chunk debe recibir el ID 0
    assert metadata["chunk_id"] == 0