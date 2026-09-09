from langchain_core.documents import Document

from chunking import DocumentProcessor


def test_clean_text():
    processor = DocumentProcessor()

    texto = "Hola    mundo\n\n\n\n  esto   es una prueba"

    resultado = processor.clean_text(texto)

    assert resultado == "Hola mundo\n\nesto es una prueba"


def test_calculate_tokens():
    processor = DocumentProcessor()

    texto = "La correlación estudia la relación entre dos variables."

    resultado = processor.calculate_tokens(texto)

    assert isinstance(resultado, int)
    assert resultado > 0


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

    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(chunk, Document) for chunk in chunks)
    assert all(chunk.page_content.strip() for chunk in chunks)


def test_process_document_agrega_metadata():
    processor = DocumentProcessor()

    documento = Document(
        page_content="La correlación estudia la relación entre dos variables.",
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

    assert metadata["fuente"] == (
        "Pagano 2006 CAP 6 - "
        "Estadística para las ciencias del comportamiento"
    )
    assert metadata["categoria"] == "correlacion"
    assert metadata["pagina"] == 1
    assert metadata["chunk_id"] == 0


# ¿Qué estamos comprobando?

# Los cuatro tests cubren las partes importantes de DocumentProcessor:

# clean_text → que la limpieza de espacios y saltos de línea funcione.
# calculate_tokens → que el conteo mediante tiktoken produzca un número válido.
# process_document_crea_chunks → que efectivamente transforme los documentos en chunks.
# process_document_agrega_metadata → que agregue correctamente fuente, pagina, categoria y chunk_id.