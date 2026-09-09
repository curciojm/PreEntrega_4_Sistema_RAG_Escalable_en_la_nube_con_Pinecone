from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document

import os
import re
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from logging_config import logger

from db_ingest import NAMESPACE
# Preprocesamiento, chunking e ingesta

# Procesar chunkings
class DocumentProcessor:

    def __init__(self, model_encoding: str = "cl100k_base"):
        self.tokenizer = tiktoken.get_encoding(model_encoding)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
            length_function=self.calculate_tokens,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    def clean_text(self, text: str) -> str:
        """Limpia espacios excesivos manteniendo la estructura del documento."""

        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n[ \t]+", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def calculate_tokens(self, text: str) -> int:
        """Calcula la cantidad de tokens usando tiktoken."""
        return len(self.tokenizer.encode(text))

    def process_document(
        self,
        documents: list[Document]
    ) -> list[Document]:
        """Limpieza, chunking y enriquecimiento de metadata."""

        processed_chunks = []

        for document in documents:

            # 1. Limpiar el contenido de la página
            cleaned_text = self.clean_text(document.page_content)

            # 2. Crear chunks conservando la metadata original
            chunks = self.splitter.create_documents(
                [cleaned_text],
                metadatas=[document.metadata]
            )

            processed_chunks.extend(chunks)

        # 3. Agregar metadata avanzada
        for i, chunk in enumerate(processed_chunks):

            nombre_archivo = os.path.basename(
                chunk.metadata["source"]
            )

            nombre_sin_extension = os.path.splitext(
                nombre_archivo
            )[0]

            # las categorias se forman a partir de la segunda mitad del titulo del texto
            fuente, categoria = nombre_sin_extension.split(
                ",",
                maxsplit=1
            )

            chunk.metadata["fuente"] = fuente.strip()

            chunk.metadata["pagina"] = int(chunk.metadata["page"]) + 1
            
            chunk.metadata["categoria"] = categoria.strip()

            # por alguna razon pincecone asigna aveces float
            chunk.metadata["chunk_id"] = int(i)

            token_count = self.calculate_tokens(
                chunk.page_content
            )

            logger.info(
                f"Chunk {i} creado: {token_count} tokens "
                f"(página {chunk.metadata['pagina']})."
            )

        return processed_chunks

# Cargar pdfs
def procesamiento_desde_pdfs():
    print("📄 Pinecone vacío. Procesando PDFs...")

    loader = DirectoryLoader(
        "data",
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documentos_crudos = loader.load()

    processor = DocumentProcessor()

    return processor.process_document(documentos_crudos)

# Si ya estan cargados
def recuperar_documentos_de_pinecone(index):
    print("📦 Recuperando documentos desde Pinecone...")

    ids = []

    for pagina in index.list(namespace=NAMESPACE):
        ids.extend(pagina)

    documentos = []

    batch_size = 100

    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]

        registros = index.fetch(
            ids=batch_ids,
            namespace=NAMESPACE
        )

        for vector in registros.vectors.values():
            metadata = vector.metadata

            documentos.append(
                Document(
                    page_content=metadata["text"],
                    metadata=metadata
                )
            )

    print(f"📄 Documentos recuperados: {len(documentos)}")

    return documentos