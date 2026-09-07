from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

from chunking import DocumentProcessor


loader = DirectoryLoader(
    "data",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

documentos_crudos = loader.load()

processor = DocumentProcessor()

documentos_procesados = processor.process_document(documentos_crudos)