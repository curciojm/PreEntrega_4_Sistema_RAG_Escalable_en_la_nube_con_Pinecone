import asyncio
import os

from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from setup import documentos_procesados


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = "statistics-methodology"

# namespace es para separar vectores directamente. como una biblioteca aparte. distinto a los metadatos que comparten vectores obviamente
NAMESPACE = "Statistics_and_methodolgy_texts"

EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

DIMENSIONS = 384


async def setup_vector_infrastructure(
    INDEX_NAME: str,
    DIMENSIONS: int
):

    pc = Pinecone(api_key=PINECONE_API_KEY)

    if INDEX_NAME not in pc.list_indexes().names():

        print(f"Creando índice: {INDEX_NAME}...")

        pc.create_index(
            name=INDEX_NAME,
            # Asi la dimension de indexacion con la de vectorizacion coinciden
            dimension=DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        while not pc.describe_index(INDEX_NAME).status["ready"]:
            await asyncio.sleep(1)

    index = pc.Index(INDEX_NAME)

    vectorstore = PineconeVectorStore.from_documents(
        documents=documentos_procesados,
        embedding=EMBEDDINGS,
        index_name=INDEX_NAME,
        namespace=NAMESPACE,
    )

    stats = index.describe_index_stats()

    print(
        "📦 Vectores en el namespace:",
        stats["namespaces"].get(NAMESPACE, {})
    )

    print(f"Estado del índice: {stats}")

    return index, vectorstore

index, vectorstore = asyncio.run(
    setup_vector_infrastructure(INDEX_NAME, DIMENSIONS)
)