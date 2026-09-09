import asyncio
from rag import get_rag_response
from db_config import INDEX_NAME, DIMENSIONS
from db_ingest import setup_vector_infrastructure, index
from retrieval import crear_retriever_hibrido


async def main():
    _, vectorstore, documentos_procesados = (
        await setup_vector_infrastructure(INDEX_NAME, DIMENSIONS)
    )
    retriever_hibrido = crear_retriever_hibrido(
        vectorstore,
        documentos_procesados
    )
    # estos prints se deberian quedar son la respuesta del llm
    respuesta_ok = await get_rag_response("¿Que es la correlacion?", retriever_hibrido)
    # Decidi agregar la pregunta original para comparar mejor en el print
    print(f"\nPREGUNTA ORIGINAL: {respuesta_ok.query}")
    print(f"\nRESPUESTA: {respuesta_ok.respuesta}")
    print(f"\nFUENTES: {respuesta_ok.fuentes}")
    print(f"\nPAGINAS: {respuesta_ok.paginas}")
    print(f"\nCATEGORIAS: {respuesta_ok.categorias}")
    print(f"\nCHUNKS: {respuesta_ok.chunks_ids}")
    print(f"\nFragmentos usados: {respuesta_ok.fragmentos_recuperados}")


if __name__ == "__main__":
    asyncio.run(main())