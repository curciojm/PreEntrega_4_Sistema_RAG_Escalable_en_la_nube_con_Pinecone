import asyncio
from logging_config import logger

from schemas import RAGSystem, RAGResponse, RespuestaLLM
from retriever import retriever_hibrido
from errors import classify_error
from chain import chain, parser_llm


# Formatear el documento para que lo lea el LLM
# documentos_procesados es el procesamiento para generar los embedings
def formatear_documentos(docs) -> str:
    return "\n\n---\n\n".join(
        f"[Fuente: {d.metadata.get('fuente', 'desconocida')}]\n"
        f"[Página: {d.metadata.get('pagina', 'desconocida')}]\n"
        f"[Categoría: {d.metadata.get('categoria', 'desconocida')}]\n"
        f"[Chunk: {d.metadata.get('chunk_id', 'desconocida')}]\n"
        f"{d.page_content}"
        for d in docs
    )

# Orquestación
async def get_rag_response(query: str) -> RAGSystem:
    try:
        logger.info(f"Procesando consulta: {query}")

        docs = await retriever_hibrido.ainvoke(query)

        contexto = formatear_documentos(docs)

        salida_llm: RespuestaLLM = await chain.ainvoke(
            {
                "contexto": contexto,
                "pregunta": query,
                "formato": parser_llm.get_format_instructions(),
            }
        )

        # sorted devuelve lista
        # por alguna razon que desconozco paginas y chunks aveces salian con float asi que force la transformacion en la respuesta
        fuentes = sorted({d.metadata.get("fuente", "desconocida") for d in docs})
        paginas = sorted({int(d.metadata.get("pagina", "desconocida")) for d in docs})
        categorias = sorted({d.metadata.get("categoria", "desconocida") for d in docs})
        chunks_ids = sorted({int(d.metadata.get("chunk_id", "desconocida")) for d in docs})
        

        return RAGResponse(
            query=query,
            respuesta=salida_llm.respuesta,
            fuentes=fuentes,
            paginas=paginas,
            categorias=categorias,
            chunks_ids=chunks_ids,
            fragmentos_recuperados=len(docs),
        )
    except Exception as e:
        logger.error(f"Error durante la ejecución: {e}")
        # Excepciones centralizadas para convertirlas en errores legibles para la aplicación.
        raise classify_error(e)


async def main():
    # estos prints se deberian quedar son la respuesta del llm
    respuesta_ok = await get_rag_response("¿Que es la correlacion?")
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