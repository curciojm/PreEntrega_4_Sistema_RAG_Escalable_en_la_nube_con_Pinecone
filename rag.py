from chain import chain, parser_llm
from errors import classify_error
from logging_config import logger
from retriever import retriever_hibrido
from schemas import RAGResponse, RespuestaLLM


def formatear_documentos(docs) -> str:
    """Formatea los documentos recuperados para incorporarlos al contexto del LLM."""
    return "\n\n---\n\n".join(
        f"[Fuente: {d.metadata.get('fuente', 'desconocida')}]\n"
        f"[Página: {d.metadata.get('pagina', 'desconocida')}]\n"
        f"[Categoría: {d.metadata.get('categoria', 'desconocida')}]\n"
        f"[Chunk: {d.metadata.get('chunk_id', 'desconocida')}]\n"
        f"{d.page_content}"
        for d in docs
    )



async def get_rag_response(query: str) -> RAGResponse:
    try:
        logger.info(f"Procesando consulta: {query}")

        docs = await retriever_hibrido.ainvoke(query)

        logger.info(f"Fragmentos recuperados: {len(docs)}")

        contexto = formatear_documentos(docs)

        salida_llm: RespuestaLLM = await chain.ainvoke(
            {
                "contexto": contexto,
                "pregunta": query,
                "formato": parser_llm.get_format_instructions(),
            }
        )

        fuentes = sorted({d.metadata.get("fuente", "desconocida") for d in docs})
        paginas = sorted({int(d.metadata.get("pagina", "desconocida")) for d in docs})
        categorias = sorted({d.metadata.get("categoria", "desconocida") for d in docs})
        chunks_ids = sorted({int(d.metadata.get("chunk_id", "desconocida")) for d in docs})
        
        
        logger.info(f"Consulta procesada correctamente: {query}")
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
        # Se centraliza la clasificación de excepciones para devolver errores legibles a la aplicación.
        raise classify_error(e)