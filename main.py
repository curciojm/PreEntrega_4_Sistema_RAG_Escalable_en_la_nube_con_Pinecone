import asyncio

from rag import get_rag_response


async def main():
    respuesta_ok = await get_rag_response("¿Que es la correlacion?")
    print(f"\nPREGUNTA ORIGINAL: {respuesta_ok.query}")
    print(f"\nRESPUESTA: {respuesta_ok.respuesta}")
    print(f"\nFUENTES: {respuesta_ok.fuentes}")
    print(f"\nPAGINAS: {respuesta_ok.paginas}")
    print(f"\nCATEGORIAS: {respuesta_ok.categorias}")
    print(f"\nCHUNKS: {respuesta_ok.chunks_ids}")
    print(f"\nFragmentos usados: {respuesta_ok.fragmentos_recuperados}")

if __name__ == "__main__":
    asyncio.run(main())