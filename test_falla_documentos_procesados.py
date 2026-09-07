from setup import documentos_procesados

query = "¿Qué es la probabilidad?"

print("\n================ DOCUMENTOS PROCESADOS ================\n")

for d in documentos_procesados:
    if "la probabilidad se puede estudiar" in d.page_content.lower():
        print("ENCONTRADO")
        print("Fuente:", d.metadata.get("fuente"))
        print("Categoría:", d.metadata.get("categoria"))
        print("Página:", d.metadata.get("pagina"))
        print("Chunk:", d.metadata.get("chunk_id"))
        print("\nContenido:\n")
        print(d.page_content[:2000])