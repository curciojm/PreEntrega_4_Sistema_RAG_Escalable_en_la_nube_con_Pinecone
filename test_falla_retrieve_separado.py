from retriever import retriever_hibrido, retriever_bm25, retriever_vectorial

query = "¿Qué es la probabilidad?"

print("\n================ BM25 ================\n")

docs_bm25 = retriever_bm25.invoke(query)

for i, d in enumerate(docs_bm25):
    print(f"\n--- Resultado {i + 1} ---")
    print("Fuente:", d.metadata.get("fuente"))
    print("Categoría:", d.metadata.get("categoria"))
    print("Página:", d.metadata.get("pagina"))
    print("Chunk:", d.metadata.get("chunk_id"))
    print("Contenido:", d.page_content[:500])


print("\n================ PINECONE ================\n")

docs_vectorial = retriever_vectorial.invoke(query)

for i, d in enumerate(docs_vectorial):
    print(f"\n--- Resultado {i + 1} ---")
    print("Fuente:", d.metadata.get("fuente"))
    print("Categoría:", d.metadata.get("categoria"))
    print("Página:", d.metadata.get("pagina"))
    print("Chunk:", d.metadata.get("chunk_id"))
    print("Contenido:", d.page_content[:500])


print("\n================ HÍBRIDO ================\n")

docs_hibrido = retriever_hibrido.invoke(query)

for i, d in enumerate(docs_hibrido):
    print(f"\n--- Resultado {i + 1} ---")
    print("Fuente:", d.metadata.get("fuente"))
    print("Categoría:", d.metadata.get("categoria"))
    print("Página:", d.metadata.get("pagina"))
    print("Chunk:", d.metadata.get("chunk_id"))
    print("Contenido:", d.page_content[:500])