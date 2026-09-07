from retriever import retriever_vectorial

consultas = [
    "probabilidad",
    "qué es probabilidad",
    "definición de probabilidad",
    "qué significa probabilidad",
    "concepto de probabilidad",
]

for query in consultas:
    print("\n" + "=" * 60)
    print(query)
    print("=" * 60)

    docs = retriever_vectorial.invoke(query)

    for i, d in enumerate(docs):
        print(
            i + 1,
            "|",
            d.metadata.get("categoria"),
            "|",
            d.metadata.get("pagina"),
            "|",
            d.metadata.get("chunk_id")
        )

# El embedding de MiniLM está interpretando "qué es probabilidad" de una manera bastante problemática: 
# parece asociar la construcción de la pregunta con el concepto de muestra probabilística, aunque el chunk correcto 
# está perfectamente indexado

# al juntar la conexion con el indice se llena top k con muestra probabilistica y no llega a encontrar probabilidad