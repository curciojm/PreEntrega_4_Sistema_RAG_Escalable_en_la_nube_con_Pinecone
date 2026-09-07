from retriever import retriever_bm25

docs = retriever_bm25.invoke("probabilidad")

for i, d in enumerate(docs):
    print(
        i + 1,
        "|",
        d.metadata.get("fuente"),
        "|",
        d.metadata.get("categoria"),
        "| página:",
        d.metadata.get("pagina"),
        "| chunk:",
        d.metadata.get("chunk_id")
    )

# ¿Qué es la probabilidad?

# no representa simplemente "probabilidad". 
# Representa una frase completa y su espacio semántico puede acercarla a textos sobre qué es una muestra probabilística, 
# especialmente porque tu corpus tiene muchísimo contenido relacionado con "probabilidad/probabilística".