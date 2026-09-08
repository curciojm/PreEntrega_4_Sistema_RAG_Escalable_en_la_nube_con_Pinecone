import json

from vector_db import vectorstore


with open("golden_set.json", "r", encoding="utf-8") as f:
    golden_set = json.load(f)


def analizar_scores(golden_set):
    for caso in golden_set:

        query = caso["pregunta"]

        resultados = vectorstore.similarity_search_with_score(
            query,
            k=5
        )

        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print(f"Respuesta disponible: {caso['respuesta_disponible']}")

        for i, (doc, score) in enumerate(resultados, 1):
            print(
                f"{i}. score={score:.4f} | "
                f"fuente={doc.metadata.get('fuente')} | "
                f"categoria={doc.metadata.get('categoria')}"
            )


analizar_scores(golden_set)

# ESTE TEST DEMOSTRO QUE BM25 ESTABA INFLUENCIANDO CHUNKS NO TAN RELEVANTES YA QUE LA BUSQUEDA VECTORIAL FUNCIONABA
# MUCHO MEJOR, POR ELLO SE DISMINUYO BM25

# bm25 vectorial
# 1 0 62.5 - 42.5
# 0.95 0.5 62.5 - 42.5
# 0.85 0.15 62.5 - 42.5
# 0.75 0.25 62.5 - 42.5
# 0.5 0.5 75 - 50 
# 0.25 0.75 87.5 - 65
# 0.15 0.85 87.5 - 65
# 0.05 0.95 87.5 - 65
# 0 1 87.5 - 65

# En este corpus y sobre el Golden Set utilizado, la incorporación de BM25 no mejora 
# el rendimiento del sistema de recuperación. Las configuraciones con mayor peso de BM25 
# presentan un deterioro tanto en Precision@5 como en Recall@5. La recuperación vectorial 
# obtiene el mejor desempeño observado, mientras que las configuraciones híbridas con un peso 
# reducido para BM25 alcanzan el mismo desempeño que la recuperación vectorial pura.