import json

from db_ingest import vectorstore


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

# 600 tokens chunk
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

# 500 tokens chunk
# ==============================================================================
#     BM25   Vectorial    Precision@5    Recall@5
# --------------------------------------------------------------------------------
#     1.00        0.00         50.0%      62.5%
#     0.95        0.05         50.0%      62.5%
#     0.85        0.15         50.0%      62.5%
#     0.75        0.25         50.0%      62.5%
#     0.50        0.50         55.0%     100.0%
#     0.25        0.75         72.5%     100.0%
#     0.15        0.85         72.5%     100.0%
#     0.05        0.95         72.5%     100.0%
#     0.00        1.00         72.5%     100.0%

# 400 tokens chunk

# ================================================================================
# ANÁLISIS DE PESOS — ENSEMBLE RETRIEVER
# ================================================================================
#     BM25   Vectorial    Precision@5    Recall@5
# --------------------------------------------------------------------------------
#     1.00        0.00         47.5%      62.5%
#     0.95        0.05         47.5%      62.5%
#     0.85        0.15         47.5%      62.5%
#     0.75        0.25         47.5%      62.5%
#     0.50        0.50         60.0%     100.0%
#     0.25        0.75         80.0%     100.0%
#     0.15        0.85         80.0%     100.0%
#     0.05        0.95         80.0%     100.0%
#     0.00        1.00         80.0%     100.0%
# ================================================================================

# 300 tokens chunk

# ================================================================================
# ANÁLISIS DE PESOS — ENSEMBLE RETRIEVER
# ================================================================================
#     BM25   Vectorial    Precision@5    Recall@5
# --------------------------------------------------------------------------------
#     1.00        0.00         42.5%      62.5%
#     0.95        0.05         42.5%      62.5%
#     0.85        0.15         42.5%      62.5%
#     0.75        0.25         42.5%      62.5%
#     0.50        0.50         65.0%     100.0%
#     0.25        0.75         72.5%     100.0%
#     0.15        0.85         72.5%     100.0%
#     0.05        0.95         72.5%     100.0%
#     0.00        1.00         72.5%     100.0%




# En este corpus y sobre el Golden Set utilizado, la incorporación de BM25 no mejora 
# el rendimiento del sistema de recuperación. Las configuraciones con mayor peso de BM25 
# presentan un deterioro tanto en Precision@5 como en Recall@5. La recuperación vectorial 
# obtiene el mejor desempeño observado, mientras que las configuraciones híbridas con un peso 
# reducido para BM25 alcanzan el mismo desempeño que la recuperación vectorial pura.