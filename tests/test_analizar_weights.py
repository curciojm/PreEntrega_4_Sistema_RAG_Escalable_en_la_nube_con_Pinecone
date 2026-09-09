import json

from langchain_classic.retrievers import EnsembleRetriever
from PreEntrega_4_Sistema_RAG_Escalable_en_la_nube_con_Pinecone.retriever import retriever_bm25, retriever_vectorial


WEIGHTS = [
    (1.00, 0.00),
    (0.95, 0.05),
    (0.85, 0.15),
    (0.75, 0.25),
    (0.50, 0.50),
    (0.25, 0.75),
    (0.15, 0.85),
    (0.05, 0.95),
    (0.00, 1.00),
]


with open("golden_set.json", "r", encoding="utf-8") as f:
    golden_set = json.load(f)


def crear_retriever(bm25_weight, vector_weight):
    return EnsembleRetriever(
        retrievers=[
            retriever_bm25,
            retriever_vectorial,
        ],
        weights=[
            bm25_weight,
            vector_weight,
        ],
    )


def evaluar_retriever(retriever, golden_set):
    precision_resultados = []
    recall_resultados = []

    for caso in golden_set:

        if not caso["respuesta_disponible"]:
            continue

        resultados = retriever.invoke(caso["pregunta"])[:5]

        fuentes_esperadas = caso.get("fuentes_esperadas", [])

        fuentes_recuperadas = [
            doc.metadata.get("fuente")
            for doc in resultados
        ]

        # Precision@5
        coincidencias = sum(
            1
            for fuente in fuentes_recuperadas
            if fuente in fuentes_esperadas
        )

        precision = (
            coincidencias / len(resultados)
            if resultados
            else 0.0
        )

        # Recall@5 / hit-rate@5
        recall = (
            1.0
            if any(
                fuente in fuentes_esperadas
                for fuente in fuentes_recuperadas
            )
            else 0.0
        )

        precision_resultados.append(precision)
        recall_resultados.append(recall)

    precision_promedio = (
        sum(precision_resultados)
        / len(precision_resultados)
    )

    recall_promedio = (
        sum(recall_resultados)
        / len(recall_resultados)
    )

    return precision_promedio, recall_promedio


def analizar_weights():

    print("\n" + "=" * 80)
    print("ANÁLISIS DE PESOS — ENSEMBLE RETRIEVER")
    print("=" * 80)

    print(
        f"{'BM25':>8}"
        f"{'Vectorial':>12}"
        f"{'Precision@5':>15}"
        f"{'Recall@5':>12}"
    )

    print("-" * 80)

    resultados = []

    for bm25_weight, vector_weight in WEIGHTS:

        retriever = crear_retriever(
            bm25_weight,
            vector_weight,
        )

        precision, recall = evaluar_retriever(
            retriever,
            golden_set,
        )

        resultados.append(
            {
                "bm25": bm25_weight,
                "vectorial": vector_weight,
                "precision_at_5": precision,
                "recall_at_5": recall,
            }
        )

        print(
            f"{bm25_weight:>8.2f}"
            f"{vector_weight:>12.2f}"
            f"{precision:>14.1%}"
            f"{recall:>11.1%}"
        )

    print("=" * 80)

    mejor_precision = max(
        resultados,
        key=lambda x: x["precision_at_5"]
    )

    mejor_recall = max(
        resultados,
        key=lambda x: x["recall_at_5"]
    )

    print(
        "\nMejor Precision@5:"
        f" BM25={mejor_precision['bm25']:.2f},"
        f" Vectorial={mejor_precision['vectorial']:.2f}"
        f" → {mejor_precision['precision_at_5']:.1%}"
    )

    print(
        "Mejor Recall@5:"
        f" BM25={mejor_recall['bm25']:.2f},"
        f" Vectorial={mejor_recall['vectorial']:.2f}"
        f" → {mejor_recall['recall_at_5']:.1%}"
    )


if __name__ == "__main__":
    analizar_weights()
