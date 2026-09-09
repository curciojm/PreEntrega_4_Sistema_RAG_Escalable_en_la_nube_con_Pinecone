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