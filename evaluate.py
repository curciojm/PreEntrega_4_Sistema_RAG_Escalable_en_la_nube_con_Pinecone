import json
from typing import List, Dict

from schemas import RAGSystem
from retriever import retriever_hibrido

with open("golden_set.json", "r", encoding="utf-8") as f:
    golden_set = json.load(f)

def evaluar(rag_system: RAGSystem, golden_set: List[Dict]) -> Dict:

    resultados_por_pregunta = []

    for caso in golden_set:

        top_k = rag_system.obtener_top_k(caso["pregunta"])

        fuentes_recuperadas = [r["fuente"] for r in top_k]
        categorias_recuperadas = [r["categoria"] for r in top_k]

        fuentes_esperadas = caso.get("fuentes_esperadas", [])
        categorias_esperadas = caso.get("categorias_esperadas", [])

        respuesta_disponible = caso.get("respuesta_disponible", False)

        if respuesta_disponible:

            # RECALL
            # Recall de fuentes:
            # ¿Aparece al menos una de las fuentes esperadas?
            recall_fuentes = 1.0 if any(
                fuente in fuentes_esperadas
                for fuente in fuentes_recuperadas
            ) else 0.0

            # Recall de categorías:
            # ¿Aparece al menos una de las categorías esperadas?
            recall_categorias = 1.0 if any(
                categoria in categorias_esperadas
                for categoria in categorias_recuperadas
            ) else 0.0

            # Precision@5:
            # Un fragmento es relevante si su fuente O su categoría
            # coincide con alguna de las esperadas.
            coincidencias_fuentes = sum(
                1
                for fuente in fuentes_recuperadas
                    if fuente in fuentes_esperadas
            )

            precision_fuentes = (
                coincidencias_fuentes / len(top_k)
                if top_k
                else 0.0
            )
            coincidencias_categorias = sum(
                1
                for categoria  in categorias_recuperadas
                    if categoria  in categorias_esperadas
            )
            
            precision_categorias = (
                coincidencias_categorias / len(top_k)
                if top_k
                else 0.0
            )

        else:

            # Cuando la respuesta no está disponible en el corpus,
            # el retriever puede recuperar fragmentos igualmente.
            # Por lo tanto, no evaluamos esos fragmentos como
            # relevantes/irrelevantes.
            #
            # El caso se considera correcto porque no esperamos
            # fuentes ni categorías específicas.
            recall_fuentes = None
            recall_categorias = None
            precision_fuentes = None
            precision_categorias = None

        resultados_por_pregunta.append({
            "pregunta": caso["pregunta"],
            "fuentes_esperadas": fuentes_esperadas,
            "fuentes_recuperadas": fuentes_recuperadas,
            "categorias_esperadas": categorias_esperadas,
            "categorias_recuperadas": categorias_recuperadas,
            "recall_fuentes": recall_fuentes,
            "recall_categorias": recall_categorias,
            "precision_fuentes": precision_fuentes,
            "precision_categorias": precision_categorias,
        })

    recall_fuentes_promedio = (
        sum(
            r["recall_fuentes"]
            for r in resultados_por_pregunta
            if r["recall_fuentes"] is not None
        )
        / sum(
            1
            for r in resultados_por_pregunta
            if r["recall_fuentes"] is not None
    )
    )

    recall_categorias_promedio = (
        sum(
            r["recall_categorias"]
            for r in resultados_por_pregunta
            if r["recall_categorias"] is not None
        )
        / sum(
            1
            for r in resultados_por_pregunta
            if r["recall_categorias"] is not None
        )
    )

    precision_fuentes_promedio = (
        sum(
            r["precision_fuentes"]
            for r in resultados_por_pregunta
            if r["precision_fuentes"] is not None
        )
        / sum(
            1
            for r in resultados_por_pregunta
            if r["precision_fuentes"] is not None
        )
    )
    
    precision_categorias_promedio = (
            sum(
                r["precision_categorias"]
                for r in resultados_por_pregunta
                if r["precision_categorias"] is not None
            )
            / sum(
                1
                for r in resultados_por_pregunta
                if r["precision_categorias"] is not None
        )
    )
    

    return {
        "detalle": resultados_por_pregunta,
        "recall_fuentes_promedio": recall_fuentes_promedio,
        "recall_categorias_promedio": recall_categorias_promedio,
        "precision_fuentes_promedio": precision_fuentes_promedio,
        "precision_categorias_promedio": precision_categorias_promedio,
    }


rag_system = RAGSystem(retriever_hibrido)

reporte = evaluar(rag_system, golden_set)

def mostrar_metrica(valor):
    return "N/A" if valor is None else f"{valor:.0%}"

for r in reporte["detalle"]:

    if (
        r["recall_fuentes"] is None and r["recall_categorias"] is None
    ):
        estado = "➖"
    elif (
            r["recall_fuentes"] == 1.0
            and r["recall_categorias"] == 1.0
        ):
        estado = "✅"
    else:
        estado = "❌"

    print(f"{estado} {r['pregunta']}")

    print(f"   Fuentes esperadas: {r['fuentes_esperadas']}")
    print(f"   Fuentes recuperadas: {r['fuentes_recuperadas']}")
    print(f"   Categorías esperadas: {r['categorias_esperadas']}")
    print(f"   Categorías recuperadas: {r['categorias_recuperadas']}")
    print(f"   Recall fuentes: {mostrar_metrica(r['recall_fuentes'])}")
    print(f"   Precision fuentes: {mostrar_metrica(r['precision_fuentes'])}")
    print(f"   Recall categorías: {mostrar_metrica(r['recall_categorias'])}")
    print(f"   Precision categorías: {mostrar_metrica(r['precision_categorias'])}")

print("=" * 80)

print(
    f"📊 RECALL FUENTES: "
    f"{reporte['recall_fuentes_promedio']:.1%}"
)

print(
    f"📊 PRECISION FUENTES@5: "
    f"{reporte['precision_fuentes_promedio']:.1%}"
)

print(
    f"📊 RECALL CATEGORÍAS: "
    f"{reporte['recall_categorias_promedio']:.1%}"
)

print(
    f"📊 PRECISION CATEGORÍAS@5: "
    f"{reporte['precision_categorias_promedio']:.1%}"
)