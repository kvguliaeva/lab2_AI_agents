
from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings


# ==========================================
# 1. ПОДКЛЮЧЕНИЕ К QDRANT
# ==========================================

qdrant = QdrantClient(
    url="http://localhost:6333",
    timeout=15
)

collection_name = "bridalflow_designs"


# ==========================================
# 2. МОДЕЛЬ ДЛЯ ВЕКТОРНОГО ПОИСКА
# ==========================================

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)


# ==========================================
# 3. ПОИСК ПОДХОДЯЩИХ ДИЗАЙНОВ
# ==========================================

def search_designs(query: str, limit: int = 3):

    print("\nПоиск дизайнов в Qdrant...")

    # Превращаем пожелания в вектор
    query_vector = embeddings.embed_query(query)

    print("Размер вектора:", len(query_vector))

    if len(query_vector) != 768:
        raise ValueError(
            "Размер вектора не совпадает с коллекцией Qdrant"
        )

    # Ищем похожие дизайны
    response = qdrant.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        with_payload=True
    )

    results = []

    for point in response.points:

        payload = point.payload or {}
        metadata = payload.get("metadata") or {}

        results.append({
            "design_name": metadata.get(
                "designName", "Без названия"
            ),
            "description": payload.get("content", ""),
            "score": round(point.score, 4)
        })

    return results


# ==========================================
# 4. ТЕСТИРОВАНИЕ
# ==========================================

if __name__ == "__main__":

    client_query = """
    Свадьба в классическом элегантном стиле.
    Предпочитаемые цвета — молочный и бордовый.
    Хотим утонченный, романтичный дизайн
    свадебного сайта без лишнего декора.
    """

    print("=" * 50)
    print("BRIDALFLOW AI — RAG ПОИСК ДИЗАЙНОВ")
    print("=" * 50)

    designs = search_designs(client_query)

    print("\nНАЙДЕННЫЕ ДИЗАЙНЫ:")

    for index, design in enumerate(designs, start=1):

        print(f"\n{index}. {design['design_name']}")
        print(f"Сходство: {design['score']}")
        print(f"Описание: {design['description']}")

    print("\nПоиск завершён!")
