
import json
import re

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage


# Подключение локальной модели
model = ChatOllama(
    model="qwen2.5:3b",
    base_url="http://localhost:11434",
    temperature=0,
    num_predict=350
)


# Структура проверки
class QualityReview(BaseModel):
    score: float = Field(ge=0, le=10)
    approved: bool
    problems: list[str] = Field(default_factory=list)
    feedback: str


# Агент контроля качества
def quality_controller(brief, design, recommendation):

    print("\nQuality Controller проверяет рекомендацию...")

    structured_model = model.with_structured_output(
        QualityReview,
        method="json_mode"
    )

    prompt = """
    Ты — независимый агент Quality Controller.

    Проверяй рекомендации свадебных дизайнов.

    Критерии:
    1. Дизайн должен существовать в каталоге.
    2. Стиль должен соответствовать пожеланиям.
    3. Нельзя придумывать характеристики дизайна.
    4. Нельзя заявлять, что цвет поддерживается,
       если он отсутствует в описании каталога.
    5. Несовпадение цветов допустимо,
       если об этом честно предупреждают клиента.
    6. Обоснование должно быть понятным.

    Если есть фактическая ошибка, отклони результат.

    Если оценка ниже 8.5, approved должен быть false.

    Верни только JSON:
    {
      "score": 7.0,
      "approved": false,
      "problems": ["описание проблемы"],
      "feedback": "как исправить рекомендацию"
    }
    """

    data = {
        "client": brief,
        "catalog_design": design,
        "recommendation": recommendation
    }

    response = structured_model.invoke([
        SystemMessage(content=prompt),
        HumanMessage(
            content=json.dumps(
                data,
                ensure_ascii=False
            )
        )
    ])

    review = QualityReview.model_validate(response)

    # Дополнительная программная проверка цветов
    description = design["description"]

    match = re.search(
        r"Цвета:\s*([^.!?\n]+)",
        description,
        flags=re.IGNORECASE
    )

    catalog_colors = []

    if match:
        catalog_colors = [
            color.strip().lower()
            for color in match.group(1).split(",")
        ]

    requested_colors = [
        color.lower()
        for color in brief.get("colors", [])
    ]

    missing_colors = [
        color for color in requested_colors
        if color not in catalog_colors
    ]

    # Проверяем, предупредил ли агент о расхождении
    reason = recommendation["reason"].lower()

    warning_words = [
        "отлича",
        "не совпада",
        "отсутств",
        "нет в палитре",
        "не предусмотр",
        "адаптац",
        "потребуется",
        "понадобится"
    ]

    has_warning = any(
        word in reason
        for word in warning_words
    )

    if missing_colors and not has_warning:

        problem = (
            "Не указано расхождение с палитрой. "
            "В каталоге отсутствуют цвета: "
            + ", ".join(missing_colors)
        )

        review.problems.append(problem)
        review.approved = False
        review.score = min(review.score, 7.0)
        review.feedback = (
            "Исправь рекомендацию. "
            "Не утверждай, что дизайн поддерживает "
            "отсутствующие цвета. "
            + review.feedback
        )

    if review.score < 8.5:
        review.approved = False

    return review.model_dump()


# Тестируем на реальной ошибке
if __name__ == "__main__":

    brief = {
        "client_name": "Анна",
        "style": "классический",
        "colors": ["молочный", "бордовый"],
        "wishes": "элегантное электронное приглашение"
    }

    design = {
        "design_name": "Old Money",
        "description": (
            "Old Money — классический элегантный "
            "минималистичный дизайн. "
            "Цвета: молочный, шоколадный, темно-зеленый. "
            "Подходит для исторических усадеб, "
            "дворцов и загородных площадок."
        )
    }

    recommendation = {
        "best_design": "Old Money",
        "reason": (
            "Old Money соответствует классическому "
            "стилю и цветам молочный, бордовый."
        )
    }

    print("=" * 50)
    print("BRIDALFLOW AI — ПРОВЕРКА КАЧЕСТВА")
    print("=" * 50)

    result = quality_controller(
        brief,
        design,
        recommendation
    )

    print("\nРЕЗУЛЬТАТ ПРОВЕРКИ:")

    print(json.dumps(
        result,
        ensure_ascii=False,
        indent=2
    ))

    if result["approved"]:
        print("\nРекомендация принята.")
    else:
        print("\nРекомендация отправляется на доработку.")
