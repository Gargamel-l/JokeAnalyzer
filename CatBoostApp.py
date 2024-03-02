# Импортируем необходимые библиотеки
import numpy as np
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# Реальные данные для обучения модели
jokes = [
    "Почему мухи не знают таблеток? Потому что их не было на курсах повышения квалификации!",
    "Почему пираты - плохие бармены? Потому что они всегда сдаются на последнем рубеже!",
    "Как называется рыба, которая всё время лжёт? Угрюмая.",
    "Камень серого цвета",
    "Земля несъедобна"
]

labels = [1, 1, 0, 0, 0]  # 1 - смешная, 0 - несмешная

# Преобразование текстов в числовые признаки
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(jokes)

# Обучение модели CatBoost
model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=2, loss_function='Logloss')
model.fit(X, labels)

# Функция для предсказания смешности шутки
def predict_joke(joke):
    joke_features = vectorizer.transform([joke])  # Преобразование текста шутки в числовые признаки
    prediction = model.predict(joke_features)[0]
    return "Смешная шутка" if prediction == 1 else "Несмешная шутка"

# Пара шуток для проверки
joke1 = "Почему стул упал? Потому что у него ноги!"
joke2 = "Трава зелёная"
joke3 = "Почему зебра полосатая? Потому что ее не красят!"
joke4 = "Знаете почему разбитая тарелка к счастью? потому что её мыть не надо "

# Проверка шуток и вывод результата
print("Шутка 1:", predict_joke(joke1))
print("Шутка 2:", predict_joke(joke2))
print("Шутка 3:", predict_joke(joke3))
print("Шутка 4:", predict_joke(joke4))
