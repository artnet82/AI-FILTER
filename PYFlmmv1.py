import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train_classifier(texts, labels):
    # Инициализация токенизатора и векторизатора
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    vectorizer = TfidfVectorizer(tokenizer=tokenizer.tokenize)

    # Векторизация текстов
    X = vectorizer.fit_transform(texts)

    # Обучение классификатора
    classifier = LogisticRegression()
    classifier.fit(X, labels)

    return classifier, vectorizer

def check_forbidden_words(model_name, text, forbidden_words, classifier, vectorizer):
    # Загрузка токенизатора и модели
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Токенизация текста
    input_ids = tokenizer.encode(text, return_tensors='pt')

    # Генерация текста с помощью модели
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Векторизация сгенерированного текста
    X_generated = vectorizer.transform([generated_text])

    # Классификация сгенерированного текста
    predicted_label = classifier.predict(X_generated)[0]

    # Проверка наличия запрещенных слов, если классификатор предсказал положительный результат
    if predicted_label == 1:
        for word in forbidden_words:
            if word in generated_text:
                print(f"Обнаружено запрещенное слово: {word}")

def main():
    model_name = "gpt2"  # Замените на имя или путь к вашей модели
    text = "Пример начального текста для генерации."
    forbidden_words = ["запрещенное", "слово", "недопустимое"]  # Список запрещенных слов

    # Обучение классификатора
    texts = ["Пример текста без запрещенных слов.", "Пример текста с запрещенными словами."]
    labels = [0, 1]
    classifier, vectorizer = train_classifier(texts, labels)

    # Проверка сгенерированного текста на запрещенные слова
    check_forbidden_words(model_name, text, forbidden_words, classifier, vectorizer)

if __name__ == '__main__':
    main()
