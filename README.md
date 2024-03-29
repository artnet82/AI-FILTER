# AI-FILTER
Инструментарий для проверки модели ИИ + фильтра


TF-IDF (Term Frequency-Inverse Document Frequency) - это метод векторизации текстовых данных, который позволяет представить тексты в виде числовых векторов, учитывая важность каждого термина (слова) внутри текста и внутри всей коллекции текстов.

В процессе TF-IDF векторизации текста выполняются два основных шага:

1. Term Frequency (Частота термина): TF измеряет, насколько часто термин (слово) появляется внутри текста. Проще говоря, это отношение количества раз, когда термин появляется в тексте, к общему числу слов в тексте. Более высокое значение TF указывает на большую важность термина внутри текста.

2. Inverse Document Frequency (Обратная частота документа): IDF измеряет, насколько уникален или информативен термин внутри коллекции текстов. Он вычисляется как логарифм отношения общего числа текстов в коллекции к числу текстов, в которых термин появляется. Более высокое значение IDF указывает на большую важность термина внутри коллекции текстов.

TF-IDF векторизация комбинирует эти два значения, умножая TF на IDF. Результатом является числовой вектор, где каждый элемент соответствует термину, а значение элемента отражает важность этого термина внутри текста и коллекции.

В задаче обнаружения запрещенных слов TF-IDF векторизация может быть использована для обучения классификатора на текстах с известными метками (например, тексты с запрещенными словами и без них). Классификатор может выявить закономерности и паттерны, связанные с наличием запрещенных слов в тексте.

При проверке сгенерированного текста на наличие запрещенных слов, TF-IDF векторизация позволяет представить сгенерированный текст в виде числового вектора. Затем этот вектор подается на вход обученному классификатору, который предсказывает метку (например, наличие или отсутствие запрещенных слов). Если классификатор предсказывает положительную метку (наличие запрещенных слов), мы можем просмотреть сгенерированный текст и определить, какие конкретные запрещенные слова были обнаружены.

Таким образом, TF-IDF векторизация позволяет учесть важность каждого термина внутри текста и коллекции и использовать эту информацию для обнаружения запрещенных слов в сгенерированном тексте.
