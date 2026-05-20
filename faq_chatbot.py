import nltk
nltk.download('punkt')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

questions = [
    "What is AI?",
    "What is Python?",
    "What is Machine Learning?",
    "What is NLP?"
]

answers = [
    "AI means Artificial Intelligence.",
    "Python is a programming language.",
    "Machine Learning is a subset of AI.",
    "NLP means Natural Language Processing."
]

vectorizer = TfidfVectorizer()

question_vectors = vectorizer.fit_transform(questions)

print("FAQ Chatbot (type 'exit' to stop)")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    user_vector = vectorizer.transform([user_input])

    similarity = cosine_similarity(user_vector, question_vectors)

    index = similarity.argmax()

    print("Bot:", answers[index])