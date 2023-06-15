import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # Return the preprocessed text
    return ' '.join(filtered_tokens)

def get_similarity_score(user_answer, stored_answer):
    preprocessed_user_answer = preprocess_text(user_answer)
    preprocessed_stored_answer = preprocess_text(stored_answer)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([preprocessed_user_answer, preprocessed_stored_answer])
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

    return similarity_score

# Step 1: Read questions and answers from the text file
questions = []
answers = []

with open('data.txt', 'r') as file:
    lines = file.readlines()
    for i in range(0, len(lines), 2):
        questions.append(lines[i].strip())
        answers.append(lines[i+1].strip())

# Step 2: Randomly select 10 questions
selected_questions = random.sample(questions, 10)

# Step 3-7: Ask questions, evaluate answers, and calculate score
total_score = 0
scores = []

print("Let's begin!")
print()

for i, question in enumerate(selected_questions, 1):
    print("Question", i, ":", question)
    user_answer = input('Your Answer: ')

    # Step 4-5: Evaluate and compare answers
    answer_index = questions.index(question)
    similarity_score = get_similarity_score(user_answer, answers[answer_index])

    # Step 6: Calculate score
    score = int(similarity_score * 10)
    total_score += score
    scores.append(score)

    print()

# Step 8: Display the final score
final_score = total_score
formatted_scores = ', '.join(str(score) for score in scores)
print('Thank you!')
print('Final Score:', final_score, '(', formatted_scores, ')')

