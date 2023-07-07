import random
import nltk
import time
import speech_recognition as sr
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pyttsx3
from gtts import gTTS
import os

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(filtered_tokens)

def get_similarity_score(user_answer, stored_answer):
    preprocessed_user_answer = preprocess_text(user_answer)
    preprocessed_stored_answer = preprocess_text(stored_answer)

    # Create Word2Vec models
    user_model = Word2Vec([preprocessed_user_answer.split()], min_count=1)
    stored_model = Word2Vec([preprocessed_stored_answer.split()], min_count=1)

    # Convert user answer and stored answer to vectors
    user_vector = user_model.wv[preprocessed_user_answer.split()]
    stored_vector = stored_model.wv[preprocessed_stored_answer.split()]

    # Compute soft cosine similarity using TF-IDF weighted vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([preprocessed_user_answer, preprocessed_stored_answer])
    tfidf_user_vector = tfidf_matrix[0].toarray()
    tfidf_stored_vector = tfidf_matrix[1].toarray()
    similarity_score = cosine_similarity(tfidf_user_vector, tfidf_stored_vector)[0][0]

    return similarity_score

# Rest of the code remains the same

# Step 1: Read questions and answers from the text file
questions = []
answers = []

with open('datai.txt', 'r') as file:
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

# Initialize the speech recognizer
recognizer = sr.Recognizer()

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Set the sleep duration between questions (in seconds)
sleep_duration = 60

for i, question in enumerate(selected_questions, 1):
    print("Question", i, ":", question)

    # Convert the question text to speech and save as an audio file
    question_tts = gTTS(text=question, lang='en')
    question_tts.save('question.mp3')

    # Play the question audio
    os.system('start question.mp3')

    # Wait for user's verbal answer
    print("Listening for your answer (10 seconds timeout)...")
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=10)

    try:
        # Convert speech to text
        user_answer = recognizer.recognize_google(audio)
        print("Your Answer:", user_answer)

        # Step 4-5: Evaluate and compare answers
        answer_index = questions.index(question)
        similarity_score = get_similarity_score(user_answer, answers[answer_index])

        # Step 6: Calculate score
        score = int(similarity_score * 10)
        total_score += score
        scores.append(score)

        print()

        # Wait before asking the next question
        time.sleep(sleep_duration)

    except sr.UnknownValueError:
        print("Couldn't understand your answer. Moving to the next question.")
        print()
    except sr.RequestError:
        print("Sorry, I'm facing some technical issues. Moving to the next question.")
        print()

# Step 8: Display the final score
final_score = total_score
formatted_scores = ', '.join(str(score) for score in scores)
print('Thank you!')
print('Final Score:', final_score, '(', formatted_scores, ')')