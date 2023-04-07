from flask import Flask, render_template, request
import random
import pandas as pd
import numpy as np
import nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')
# initialize the stemmer
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

app = Flask(__name__)

class SikkaBot:
    def __init__(self):
        # Load data from CSV file
        self.data = pd.read_csv(r'full_data.csv', encoding='unicode_escape')

        # create a TfidfVectorizer object
        self.vectorizer = TfidfVectorizer()

        # fit the vectorizer to your data
        self.vectorizer.fit(self.data['question'])

        # transform the questions into vectors of TF-IDF values
        self.vectors = self.vectorizer.transform(self.data['question'])

    def preprocessUserTextAndPredict(self, message):
        tokens = word_tokenize(message.lower())
        tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
        message = ' '.join(tokens)

        # transform the user input into a vector of TF-IDF values
        user_vector = self.vectorizer.transform([message])

        # calculate the cosine similarity between the user input and each question in the dataframe
        similarity_scores = cosine_similarity(user_vector, self.vectors)
        # print(f'type(similarity_scores): {similarity_scores}')
        # max_similarity_scores = (list(similarity_scores))
        # print(f'max_similarity_scores: {type(similarity_scores)}')
        max_value = np.max(similarity_scores)
        # print(f'max_value: {max_value}')

        if max_value > 0.5:
        # find the index of the question with the highest similarity score
            most_similar_index = similarity_scores.argmax()
            # print(f'most_similar_index: {most_similar_index}')

            # retrieve the corresponding answer
            answer = self.data['answer'][most_similar_index]
        else:
            answer= "I don't have an answer for this. Would you mind rephrasing the question."

        return answer

@app.route('/')
def chatbot():
    # user_message = request.form['user_message']
    # bot_message = generate_bot_response(user_message)
    # return jsonify({'response': bot_message})
    return render_template("index.html")

@app.route("/chatbot", methods=['POST'])
def generate_bot_response():
    message = request.form['msg']
    # print(f'In Function generate_bot_response')
    # return "I am sorry, I did not understand your message."
    # return SikkaBot.preprocessUserTextAndPredict(message = request.form['msg'])
    tokens = word_tokenize(message.lower())
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    message = ' '.join(tokens)

    # transform the user input into a vector of TF-IDF values
    user_vector = SB.vectorizer.transform([message])

    # calculate the cosine similarity between the user input and each question in the dataframe
    similarity_scores = cosine_similarity(user_vector, SB.vectors)
    # print(f'type(similarity_scores): {similarity_scores}')
    # max_similarity_scores = (list(similarity_scores))
    # print(f'max_similarity_scores: {type(similarity_scores)}')
    max_value = np.max(similarity_scores)
    # print(f'max_value: {max_value}')

    if max_value > 0.5:
    # find the index of the question with the highest similarity score
        most_similar_index = similarity_scores.argmax()
        # print(f'most_similar_index: {most_similar_index}')

        # retrieve the corresponding answer
        answer = SB.data['answer'][most_similar_index]
    else:
        answer= "I don't have an answer for this. Would you mind rephrasing the question."

    return answer


if __name__ == '__main__':
    SB = SikkaBot()
    app.run(debug=True)
