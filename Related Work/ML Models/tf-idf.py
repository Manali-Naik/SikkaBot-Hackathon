import pandas as pd

# read the CSV file into a pandas dataframe
data = pd.read_csv(r"C:\Users\shashiraj.walsetwar\Desktop\Solo-Hackathon\Solo Hackathon\Database\Raw_data_question_answers_new.csv", encoding= 'unicode_escape')

print(data.columns)

import nltk
nltk.download('punkt')

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# download the stopwords corpus if not already downloaded
nltk.download('stopwords')

# initialize the stemmer
stemmer = PorterStemmer()

# remove stop words and perform stemming
stop_words = set(stopwords.words('english'))

for i in range(len(data)):
    # tokenize the question
    question = data['question'][i]
    tokens = word_tokenize(question.lower())
    
    # remove stop words and perform stemming
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    
    # update the question in the dataframe
    data.at[i, 'question'] = ' '.join(tokens)

print(data['question'])

from sklearn.feature_extraction.text import TfidfVectorizer

# create a TfidfVectorizer object
vectorizer = TfidfVectorizer()

# fit the vectorizer to your data
vectorizer.fit(data['question'])

# transform the questions into vectors of TF-IDF values
vectors = vectorizer.transform(data['question'])

from sklearn.metrics.pairwise import cosine_similarity

# get the user input
user_input = input('You: ')

while user_input.lower() != 'exit':

    # preprocess the user input
    tokens = word_tokenize(user_input.lower())
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    user_input = ' '.join(tokens)

    # transform the user input into a vector of TF-IDF values
    user_vector = vectorizer.transform([user_input])

    # calculate the cosine similarity between the user input and each question in the dataframe
    similarity_scores = cosine_similarity(user_vector, vectors)

    # find the index of the question with the highest similarity score
    most_similar_index = similarity_scores.argmax()
    print(f'most_similar_index: {most_similar_index}')

    # retrieve the corresponding answer
    answer = data['answer'][most_similar_index]

    print('Bot: ' + answer)
    user_input = input('You: ')

