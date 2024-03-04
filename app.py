import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import re
from nltk.corpus import stopwords
from flask import Flask, render_template, request, url_for
from datetime import datetime
from urllib.parse import unquote

lemmatizer = WordNetLemmatizer()

model = load_model(r'model\model.h5')
words = pickle.load(open(r'model\texts.pkl', 'rb'))
classes = pickle.load(open(r'model\labels.pkl','rb'))
df = pd.read_csv(r'dataset\FAQ.csv')

class prediction():
    def __init__(self, model, words, classes, qna):
        self.model = model
        self.words = words
        self.classes = classes
        self.qna = qna

    def standardize(self, sentence):
        sentence = list(sentence)
        std = list()
        for index, i in enumerate(sentence):
            sentence[index] = i.lower()
            sentence[index] = re.sub(r'https?://\S+', '', sentence[index])
            sentence[index] = re.sub(r"[^a-zA-Z]", " ", sentence[index])
            std.append(sentence[index])
        return std
    
    def tokenize(self, std):
        token = list()
        for index, i in enumerate(std):
            token.append(i.split())
        for num1, j in enumerate(token):
            for num, k in enumerate(j):
                token[num1][num] = lemmatizer.lemmatize(k)
        return token
    
    def remove_stop(self, token):
        filtered_sentence = []
        for index, i in enumerate(token):
            filtered_word = []
            for j in i:
                if j not in stopwords.words("english"):
                    filtered_word.append(j)    
            filtered_sentence.append(filtered_word)
        token= filtered_sentence 
        return token
    
    def bag_of_words(self, token):
        bags_words = [0]*len(self.words)
        bags = list()
        for i in token:
            for index, j in enumerate(i):
                for num, k in enumerate(self.words):
                    if j == k:
                        bags_words[num]=1                        
            bags.append(bags_words)
        return bags
    
    def model_prediction(self, bags):
        prediction = self.model.predict(np.array(bags))
        results = []
        for index, r in enumerate(prediction[0]):
            if r > 0.3:
                results.append([classes[index], r])
        if not results:
                print("No prediction is found")
                return []
        results_list = sorted(results, key=lambda x: x[1], reverse=True)
        print('prediction is found.', results_list)
        return results_list

    def getResponse(self, results):
        category = self.qna['Category']
        response = self.qna['Response']
        chatbot_res = list()
        if not results:
            chatbot_res = "I dont know your question."
        else:
            response_tag = results[0][0]
            for index, i in enumerate(category):
                if i == response_tag:
                    chatbot_res = response[index]
                    break        
        return chatbot_res
    
    def chatbot_response(self, sentence):
        chatbot = prediction(self.model, self.words, self.classes, self.qna)
        std = chatbot.standardize(sentence)
        token = chatbot.tokenize(std)
        token = chatbot.remove_stop(token)
        bags = chatbot.bag_of_words(token)
        predict = chatbot.model_prediction(bags)
        results = chatbot.getResponse(predict)
        
        return results
    
chatbot = prediction(model, words, classes, df)



app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    userText = str(userText)
    return chatbot.chatbot_response([userText])

thumb_reivew = pd.DataFrame(columns=['TimeStamp', 'UserInput', 'ChatbotResponse', 'ThumbRating'])
@app.route('/thumb_up', methods=['POST'])
def handle_thumb_up():
    global thumb_reivew
    message_id = []
    timestamp = request.form.get('time')
    user_input = unquote(request.form.get('input'))
    chatbot_response = unquote(request.form.get('response'))
    rating = request.form.get('rating')
    # Store '1' in the DataFrame for thumb-up
    new_data = pd.DataFrame({
        'UserInput': [user_input],
        'ChatbotResponse': [chatbot_response],
        'ThumbRating': [rating],
        'TimeStamp': [timestamp]
    })
    existing_df = pd.read_excel(r'dataset\feedback_data.xlsx')
    combine_df = pd.concat([existing_df, new_data])
    # Save the DataFrame to Excel
    combine_df.to_excel(r'dataset\feedback_data.xlsx', index=False)

    return "Thumb-up received for message_id: {}".format(message_id)





if __name__ == "__main__":
    app.run(debug=True)