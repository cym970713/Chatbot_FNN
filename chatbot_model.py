import numpy as np 
import pandas as pd 
import re
import os
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

lemmatizer = WordNetLemmatizer()
df = pd.read_csv('dataset/FAQ.csv')

class preprocess():
    def __init__(self, texts, category):
        self.texts = list(texts)
        self.category = list(category)

    def standardize(self):
        std = list()
        for index, i in enumerate(self.texts):
            self.texts[index] = i.lower()
            self.texts[index] = re.sub(r'https?://\S+', '', self.texts[index])
            self.texts[index] = re.sub(r"[^a-zA-Z]", " ", self.texts[index])
            std.append(self.texts[index])
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
        words = set()
        bags = list()
        outputs = list()
        for i in token:
            for j in i:
                words.add(j)
        words = list(sorted(words))
        
        bags_of_words = [0]*len(words)
        for i in token:
            for index, j in enumerate(i):
                for num, k in enumerate(words):
                    if j == k:
                        bags_of_words[num] = 1
            bags.append(bags_of_words)
            bags_of_words = [0]*len(words)
        
        classes = list(sorted(set(self.category)))
        outputs_0 = [0]*len(classes)
        for l in self.category:
            for num1, m in enumerate(classes):
                if l == m:
                    outputs_0[num1] = 1
            outputs.append(outputs_0)
            outputs_0 = [0]*len(classes)
                
        return words, classes, bags, outputs
    
    def generate_pickles(self, words, classes):
        file_path = 'model/'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        pickle.dump(words, open(r'model/texts.pkl', 'wb'))
        pickle.dump(classes, open(r'model/labels.pkl', 'wb'))
        print("generating file complete")

    def all_compute(self):
        Z = preprocess(self.texts, self.category)
        std = Z.standardize()
        token = Z.tokenize(std)
        token = Z.remove_stop(token)
        words, classes, bags, outputs = Z.bag_of_words(token)
        Z.generate_pickles(words, classes)
        df['preprocessed']=std
        df['token']=token
        df['bags_of_words']=bags
        df['output']=outputs
        
        return shuffle(df).reset_index()

class model_evaluate:
    def __init__(self, train_x, train_y):
        self.train_x = np.array(list(train_x))
        self.train_y = np.array(list(train_y))
    
    
    def training_split(self):
        x_train, x_val, y_train, y_val = train_test_split(self.train_x, self.train_y, test_size=0.2, random_state=50)
        return x_train, x_val, y_train, y_val
    
    def model_build(self, x_train, x_val, y_train, y_val):
        model = Sequential()
        model.add(Dense(128, input_shape=(len(x_train[0]),), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(y_train[0]), activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
        hist = model.fit(np.array(x_train), np.array(y_train), epochs=100, batch_size=5, verbose=1, validation_data=(x_val, y_val))
        return hist, model
            
    def generate_model(self, hist, model):
        model.save(r'model/model.h5', hist)
        print('generate file complete')        
    
    def compute_all(self):
        training=model_evaluate(self.train_x, self.train_y)
        x_train, x_val, y_train, y_val = training.training_split()
        hist, model = training.model_build(x_train, x_val, y_train, y_val)
        training.generate_model(hist, model)
        return "Model trained and saved."


data_process = preprocess(df['Question'], df['Category'])
newdf = data_process.all_compute()
X_train = newdf ['bags_of_words']
Y_train = newdf ['output']
model_training = model_evaluate(X_train, Y_train)
model_training.compute_all()