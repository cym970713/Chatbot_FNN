# Chatbot with Feedforward Neural Network

## Before run py. file, please setup a virtual environment. 

1. **Create a Virtual Environment:**
`python -m venv env`
2. **Activate the Virtual Environment:**
`.\env\Scripts\activate`
3. **Install Requirements:**
`pip install -r requirements.txt `
4. **Run the Project:**
`python app.py`

## Project Description

This project aim to create a chatbot model that can be easily trained from FAQ dataset. 

## File and Folder Structure

1. **dataset:**
dataset/FAQ.csv  #data input for FNN model training

2. **model:**
Trained FNN model will be stored in "model" folder after runnning chatbot_model.py.
It will be utilized in app.py for predicting the user input. 

3. **static & template:**
Contains html and CSS file for the chatbot interface 

4. **chatbot_model.py:**
Script for performing text preprocessing and FNN model training.
Case Lowering, Symbol removal, hyperlink removal, tokenization, stopword removal

5. **app.py**
import Flask for REST API


