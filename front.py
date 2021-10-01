import streamlit as st
import uuid
import SessionState
import requests
import json

## Global Var
AVAILABLE_CATEGORIES = ['Indonesian Names', 'Laptops']
AVAILABLE_MODE = ['Training', 'Inference']
BACKEND_URL = 'http://localhost:8000'

if 'text_area_value' not in st.session_state:
    st.session_state['text_area_value'] = ""

navbar = st.sidebar.title('Navigation Bar')
mode = st.sidebar.radio('Select Mode', AVAILABLE_MODE)

def post_training_data(train_data):
    # generated_id = uuid.uuid4()
    data = {
        "train_data" : train_data
    }
    r = requests.post(BACKEND_URL+'/similarity/train', json = data)
    content = json.loads(r._content)
    if r.status_code == 200:
        response = [f"Success Submitting Training Data with ID: {content['generated_id']}", "You can check the data in the inference page!"] 
    elif r.status_code == 422:
        response = ["Internal Server Error"]
    else:
        response = ["Unknown Error"]
    return response

def page_1():
    st.title('Input Data')
    input_text = st.text_area('Separate each sentences with enter')
    train_data = input_text.split('\n')
    if st.button('Submit'):
        response = post_training_data(train_data=train_data)
        for line in response:
            st.write(line)

def page_2():
    st.title('Similarity Search')
    st.selectbox('Select Available Categories', AVAILABLE_CATEGORIES)
    st.text_input('Enter some text', max_chars=50)
    if st.button('Submit'):
        st.write('Result')

if mode == AVAILABLE_MODE[0]:
    page_1()
elif mode == AVAILABLE_MODE[1]:
    page_2()


