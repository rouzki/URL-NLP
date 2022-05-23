import streamlit as st
import requests
from io import StringIO


def main():

    st.title('URL Classification - Streamlit APP')
    message = st.text_input('Enter Text to Classify')

    if st.button('Predict'):
        payload = {
            "urls": [message]
        }
        res = requests.post(f"http://service:8000/predict/",json=payload )
        with st.spinner('Classifying, please wait....'):
            st.write(res.json())


    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:


        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

        # To read file as string:
        string_data = stringio.read()

        payload = {
            "text": string_data
        }
        res = requests.post(f"http://service:8000/predict/",json=payload )
        with st.spinner('Classifying, please wait....'):
            json_rs = res.json()
            st.write(json_rs)
            st.write('Text classified as ' + json_rs['prediction'])

if __name__ == '__main__':
    main()