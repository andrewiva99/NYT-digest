from chain import get_qa_chain
from dotenv import set_key, load_dotenv
import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()
st.markdown("[Get your Google API Key here](https://console.cloud.google.com/apis/credentials)")
new_key = st.text_input("Enter Google API KEY")
press_save = st.button("Save")

if new_key and press_save:
        try:
                ###Check key
                llm = GoogleGenerativeAI(model='gemini-1.5-flash', google_api_key=new_key)
                response = llm.invoke("Hello, world!")
                ###

                set_key(".env", "GOOGLE_API_KEY", new_key)
                load_dotenv()
                os.environ["GOOGLE_API_KEY"] = new_key
                st.success("API key updated successfully!")
        except Exception:
                st.error("The key is not valid. Enter new key.")


chain_qa = get_qa_chain()
chain_qa_answer = chain_qa.pick('answer')

st.title("NYT Digest")
question = st.text_input("Query")

if question:
    st.write_stream(chain_qa_answer.stream({'input': question}))