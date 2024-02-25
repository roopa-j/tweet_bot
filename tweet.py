import os
import streamlit as st
from langchain import PromptTemplate
from langchain_openai import OpenAI, ChatOpenAI
from langchain import LLMChain
from getpass import getpass

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

tweet_template = """
Give me {number} tweets on {topic}.
"""

tweet_prompt = PromptTemplate(template = tweet_template, input_variables = ['number', 'topic'])
gpt3_model = ChatOpenAI(model_name = 'gpt-3.5-turbo-0125')
tweet_chain = LLMChain(
    prompt = tweet_prompt,
    llm = gpt3_model
)
tweet_generator = LLMChain(prompt = tweet_prompt, llm = gpt3_model)

st.title("Tweet Generator - GenAI")
st.subheader("Generate tweets using AI")
topic = st.text_input("Enter the topic: ")
number = st.number_input("Number of tweets: ", min_value=1, max_value=10, value=1, step=1)
if st.button("Generate"):
    tweets = tweet_generator.run(number=number, topic= topic)
    #tweets = tweet_generator.invoke({"input": {"number": number, "topic": topic}})
    st.write("You entered the query")
    st.write(tweets)
