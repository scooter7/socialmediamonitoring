import os
import streamlit as st
import openai
from dotenv import load_dotenv
import requests
import sqlite3  # Using pysqlite3-binary for SQLite compatibility

# Load environment and set up Streamlit Secrets for API keys
load_dotenv()

# Set environment variables directly from Streamlit secrets
os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Check SQLite version to ensure compatibility
st.write("SQLite version:", sqlite3.sqlite_version)

# Function to create LLM using GPT-4o-mini only
def create_llm():
    return "gpt-4o-mini"

# Function to perform topic research using LLM
def research_topic(brand_name):
    # Use OpenAI to create a prompt and return a response
    prompt = f"Conduct a thorough research on recent social media mentions of {brand_name}. Provide key points and insights."
    response = openai.chat.completions.create(
        model=create_llm(),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

# Function to analyze sentiment of social media mentions
def analyze_sentiment(text):
    sentiment_prompt = f"Analyze the sentiment of the following text and classify it as Positive, Negative, or Neutral: {text}"
    sentiment = openai.chat.completions.create(
        model=create_llm(),
        messages=[{"role": "user", "content": sentiment_prompt}],
        max_tokens=10
    )
    return sentiment.choices[0].message.content.strip()

# Streamlit UI Setup
st.title("Social Media Monitoring and Sentiment Analysis")
st.write("Analyze a brand or topic with integrated social media monitoring and sentiment analysis.")

# User input for brand or topic
brand_name = st.text_input("Enter the Brand or Topic Name")

# Run the analysis on button click
if st.button("Start Analysis"):
    if brand_name:
        st.write("Running social media monitoring and sentiment analysis...")

        # Run research task
        research_summary = research_topic(brand_name)
        st.write("Research Summary:", research_summary)
        
        # Run sentiment analysis task
        sentiment_analysis = analyze_sentiment(research_summary)
        st.write("Sentiment Analysis:", sentiment_analysis)
    else:
        st.error("Please enter a brand or topic name to proceed.")
