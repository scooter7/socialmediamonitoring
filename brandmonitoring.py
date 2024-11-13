import os
import time
import streamlit as st
import openai
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
import csv
import requests

# Load environment and set up Streamlit Secrets for API keys
load_dotenv()

# Directly set environment variables and API keys from Streamlit secrets
os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Function to select LLM based on user choice
def create_llm(use_gpt=True):
    return ChatOpenAI(model="gpt-4o-mini") if use_gpt else Ollama(model="llama3.1")

# Function to perform topic research using LLM
def research_topic(topic, llm):
    prompt = f"Conduct a thorough research on the following topic: {topic}. Provide a summary with key points and insights."
    response = llm.generate(prompt)
    return response

# Function to monitor social media for mentions of a brand/topic
def monitor_social_media(topic):
    # Using SerperDevTool or alternative API for social media search simulation
    response = requests.get(f"https://api.serper.dev/search?q={topic}", headers={"X-API-Key": os.environ["SERPER_API_KEY"]})
    if response.status_code == 200:
        mentions = response.json().get("results", [])
        return f"Top mentions for {topic}: {mentions}"
    return "No social media data available."

# Function to analyze sentiment using OpenAI's GPT-4o-mini model with chat completion
def analyze_sentiment(text):
    sentiment_prompt = f"Analyze the sentiment of the following text and classify it as Positive, Negative, or Neutral: {text}"
    sentiment = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # Specifying the GPT-4o-mini model
        messages=[{"role": "user", "content": sentiment_prompt}],
        max_tokens=10
    )
    return sentiment.choices[0].message['content'].strip()

# Function to compile and save the report to a CSV file in GitHub
def generate_report(topic, research_summary, social_media_summary, sentiment_analysis):
    # Create or append to a CSV file in GitHub
    csv_url = "https://raw.githubusercontent.com/yourusername/yourrepo/main/report.csv"  # replace with your actual CSV file URL in GitHub
    with open(csv_url, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Topic", "Research Summary", "Social Media Summary", "Sentiment Analysis"])
        writer.writerow([topic, research_summary, social_media_summary, sentiment_analysis])

# Streamlit UI Setup
st.title("Unified Brand and Topic Analysis without Database")
st.write("Analyze a brand or topic with integrated research, social media monitoring, sentiment analysis, and report generation.")

# User input for brand or topic and model selection
topic_or_brand = st.text_input("Enter the Brand or Topic Name")
use_gpt = st.checkbox("Use GPT-4o-mini model (uncheck for Llama)")

# Run the analysis on button click
if st.button("Start Integrated Analysis"):
    if topic_or_brand:
        llm = create_llm(use_gpt)
        
        # Perform each task
        st.write("Starting Research...")
        research_summary = research_topic(topic_or_brand, llm)
        st.write("Research Summary:", research_summary)
        
        st.write("Monitoring Social Media...")
        social_media_summary = monitor_social_media(topic_or_brand)
        st.write("Social Media Summary:", social_media_summary)
        
        st.write("Analyzing Sentiment...")
        sentiment_analysis = analyze_sentiment(social_media_summary)
        st.write("Sentiment Analysis:", sentiment_analysis)
        
        # Generate final report
        st.write("Generating Report...")
        generate_report(topic_or_brand, research_summary, social_media_summary, sentiment_analysis)
        st.success("Report generated and saved to CSV.")
    else:
        st.error("Please enter a brand or topic name to proceed.")
