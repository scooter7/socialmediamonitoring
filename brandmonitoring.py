import os
import streamlit as st
import openai
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import requests
import json
import base64

# Load environment and set up Streamlit Secrets for API keys
load_dotenv()

# Directly set environment variables and API keys from Streamlit secrets
os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Function to create LLM using GPT-4o-mini only
def create_llm():
    return ChatOpenAI(model="gpt-4o-mini")

# Function to perform topic research using LLM
def research_topic(topic, llm):
    prompt = f"Conduct a thorough research on the following topic: {topic}. Provide a summary with key points and insights."
    response = llm.generate([prompt])  # Pass prompt as a list
    return response

# Function to monitor social media for mentions of a brand/topic
def monitor_social_media(topic):
    response = requests.get(f"https://api.serper.dev/search?q={topic}", headers={"X-API-Key": os.environ["SERPER_API_KEY"]})
    if response.status_code == 200:
        mentions = response.json().get("results", [])
        return f"Top mentions for {topic}: {mentions}"
    return "No social media data available."

# Function to analyze sentiment using OpenAI's GPT-4o-mini model with chat completion
def analyze_sentiment(text):
    sentiment_prompt = f"Analyze the sentiment of the following text and classify it as Positive, Negative, or Neutral: {text}"
    sentiment = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": sentiment_prompt}],
        max_tokens=10
    )
    return sentiment.choices[0].message.content.strip()

# Function to compile and save the report to a CSV file in GitHub
def generate_report(topic, research_summary, social_media_summary, sentiment_analysis):
    # GitHub repository details
    repo_owner = "yourusername"  # Replace with your GitHub username
    repo_name = "yourrepo"       # Replace with your GitHub repository name
    file_path = "report.csv"     # Path in the repository where the file will be saved
    github_token = st.secrets["GITHUB_TOKEN"]  # Store your GitHub token in Streamlit secrets

    # Prepare CSV content
    csv_content = "Topic,Research Summary,Social Media Summary,Sentiment Analysis\n"
    csv_content += f'"{topic}","{research_summary}","{social_media_summary}","{sentiment_analysis}"\n'
    encoded_content = base64.b64encode(csv_content.encode()).decode()

    # GitHub API URL to create or update a file
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}"
    
    # Prepare the request payload
    payload = {
        "message": "Add report entry",
        "content": encoded_content,
        "branch": "main"  # Update if you want to save to a different branch
    }

    # Make a GET request to check if the file already exists to get the `sha`
    response = requests.get(url, headers={"Authorization": f"token {github_token}"})
    if response.status_code == 200:
        # File exists, add 'sha' to payload for update
        payload["sha"] = response.json()["sha"]

    # Make the PUT request to create or update the file
    response = requests.put(url, headers={"Authorization": f"token {github_token}"}, data=json.dumps(payload))

    if response.status_code in [200, 201]:
        st.success("Report generated and saved to GitHub.")
    else:
        st.error("Failed to save report to GitHub.")

# Streamlit UI Setup
st.title("Unified Brand and Topic Analysis without Database")
st.write("Analyze a brand or topic with integrated research, social media monitoring, sentiment analysis, and report generation.")

# User input for brand or topic
topic_or_brand = st.text_input("Enter the Brand or Topic Name")

# Run the analysis on button click
if st.button("Start Integrated Analysis"):
    if topic_or_brand:
        llm = create_llm()
        
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
    else:
        st.error("Please enter a brand or topic name to proceed.")
