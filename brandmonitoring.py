import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import os
import time
import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
import openai
import matplotlib.pyplot as plt
import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import re

nltk.download('vader_lexicon')

# Load environment variables from .env file
load_dotenv()

# Set API keys from Streamlit secrets
os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Initialize social media search tool and sentiment analyzer
search_tool = SerperDevTool()
sentiment_analyzer = SentimentIntensityAnalyzer()

# Function to create LLM
def create_llm():
    return ChatOpenAI(model="gpt-4")

# Enhanced function to fetch social media mentions and news with error handling
def fetch_mentions(brand_name):
    try:
        tool_output = search_tool(brand_name)  # Directly call the tool
        mentions = parse_tool_output(tool_output)
        return mentions
    except Exception as e:
        st.warning(f"Could not retrieve data. Error: {e}")
        return []

# Parse tool output to extract structured data
def parse_tool_output(tool_output):
    entries = re.findall(r"Title: (.+?)\n\nLink: (.+?)\n\nSnippet: (.+?)(?=\n---|\Z)", tool_output, re.DOTALL)
    parsed_results = [{"title": title.strip(), "link": link.strip(), "snippet": snippet.strip()} for title, link, snippet in entries]
    return parsed_results

# Display mentions in readable format
def display_mentions(parsed_results):
    st.subheader("2. Social Media Mentions")
    if parsed_results:
        for entry in parsed_results:
            st.markdown(f"**{entry['title']}**")
            st.markdown(f"[Read more]({entry['link']})")
            st.write(entry['snippet'])
            st.write("---")
    else:
        st.write("No mentions data available.")

# Analyze sentiment by platform
def analyze_sentiment_by_platform(parsed_results):
    platform_sentiments = {"positive": 0, "negative": 0, "neutral": 0}
    
    for entry in parsed_results:
        snippet = entry["snippet"]
        sentiment_score = sentiment_analyzer.polarity_scores(snippet)
        if sentiment_score["compound"] >= 0.05:
            platform_sentiments["positive"] += 1
        elif sentiment_score["compound"] <= -0.05:
            platform_sentiments["negative"] += 1
        else:
            platform_sentiments["neutral"] += 1

    return platform_sentiments

# Display sentiment charts per platform
def display_sentiment_charts(sentiment_results):
    for platform, sentiments in sentiment_results.items():
        st.subheader(f"Sentiment Distribution")
        labels = ['Positive', 'Negative', 'Neutral']
        sizes = [sentiments.get("positive", 0), sentiments.get("negative", 0), sentiments.get("neutral", 0)]

        total_mentions = sum(sizes)
        if total_mentions == 0:
            st.write(f"No sentiment data available.")
            continue

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        ax.axis('equal')
        st.pyplot(fig)

# Extract key themes from mentions
def extract_key_themes(parsed_results):
    text_data = [entry["snippet"] for entry in parsed_results]
    
    if not text_data:
        st.warning("No text data available in mentions to extract themes.")
        return {}

    vectorizer = CountVectorizer(stop_words='english', max_features=10)
    X = vectorizer.fit_transform(text_data)
    word_counts = Counter(X.toarray().sum(axis=0))
    
    themes = {word: {"description": f"High frequency mention of '{word}'"} for word in vectorizer.get_feature_names_out()}
    return themes

# Generate recommendations based on themes
def generate_recommendations(themes):
    recommendations = []
    if "negative" in themes:
        recommendations.append({"recommendation": "Address key topics generating negative sentiment to improve public perception."})
    if "positive" in themes:
        recommendations.append({"recommendation": "Continue engagement on positive themes to maintain favorable sentiment."})
    return recommendations

# Display formatted report
def display_formatted_report(brand_name, parsed_results):
    st.header(f"Social Media and Sentiment Analysis Report for {brand_name}")
    st.write("---")

    # Section 1: Research Findings
    st.subheader("1. Research Findings")
    st.write(f"Fetched data for {brand_name}. Here’s an overview of their recent activities and online presence:")

    # Section 2: Social Media Mentions
    display_mentions(parsed_results)

    # Section 3: Sentiment Analysis
    st.subheader("3. Sentiment Analysis")
    sentiment_results = analyze_sentiment_by_platform(parsed_results)
    display_sentiment_charts(sentiment_results)

    # Section 4: Key Themes and Recommendations
    st.subheader("4. Key Themes and Recommendations")
    themes = extract_key_themes(parsed_results)
    recommendations = generate_recommendations(themes)

    st.write("**Notable Themes:**")
    if themes:
        for theme, info in themes.items():
            st.write(f"- **{theme}**: {info['description']}")
    else:
        st.write("No notable themes identified due to insufficient data.")

    st.write("**Recommendations:**")
    if recommendations:
        for rec in recommendations:
            st.write(f"- {rec['recommendation']}")
    else:
        st.write("No specific recommendations provided.")

# Streamlit app interface
st.title("Social Media Monitoring and Sentiment Analysis")
st.write("Analyze a brand or topic with integrated social media monitoring, sentiment analysis, and report generation.")

# User input for brand or topic
brand_name = st.text_input("Enter the Brand or Topic Name")

# Run the analysis on button click
if st.button("Start Analysis"):
    if brand_name:
        st.write("Starting social media monitoring and sentiment analysis...")
        parsed_results = fetch_mentions(brand_name)
        
        if parsed_results:
            display_formatted_report(brand_name, parsed_results)
        else:
            st.error("Failed to generate the report. Please try again.")
    else:
        st.error("Please enter a brand or topic name to proceed.")
