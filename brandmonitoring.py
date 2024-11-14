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
nltk.download('vader_lexicon')

# Load environment variables from .env file
load_dotenv()

# Set API keys from Streamlit secrets
os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Initialize social media search tool
search_tool = SerperDevTool()

# Function to create LLM
def create_llm():
    return ChatOpenAI(model="gpt-4")

# Enhanced function to fetch social media mentions and news with error handling
def fetch_mentions(brand_name):
    sources = ["Twitter", "Facebook", "Reddit", "Quora", "News"]
    mentions = {}
    for source in sources:
        try:
            # Attempt to fetch mentions for each platform
            result = search_tool.run(brand_name)
            mentions[source] = result or []  # Assign an empty list if result is None or empty
        except Exception as e:
            st.warning(f"Could not retrieve data from {source}. Error: {e}")
            mentions[source] = []  # Store an empty list if an error occurs
    return mentions

# Analyze sentiment by platform with improved handling for missing data
def analyze_sentiment_by_platform(mentions):
    sentiment_results = {}
    for platform, posts in mentions.items():
        platform_sentiments = {"positive": 0, "negative": 0, "neutral": 0}
        if not posts:  # If no posts were retrieved for the platform
            sentiment_results[platform] = platform_sentiments
            continue
        for post in posts:
            # Example placeholder: replace with actual sentiment analysis
            sentiment = "neutral"  # Assuming a dummy sentiment
            platform_sentiments[sentiment] += 1
        sentiment_results[platform] = platform_sentiments
    return sentiment_results

# Display sentiment charts per platform with NaN handling
def display_sentiment_charts(sentiment_results):
    for platform, sentiments in sentiment_results.items():
        st.subheader(f"Sentiment Distribution on {platform}")
        
        # Handle NaN by converting values to 0 if NaN
        positive = sentiments.get("positive", 0) or 0
        negative = sentiments.get("negative", 0) or 0
        neutral = sentiments.get("neutral", 0) or 0

        # Check for total mentions to prevent division by zero
        total_mentions = positive + negative + neutral
        if total_mentions == 0:
            st.write(f"No sentiment data available for {platform}.")
            continue

        # Prepare sizes and labels for pie chart
        labels = ['Positive', 'Negative', 'Neutral']
        sizes = [positive, negative, neutral]

        # Create pie chart
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
        st.pyplot(fig)

# JSON parsing with error handling
def parse_report_output(report_output):
    try:
        json_str = report_output.strip('```json\n').strip('\n```')
        report_data = json.loads(json_str)
        return report_data
    except json.JSONDecodeError as e:
        st.write("Error parsing JSON report.")
        st.write(str(e))
        return None

# Display Key Themes and Recommendations
def display_key_themes_and_recommendations(report_data):
    themes = report_data.get('notable_themes', {})
    recommendations = report_data.get('conclusion', {}).get('recommendations', [])
    
    # Display Key Themes
    st.subheader("Key Themes")
    for theme, details in themes.items():
        st.write(f"- **{theme.replace('_', ' ').title()}**: {details.get('description', 'No description provided')}")

    # Display Recommendations
    st.subheader("Recommendations")
    for rec in recommendations:
        st.write(f"- {rec['recommendation']}")

# Create agents with crewai for research and analysis
def create_agents(brand_name, llm):
    researcher = Agent(
        role="Social Media Researcher",
        goal=f"Research and gather information about {brand_name} from various sources.",
        backstory="You are an expert researcher with a knack for finding relevant information quickly.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
        max_iter=15
    )

    social_media_monitor = Agent(
        role="Social Media Monitor",
        goal=f"Monitor social media platforms for mentions of {brand_name}.",
        backstory="You are an experienced social media analyst with keen eyes for trends and mentions.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
        max_iter=15
    )

    sentiment_analyzer = Agent(
        role="Sentiment Analyzer",
        goal=f"Analyze the sentiment of social media mentions about {brand_name}.",
        backstory="You are an expert in natural language processing and sentiment analysis.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=15
    )

    report_generator = Agent(
        role="Report Generator",
        goal=f"Generate comprehensive reports based on the analysis of {brand_name}.",
        backstory="You are a skilled data analyst and report writer, adept at presenting insights clearly.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=15
    )

    return [researcher, social_media_monitor, sentiment_analyzer, report_generator]

# Define tasks with crewai
def create_tasks(brand_name, agents):
    research_task = Task(
        description=f"Research {brand_name} and provide a summary of their online presence, key information, and recent activities.",
        agent=agents[0],
        expected_output="A structured summary with key insights on recent activities, platform presence, and notable mentions."
    )

    monitoring_task = Task(
        description=f"Monitor social media platforms for mentions of '{brand_name}'. Provide a summary of the mentions.",
        agent=agents[1],
        expected_output="Summary of mentions including counts, platforms, notable mentions, and hashtags."
    )

    sentiment_analysis_task = Task(
        description=f"Analyze the sentiment of the social media mentions about {brand_name}. Categorize them as positive, negative, or neutral.",
        agent=agents[2],
        expected_output="Sentiment distribution and notable themes."
    )

    report_generation_task = Task(
        description=f"Generate a JSON-formatted report for {brand_name} based on findings.",
        agent=agents[3],
        expected_output="Comprehensive report in JSON format including key insights and recommendations."
    )

    return [research_task, monitoring_task, sentiment_analysis_task, report_generation_task]

# Run social media monitoring and sentiment analysis workflow
def run_social_media_monitoring(brand_name, max_retries=3):
    llm = create_llm()
    agents = create_agents(brand_name, llm)
    tasks = create_tasks(brand_name, agents)
    
    crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=True
    )

    for attempt in range(max_retries):
        try:
            result = crew.kickoff()
            return result
        except Exception as e:
            st.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                st.write("Retrying...")
                time.sleep(5)
            else:
                st.error("Max retries reached. Unable to complete the task.")
                return None

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

def extract_key_themes(mentions):
    text_data = [post["text"] for platform_posts in mentions.values() for post in platform_posts]
    # Vectorize words to find common themes
    vectorizer = CountVectorizer(stop_words='english', max_features=10)
    X = vectorizer.fit_transform(text_data)
    word_counts = Counter(X.toarray().sum(axis=0))
    themes = {word: {"description": f"High frequency mention of {word}"} for word, count in word_counts.items()}
    return themes

def generate_recommendations(themes):
    recommendations = []
    if "negative" in themes:
        recommendations.append({"recommendation": "Address key topics generating negative sentiment to improve public perception."})
    if "positive" in themes:
        recommendations.append({"recommendation": "Continue engagement on positive themes to maintain favorable sentiment."})
    return recommendations

# Display formatted report based on task outputs
def display_formatted_report(brand_name, result):
    st.header(f"Social Media and Sentiment Analysis Report for {brand_name}")
    st.write("---")

    # Extract task outputs
    task_outputs = result.tasks_output

    # Section 1: Research Findings
    st.subheader("1. Research Findings")
    research_output = task_outputs[0].raw if task_outputs[0] else "No data available"
    st.write(research_output)

    # Section 2: Social Media Mentions
    st.subheader("2. Social Media Mentions")
    mentions_output = task_outputs[1].raw if task_outputs[1] else "No mentions data available"
    st.write(mentions_output)

    # Section 3: Sentiment Analysis
    st.subheader("3. Sentiment Analysis")
    
    # Directly use mentions data from the social media task output
    mentions = task_outputs[1].raw if task_outputs[1] else {"Twitter": [], "Facebook": [], "Reddit": [], "Quora": [], "News": []}
    sentiment_results = analyze_sentiment_by_platform(mentions)
    display_sentiment_charts(sentiment_results)

    # Section 4: Key Themes and Recommendations
    st.subheader("4. Key Themes and Recommendations")
    themes = extract_key_themes(mentions)
    recommendations = generate_recommendations(themes)

    # Display extracted themes
    st.write("**Notable Themes:**")
    if themes:
        for theme, info in themes.items():
            st.write(f"- **{theme}**: {info['description']}")
    else:
        st.write("No notable themes identified.")

    # Display generated recommendations
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
        result = run_social_media_monitoring(brand_name)
        
        if result:
            display_formatted_report(brand_name, result)
        else:
            st.error("Failed to generate the report. Please try again.")
    else:
        st.error("Please enter a brand or topic name to proceed.")
