import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import os
import time
import re
import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
import openai
import matplotlib.pyplot as plt
import json
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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
    sources = ["Twitter", "Facebook", "Reddit", "Quora", "News"]
    mentions = {}
    for source in sources:
        try:
            # Attempt to fetch mentions for each platform using `_run` method
            result = search_tool._run(brand_name)
            mentions[source] = parse_tool_output(result) if result else []
        except Exception as e:
            st.warning(f"Could not retrieve data from {source}. Error: {e}")
            mentions[source] = []  # Store an empty list if an error occurs
    return mentions

# Parse tool output to extract structured data
def parse_tool_output(tool_output):
    entries = re.findall(r"Title: (.+?)\n\nLink: (.+?)\n\nSnippet: (.+?)(?=\n---|\Z)", tool_output, re.DOTALL)
    return [{"title": title.strip(), "link": link.strip(), "snippet": snippet.strip()} for title, link, snippet in entries]

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

# Define tasks for each agent with CrewAI
def create_tasks(brand_name, agents):
    research_task = Task(
        description=f"Research {brand_name} and provide a summary of their online presence, key information, and recent activities.",
        agent=agents[0],
        expected_output="A structured summary containing: \n1. Brief overview of {brand_name}\n2. Key online platforms and follower counts\n3. Recent notable activities or campaigns\n4. Main products or services\n5. Any recent news or controversies"
    )

    monitoring_task = Task(
        description=f"Monitor social media platforms for mentions of '{brand_name}' in the last 24 hours. Provide a summary of the mentions.",
        agent=agents[1],
        expected_output="A structured report containing: \n1. Total number of mentions\n2. Breakdown by platform (e.g., Twitter, Instagram, Facebook)\n3. Top 5 most engaging posts or mentions\n4. Any trending hashtags associated with {brand_name}\n5. Notable influencers or accounts mentioning {brand_name}"
    )

    sentiment_analysis_task = Task(
        description=f"Analyze the sentiment of the social media mentions about {brand_name}. Categorize them as positive, negative, or neutral.",
        agent=agents[2],
        expected_output="A sentiment analysis report containing: \n1. Overall sentiment distribution (% positive, negative, neutral)\n2. Key positive themes or comments\n3. Key negative themes or comments\n4. Any notable changes in sentiment compared to previous periods\n5. Suggestions for sentiment improvement if necessary"
    )

    report_generation_task = Task(
        description=f"Generate a comprehensive report about {brand_name} based on the research, social media mentions, and sentiment analysis. Include key insights and recommendations.",
        agent=agents[3],
        expected_output="A comprehensive report structured as follows: \n1. Executive Summary\n2. Brand Overview\n3. Social Media Presence Analysis\n4. Sentiment Analysis\n5. Key Insights\n6. Recommendations for Improvement\n7. Conclusion"
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
    sentiment_output = task_outputs[2].raw if task_outputs[2] else "No sentiment data available"
    st.write(sentiment_output)

    # Section 4: Key Themes and Recommendations
    st.subheader("4. Key Themes and Recommendations")
    report_output = task_outputs[3].raw if task_outputs[3] else "No report data available"
    
    # Try to parse JSON data from the report generator output
    try:
        report_data = json.loads(report_output.strip('```json\n').strip('\n```'))
        themes = report_data.get('notable_themes', {})
        recommendations = report_data.get('conclusion', {}).get('recommendations', [])

        # Display themes
        st.write("**Notable Themes:**")
        for theme_key, theme_info in themes.items():
            st.write(f"- **{theme_key.replace('_', ' ').title()}**: {theme_info['description']}")

        # Display recommendations
        st.write("**Recommendations:**")
        for rec in recommendations:
            st.write(f"- {rec['recommendation']}")

    except (json.JSONDecodeError, KeyError) as e:
        st.write("Error parsing the JSON-formatted report.")
        st.write(report_output)

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
