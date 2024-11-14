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
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Set API keys from Streamlit secrets
os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Initialize social media search tool
search_tool = SerperDevTool()

# Function to create LLM
def create_llm():
    return ChatOpenAI(model="gpt-4o-mini")

# Create agents with CrewAI for research and analysis
def create_agents(brand_name, llm):
    researcher = Agent(
        role="Social Media Researcher",
        goal=f"Research and gather information about {brand_name} from various sources",
        backstory="Expert researcher adept at finding relevant information.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
        max_iter=15
    )
    social_media_monitor = Agent(
        role="Social Media Monitor",
        goal=f"Monitor social media platforms for mentions of {brand_name}",
        backstory="Specializes in monitoring social media and detecting trends.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
        max_iter=15
    )
    sentiment_analyzer = Agent(
        role="Sentiment Analyzer",
        goal=f"Analyze the sentiment of social media mentions about {brand_name}",
        backstory="Expert in sentiment analysis and natural language processing.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=15
    )
    report_generator = Agent(
        role="Report Generator",
        goal=f"Generate a JSON-formatted report based on the analysis of {brand_name}",
        backstory="Experienced in data analysis and creating readable reports.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=15
    )
    return [researcher, social_media_monitor, sentiment_analyzer, report_generator]

# Define tasks with CrewAI and set JSON output
def create_tasks(brand_name, agents):
    research_task = Task(
        description=f"Research {brand_name} and provide a summary of their online presence.",
        agent=agents[0],
        expected_output="Structured summary with key insights on recent activities."
    )
    monitoring_task = Task(
        description=f"Monitor social media for mentions of '{brand_name}' and summarize findings.",
        agent=agents[1],
        expected_output="Summary of mentions including counts, platforms, notable mentions."
    )
    sentiment_analysis_task = Task(
        description=f"Analyze the sentiment of the social media mentions about {brand_name}.",
        agent=agents[2],
        expected_output="Sentiment breakdown and notable themes."
    )
    report_generation_task = Task(
        description=f"Generate a JSON-formatted report about {brand_name} based on the research and sentiment analysis.",
        agent=agents[3],
        expected_output="Detailed report in JSON format including insights and recommendations.",
        output_json=True  # Ensure output is in JSON format
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
            result = crew.kickoff().json()  # Retrieve the JSON output
            return result
        except Exception as e:
            st.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                st.write("Retrying...")
                time.sleep(5)
            else:
                st.error("Max retries reached. Unable to complete the task.")
                return None

# Function to visualize data in charts
def create_visualizations(mentions_data, sentiment_data):
    # Social media mention breakdown
    if mentions_data:
        mention_counts = pd.DataFrame(mentions_data).groupby("platform").size().reset_index(name="counts")
        st.subheader("Social Media Mentions by Platform")
        plt.figure(figsize=(8, 5))
        plt.bar(mention_counts["platform"], mention_counts["counts"])
        plt.title("Mentions by Platform")
        plt.xlabel("Platform")
        plt.ylabel("Mentions")
        st.pyplot(plt)

    # Sentiment distribution
    if sentiment_data:
        sentiment_df = pd.DataFrame(sentiment_data, columns=["Sentiment", "Count"])
        st.subheader("Sentiment Analysis Distribution")
        plt.figure(figsize=(8, 5))
        plt.pie(sentiment_df["Count"], labels=sentiment_df["Sentiment"], autopct="%1.1f%%")
        plt.title("Sentiment Distribution")
        st.pyplot(plt)

# Display formatted report
def display_report(brand_name, result):
    st.header(f"Report for {brand_name}")
    if "Research Findings" in result:
        st.subheader("Research Findings")
        st.write(result["Research Findings"])

    if "Social Media Mentions" in result:
        st.subheader("Social Media Mentions")
        social_media_mentions = result["Social Media Mentions"]
        for platform, mentions in social_media_mentions.items():
            st.write(f"**{platform}**")
            for mention in mentions:
                st.write(f"- {mention}")
    
    if "Sentiment Analysis" in result:
        st.subheader("Sentiment Analysis")
        sentiment_analysis = result["Sentiment Analysis"]
        sentiment_summary = {sentiment: count for sentiment, count in sentiment_analysis.items()}
        create_visualizations(social_media_mentions, sentiment_summary)

    if "Recommendations" in result:
        st.subheader("Recommendations")
        st.write(result["Recommendations"])

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
            display_report(brand_name, result)
        else:
            st.error("Failed to generate the report. Please try again.")
    else:
        st.error("Please enter a brand or topic name to proceed.")
