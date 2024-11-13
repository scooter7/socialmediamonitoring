# Required Libraries
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import os
import time
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
import openai

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

# Create agents with crewai for research and analysis
def create_agents(brand_name, llm):
    researcher = Agent(
        role="Social Media Researcher",
        goal=f"Research and gather information about {brand_name} from various sources",
        backstory="Expert researcher with a knack for finding relevant information.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
        max_iter=15
    )

    social_media_monitor = Agent(
        role="Social Media Monitor",
        goal=f"Monitor social media platforms for mentions of {brand_name}",
        backstory="Experienced social media analyst with keen eyes for trends.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
        max_iter=15
    )

    sentiment_analyzer = Agent(
        role="Sentiment Analyzer",
        goal=f"Analyze the sentiment of social media mentions about {brand_name}",
        backstory="Expert in natural language processing and sentiment analysis.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=15
    )

    report_generator = Agent(
        role="Report Generator",
        goal=f"Generate comprehensive reports based on the analysis of {brand_name}",
        backstory="Skilled data analyst and report writer.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=15
    )

    return [researcher, social_media_monitor, sentiment_analyzer, report_generator]

# Define tasks with crewai
def create_tasks(brand_name, agents):
    research_task = Task(
        description=f"Research {brand_name} and provide a summary of online presence.",
        agent=agents[0],
        expected_output="Structured summary with key insights and notable mentions."
    )

    monitoring_task = Task(
        description=f"Monitor social media for mentions of '{brand_name}' and summarize.",
        agent=agents[1],
        expected_output="Summary of mentions with platform breakdown."
    )

    sentiment_analysis_task = Task(
        description=f"Analyze the sentiment of the social media mentions about {brand_name}.",
        agent=agents[2],
        expected_output="Sentiment breakdown with notable themes."
    )

    report_generation_task = Task(
        description=f"Generate a comprehensive report for {brand_name}.",
        agent=agents[3],
        expected_output="Comprehensive report with insights and recommendations."
    )

    return [research_task, monitoring_task, sentiment_analysis_task, report_generation_task]

# Run social media monitoring and sentiment analysis workflow
def run_social_media_monitoring(brand_name, max_retries=3):
    llm = create_llm()
    agents = create_agents(brand_name, llm)
    tasks = create_tasks(brand_name, agents)
    
    crew = Crew(agents=agents, tasks=tasks, verbose=True)

    for attempt in range(max_retries):
        try:
            result = crew.kickoff()
            return result  # Return full result
        except Exception as e:
            st.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                st.write("Retrying...")
                time.sleep(5)
            else:
                st.error("Max retries reached. Unable to complete the task.")
                return None

# Streamlit app interface
st.title("Social Media Monitoring and Sentiment Analysis")
st.write("Analyze a brand or topic with integrated social media monitoring, sentiment analysis, and report generation.")

# User input for brand or topic
brand_name = st.text_input("Enter the Brand or Topic Name")

# Function to parse and display report data
def display_report(result):
    st.header(f"Report for {brand_name}")

    # Extracting fields directly from CrewOutput
    research_findings = getattr(result, "Research Findings", None)
    social_media_mentions = getattr(result, "Social Media Mentions", None)
    sentiment_analysis = getattr(result, "Sentiment Analysis", None)
    recommendations = getattr(result, "Recommendations", None)

    # Display Research Findings
    if research_findings:
        st.subheader("Research Findings")
        st.write(research_findings)
    else:
        st.info("No research findings available.")

    # Display Social Media Monitoring Summary
    if social_media_mentions:
        st.subheader("Social Media Monitoring Summary")
        st.write(social_media_mentions)

        # Visualization: Mentions Breakdown
        try:
            platforms = list(social_media_mentions.keys())
            mentions = list(social_media_mentions.values())
            fig, ax = plt.subplots()
            ax.bar(platforms, mentions, color="skyblue")
            ax.set_title("Social Media Mentions by Platform")
            ax.set_xlabel("Platform")
            ax.set_ylabel("Mentions")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error displaying social media mentions chart: {str(e)}")
    else:
        st.info("No social media mentions data available.")

    # Display Sentiment Analysis
    if sentiment_analysis:
        st.subheader("Sentiment Analysis")
        st.write(sentiment_analysis)

        # Visualization: Sentiment Breakdown
        try:
            sentiment_counts = {
                "Positive": sentiment_analysis.get("Positive", 0),
                "Neutral": sentiment_analysis.get("Neutral", 0),
                "Negative": sentiment_analysis.get("Negative", 0)
            }
            fig, ax = plt.subplots()
            ax.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct="%1.1f%%", startangle=140)
            ax.axis("equal")
            ax.set_title("Sentiment Distribution")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error displaying sentiment chart: {str(e)}")
    else:
        st.info("No sentiment analysis data available.")

    # Display Generated Recommendations
    if recommendations:
        st.subheader("Recommendations")
        st.write(recommendations)
    else:
        st.info("No recommendations available.")

# Run the analysis on button click
if st.button("Start Analysis"):
    if brand_name:
        st.write("Starting social media monitoring and sentiment analysis...")
        result = run_social_media_monitoring(brand_name)
        
        if result:
            display_report(result)
        else:
            st.error("Failed to generate the report. Please try again.")
    else:
        st.error("Please enter a brand or topic name to proceed.")
