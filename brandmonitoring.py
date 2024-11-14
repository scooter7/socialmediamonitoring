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
        backstory="Expert researcher with a knack for finding relevant information quickly.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
        max_iter=15
    )
    social_media_monitor = Agent(
        role="Social Media Monitor",
        goal=f"Monitor social media platforms for mentions of {brand_name}",
        backstory="Experienced social media analyst.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
        max_iter=15
    )
    sentiment_analyzer = Agent(
        role="Sentiment Analyzer",
        goal=f"Analyze sentiment of social media mentions about {brand_name}",
        backstory="Expert in NLP and sentiment analysis.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=15
    )
    report_generator = Agent(
        role="Report Generator",
        goal=f"Generate a report based on analysis of {brand_name}",
        backstory="Skilled data analyst and report writer.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=15
    )
    return [researcher, social_media_monitor, sentiment_analyzer, report_generator]

# Define tasks with CrewAI
def create_tasks(brand_name, agents):
    research_task = Task(
        description=f"Research {brand_name} and summarize online presence and activities.",
        agent=agents[0],
        expected_output="A summary of recent activities and presence."
    )
    monitoring_task = Task(
        description=f"Monitor social media platforms for mentions of '{brand_name}'.",
        agent=agents[1],
        expected_output="Summary of mentions by platform."
    )
    sentiment_analysis_task = Task(
        description=f"Analyze sentiment of the social media mentions about {brand_name}.",
        agent=agents[2],
        expected_output="Sentiment breakdown and themes."
    )
    report_generation_task = Task(
        description=f"Generate a JSON-formatted report for {brand_name} based on findings.",
        agent=agents[3],
        expected_output="JSON formatted report with insights and recommendations."
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
            output = {}

            if hasattr(result, "tasks_output"):
                for task_output in result.tasks_output:
                    if hasattr(task_output, "description"):
                        output[task_output.description] = {
                            "summary": getattr(task_output, "summary", "N/A"),
                            "raw_output": getattr(task_output, "raw", "N/A"),
                            "json_output": task_output.json_dict if hasattr(task_output, "json_dict") else None
                        }
            elif hasattr(result, "json_dict"):
                output = result.json_dict
            
            return output
        except Exception as e:
            st.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                st.write("Retrying...")
                time.sleep(5)
            else:
                st.error("Max retries reached. Unable to complete the task.")
                return None

# Function to display the formatted report
def display_report(brand_name, result):
    st.header(f"Social Media and Sentiment Analysis Report for {brand_name}")

    # Research Findings
    if "Research Findings" in result:
        st.subheader("1. Research Findings")
        st.write(result.get("Research Findings", "No research findings available."))

    # Social Media Mentions
    if "Social Media Mentions" in result:
        st.subheader("2. Social Media Mentions")
        for platform, mentions in result["Social Media Mentions"].items():
            st.markdown(f"**{platform}**")
            for mention in mentions:
                st.markdown(f"- {mention}")

    # Sentiment Analysis
    if "Sentiment Analysis" in result:
        st.subheader("3. Sentiment Analysis")
        sentiment_data = result["Sentiment Analysis"]
        for sentiment, count in sentiment_data.items():
            st.markdown(f"**{sentiment.capitalize()} Mentions:** {count}")

    # Recommendations
    if "Recommendations" in result:
        st.subheader("4. Recommendations")
        recommendations = result["Recommendations"]
        for rec in recommendations:
            st.markdown(f"- **{rec['recommendation']}**: {rec['rationale']}")

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
