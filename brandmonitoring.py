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
            # Debug output for result structure
            st.write("Debug: Crew.kickoff() output structure:")
            st.write(result)  # Print the raw output to analyze the structure

            # Check if 'tasks' key exists in result
            if "tasks" in result:
                return result["tasks"]
            else:
                st.error("Tasks key not found in CrewOutput. Please verify the output structure.")
                return None
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

# Run the analysis on button click
if st.button("Start Analysis"):
    if brand_name:
        st.write("Starting social media monitoring and sentiment analysis...")
        result = run_social_media_monitoring(brand_name)
        
        if result:
            st.subheader("Final Report:")
            
            # Process and Display the Report
            for task_name, task_output in result.items():
                st.markdown(f"### {task_name}")
                st.write(task_output)
            
            # Example: Display sentiment analysis as a pie chart
            if "Sentiment Analysis" in result:
                sentiment_data = result["Sentiment Analysis"]
                sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}

                for line in sentiment_data.splitlines():
                    for sentiment in sentiment_counts:
                        if sentiment in line:
                            sentiment_counts[sentiment] += 1

                # Plotting Pie Chart
                st.write("### Sentiment Analysis Breakdown")
                fig, ax = plt.subplots()
                ax.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct="%1.1f%%", startangle=140)
                ax.axis("equal")
                st.pyplot(fig)

            # Example: Display social media mentions as a bar chart
            if "Social Media Mentions" in result:
                mentions_data = result["Social Media Mentions"]
                platforms = ["Twitter", "Facebook", "Instagram", "Reddit", "LinkedIn"]
                mention_counts = {platform: 0 for platform in platforms}

                for line in mentions_data.splitlines():
                    for platform in platforms:
                        if platform in line:
                            mention_counts[platform] += 1

                # Plotting Bar Chart
                st.write("### Social Media Mentions Breakdown")
                fig, ax = plt.subplots()
                ax.bar(mention_counts.keys(), mention_counts.values())
                ax.set_ylabel("Number of Mentions")
                st.pyplot(fig)

        else:
            st.error("Failed to generate the report. Please try again.")
    else:
        st.error("Please enter a brand or topic name to proceed.")
