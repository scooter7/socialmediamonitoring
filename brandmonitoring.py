import json
import re
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

# Load environment variables from .env file
load_dotenv()

# Set API keys from Streamlit secrets
os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Initialize social media search tool
search_tool = SerperDevTool()

# Function to create LLM using GPT-4o-mini
def create_llm():
    return ChatOpenAI(model="gpt-4o-mini")

# Create agents with CrewAI for research and analysis
def create_agents(brand_name, llm):
    researcher = Agent(
        role="Social Media Researcher",
        goal=f"Research and gather information about {brand_name} from various sources.",
        backstory="You are an expert researcher who quickly finds relevant information.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
        max_iter=15
    )

    social_media_monitor = Agent(
        role="Social Media Monitor",
        goal=f"Monitor social media platforms for mentions of {brand_name}.",
        backstory="You are an experienced social media analyst with a focus on trends and mentions.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
        max_iter=15
    )

    sentiment_analyzer = Agent(
        role="Sentiment Analyzer",
        goal=f"Analyze the sentiment of social media mentions about {brand_name}.",
        backstory="You are skilled in natural language processing and sentiment analysis.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=15
    )

    report_generator = Agent(
        role="Report Generator",
        goal=f"Generate comprehensive reports based on the analysis of {brand_name}.",
        backstory="You are a data analyst and report writer who presents insights clearly.",
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

# Run online monitoring and sentiment analysis workflow
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

# Function to parse and display online mentions data in a structured format for Section 2
def parse_and_display_mentions(report_output):
    try:
        # Load JSON data from the CrewAI response
        report_data = json.loads(report_output.strip('```json\n').strip('\n```'))
        
        # Search for potential mentions in JSON data by navigating through all nested keys
        st.write("### Online Mentions and Sentiment Analysis")
        found_data = False
        
        # A recursive function to search for `title`, `link`, and `snippet` in any nested structure
        def search_mentions(data):
            nonlocal found_data
            if isinstance(data, dict):
                # If the dict contains title, link, and snippet, we assume it’s a mention
                if "title" in data and "link" in data and "snippet" in data:
                    st.write(f"**Title:** {data['title']}")
                    st.write(f"**Link:** {data['link']}")
                    st.write(f"**Snippet:** {data['snippet']}")
                    st.write("---")
                    found_data = True
                # Otherwise, recursively search nested dictionaries
                for key, value in data.items():
                    search_mentions(value)
            elif isinstance(data, list):
                # If it's a list, check each item
                for item in data:
                    search_mentions(item)
        
        # Start searching mentions within parsed JSON data
        search_mentions(report_data)
        
        if not found_data:
            st.write("No online mentions or sentiment data available.")
            
    except json.JSONDecodeError:
        st.error("Error parsing JSON data. Please check the JSON format.")

# Display formatted report based on task outputs
def display_formatted_report(brand_name, result):
    st.header(f"Online and Sentiment Analysis Report for {brand_name}")
    st.write("---")

    # Extract task outputs and display Section 1: Research Findings
    st.subheader("1. Research Findings")
    research_output = result.tasks_output[0].raw if result.tasks_output[0] else "No data available"
    st.write(research_output)

    # Section 2: Online Mentions and Sentiment Analysis - structured display for clarity
    st.subheader("2. Online Mentions and Sentiment Analysis")
    report_output = result.tasks_output[3].raw if result.tasks_output[3] else "No report data available"
    if report_output:
        parse_and_display_mentions(report_output)
    else:
        st.write("No online mentions or sentiment data available.")

    # Section 3: Recommendations
    st.subheader("3. Recommendations")
    report_data = json.loads(report_output.strip('```json\n').strip('\n```')) if report_output else {}
    recommendations = report_data.get('report', {}).get('recommendations', [])
    if recommendations:
        for recommendation in recommendations:
            st.write(f"- {recommendation}")
    else:
        st.write("No recommendations available.")

# Streamlit app interface
st.title("Online and Sentiment Analysis Report")
st.write("Analyze a brand or topic with integrated online monitoring, sentiment analysis, and report generation.")

# User input for brand or topic with a unique key
brand_name = st.text_input("Enter the Brand or Topic Name", key="brand_name_input")

# Run the analysis on button click with a unique key for the button
if st.button("Start Analysis", key="start_analysis_button"):
    if brand_name:
        st.write("Starting online monitoring and sentiment analysis...")
        result = run_social_media_monitoring(brand_name)
        
        if result:
            display_formatted_report(brand_name, result)
        else:
            st.error("Failed to generate the report. Please try again.")
    else:
        st.error("Please enter a brand or topic name to proceed.")
