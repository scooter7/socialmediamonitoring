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

# Create agents with crewai for research and analysis
def create_agents(brand_name, llm):
    researcher = Agent(
        role="Social Media Researcher",
        goal=f"Research and gather information about {brand_name} from various sources.",
        backstory="You are an expert researcher with a knack for finding relevant information quickly.",
        verbose=False,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
        max_iter=10
    )

    social_media_monitor = Agent(
        role="Social Media Monitor",
        goal=f"Monitor social media platforms for mentions of {brand_name}.",
        backstory="You are an experienced social media analyst with keen eyes for trends and mentions.",
        verbose=False,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
        max_iter=10
    )

    sentiment_analyzer = Agent(
        role="Sentiment Analyzer",
        goal=f"Analyze the sentiment of social media mentions about {brand_name}.",
        backstory="You are an expert in natural language processing and sentiment analysis.",
        verbose=False,
        allow_delegation=False,
        llm=llm,
        max_iter=10
    )

    report_generator = Agent(
        role="Report Generator",
        goal=f"Generate a comprehensive report based on the analysis of {brand_name}.",
        backstory="You are a skilled data analyst and report writer, adept at presenting insights clearly.",
        verbose=False,
        allow_delegation=False,
        llm=llm,
        max_iter=10
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
        verbose=False
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

# Format the final report nicely
def display_formatted_report(brand_name, result):
    st.header(f"Social Media and Sentiment Analysis Report for {brand_name}")
    st.write("---")

    # Extract task outputs
    task_outputs = result.tasks_output

    # Section 1: Research Findings
    st.subheader("1. Research Findings")
    research_output = task_outputs[0].raw
    st.write(research_output)

    # Section 2: Social Media Mentions
    st.subheader("2. Social Media Mentions")
    mentions_output = task_outputs[1].raw
    st.write(mentions_output)

    # Section 3: Sentiment Analysis
    st.subheader("3. Sentiment Analysis")
    sentiment_output = task_outputs[2].raw
    st.write(sentiment_output)

    # Extract sentiment percentages for visualization
    try:
        # Simple parsing to extract percentages
        positive_pct = int(sentiment_output.split('Positive Mentions: ')[1].split('%')[0])
        negative_pct = int(sentiment_output.split('Negative Mentions: ')[1].split('%')[0])
        neutral_pct = int(sentiment_output.split('Neutral Mentions: ')[1].split('%')[0])

        # Pie chart visualization
        st.subheader("Sentiment Distribution")
        labels = ['Positive', 'Negative', 'Neutral']
        sizes = [positive_pct, negative_pct, neutral_pct]
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        ax.axis('equal')
        st.pyplot(fig)
    except Exception as e:
        st.write("Unable to extract sentiment percentages for visualization.")

    # Section 4: Key Themes Identified and Recommendations
    st.subheader("4. Key Themes and Recommendations")
    report_output_raw = task_outputs[3].raw

    # Extract JSON data from the raw output
    try:
        json_str = report_output_raw.strip('```json\n').strip('\n```')
        report_data = json.loads(json_str)
        themes = report_data['report']['notable_themes']
        recommendations = report_data['report']['conclusion']['recommendations']

        # Display themes
        st.write("**Notable Themes:**")
        for theme_key, theme_info in themes.items():
            st.write(f"- **{theme_key.replace('_', ' ').title()}**: {theme_info['description']}")

        # Display recommendations
        st.write("**Recommendations:**")
        for rec in recommendations:
            st.write(f"- {rec['recommendation']}")

    except Exception as e:
        st.write("Error parsing the JSON-formatted report.")
        st.write(str(e))

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
