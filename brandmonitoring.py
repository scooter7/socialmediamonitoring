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
    return ChatOpenAI(model="gpt-4o-mini")

# Create agents with crewai for research and analysis
def create_agents(brand_name, llm):
    researcher = Agent(
        role="Social Media Researcher",
        goal=f"Research and gather information about {brand_name} from various sources",
        backstory="You are an expert researcher with a knack for finding relevant information quickly.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
        max_iter=15
    )

    social_media_monitor = Agent(
        role="Social Media Monitor",
        goal=f"Monitor social media platforms for mentions of {brand_name}",
        backstory="You are an experienced social media analyst with keen eyes for trends and mentions.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
        max_iter=15
    )

    sentiment_analyzer = Agent(
        role="Sentiment Analyzer",
        goal=f"Analyze the sentiment of social media mentions about {brand_name}",
        backstory="You are an expert in natural language processing and sentiment analysis.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=15
    )

    report_generator = Agent(
        role="Report Generator",
        goal=f"Generate comprehensive reports based on the analysis of {brand_name}",
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
        description=f"Generate a comprehensive report about {brand_name} based on the research and sentiment analysis.",
        agent=agents[3],
        expected_output="Comprehensive report including key insights and recommendations."
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

# Display the formatted report with charts and structure
def display_formatted_report(brand_name, result):
    st.title(f"Social Media and Sentiment Analysis Report for {brand_name}")
    st.write("---")

    # Display Debugging Data to Trace Output Structure
    st.write("Debug: Result Keys")
    st.write(list(result.keys()))

    # Section 1: Research Findings
    research_key = "Research DMACC and summarize online presence and activities."
    if research_key in result:
        st.subheader("1. Research Findings")
        research_data = result.get(research_key, {}).get("raw_output", "No research findings available.")
        st.write(research_data)
    else:
        st.write("No research findings available.")

    # Section 2: Social Media Mentions
    mentions_key = "Monitor social media platforms for mentions of 'DMACC'."
    if mentions_key in result:
        st.subheader("2. Social Media Mentions")
        mentions_data = result.get(mentions_key, {}).get("raw_output", "No social media mentions available.")
        st.write(mentions_data)
    else:
        st.write("No social media mentions available.")

    # Section 3: Sentiment Analysis
    sentiment_key = "Analyze sentiment of the social media mentions about DMACC."
    if sentiment_key in result:
        st.subheader("3. Sentiment Analysis")
        sentiment_data = result.get(sentiment_key, {}).get("raw_output", "No sentiment analysis available.")
        st.write(sentiment_data)
        
        # Display Sentiment Distribution Chart if data is available
        if "positive" in sentiment_data.lower() or "negative" in sentiment_data.lower():
            st.subheader("Sentiment Distribution")
            sentiment_breakdown = {"Positive": 70, "Neutral": 20, "Negative": 10}  # Sample values for testing
            labels = list(sentiment_breakdown.keys())
            sizes = list(sentiment_breakdown.values())
            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
            ax.axis('equal')
            st.pyplot(fig)
    else:
        st.write("No sentiment analysis available.")

    # Section 4: Themes Identified
    st.subheader("4. Key Themes Identified")
    themes_key = "Generate a JSON-formatted report for DMACC based on findings."
    if themes_key in result:
        json_data = json.loads(result[themes_key]["raw_output"]).get("report", {}).get("themes_identified", [])
        if json_data:
            for theme in json_data:
                st.write(f"**Theme**: {theme['theme']}")
                st.write(f"Description: {theme['description']}")
        else:
            st.write("No themes identified.")
    else:
        st.write("No themes identified.")

    # Section 5: Recommendations
    st.subheader("5. Recommendations")
    recommendations_key = themes_key
    if recommendations_key in result:
        recommendations = json.loads(result[recommendations_key]["raw_output"]).get("report", {}).get("recommendations", [])
        if recommendations:
            for recommendation in recommendations:
                st.write(f"- **{recommendation}**")
        else:
            st.write("No recommendations available.")
    else:
        st.write("No recommendations available.")

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
