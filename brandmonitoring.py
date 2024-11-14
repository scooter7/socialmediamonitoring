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
import json
import re

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

# Parse tool output to extract structured data
def parse_tool_output(tool_output):
    entries = re.findall(r"Title: (.+?)\nLink: (.+?)\nSnippet: (.+?)(?=\n---|\Z)", tool_output, re.DOTALL)
    return [{"title": title.strip(), "link": link.strip(), "snippet": snippet.strip()} for title, link, snippet in entries]

# Create agents with CrewAI for research and analysis
def create_agents(brand_name, llm):
    researcher = Agent(
        role="Social Media Researcher",
        goal=f"Research and gather information about {brand_name} from various sources.",
        backstory="You are an experienced social media researcher, skilled at quickly gathering accurate information and identifying notable trends across multiple online platforms.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
        max_iter=15
    )

    social_media_monitor = Agent(
        role="Social Media Monitor",
        goal=f"Monitor social media platforms for mentions of {brand_name}.",
        backstory="As a seasoned social media analyst, you excel at tracking online mentions, uncovering sentiment trends, and recognizing emerging discussions.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
        max_iter=15
    )

    sentiment_analyzer = Agent(
        role="Sentiment Analyzer",
        goal=f"Analyze the sentiment of social media mentions about {brand_name}.",
        backstory="With a strong background in natural language processing, you are skilled at evaluating sentiment and interpreting user opinions to deliver a comprehensive sentiment distribution.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=15
    )

    report_generator = Agent(
        role="Report Generator",
        goal=f"Generate comprehensive reports based on the analysis of {brand_name}.",
        backstory="As an accomplished data analyst, you excel at synthesizing insights into structured, clear reports with actionable recommendations.",
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

# Enhanced display_formatted_report function
def display_formatted_report(brand_name, result):
    st.header(f"Online and Sentiment Analysis Report for {brand_name}")
    st.write("---")

    # Extract task outputs
    task_outputs = result.tasks_output

    # Section 1: Research Findings
    st.subheader("1. Research Findings")
    research_output = task_outputs[0].summary if task_outputs[0] else "No research data available."
    st.write(research_output)

    # Section 2: Online Mentions
    st.subheader("2. Online Mentions")
    mentions_output = task_outputs[1].summary if task_outputs[1] else "No online mentions data available."
    st.write(mentions_output)

    # Section 3: Sentiment Analysis
    st.subheader("3. Sentiment Analysis")
    sentiment_output = task_outputs[2].summary if task_outputs[2] else "No sentiment data available."
    st.write(sentiment_output)

    # Section 4: Key Themes and Recommendations
    st.subheader("4. Key Themes and Recommendations")
    report_output = task_outputs[3].raw if task_outputs[3] else "No report data available."

    try:
        # Remove JSON formatting markers if present
        report_output = report_output.strip("```json\n").strip("\n```")
        report_data = json.loads(report_output).get("report", {})

        # Sentiment Distribution
        st.write("### Sentiment Distribution")
        sentiment_distribution = report_data.get("sentiment_distribution", {})
        for sentiment, details in sentiment_distribution.items():
            st.write(f"**{sentiment.capitalize()} Mentions**")
            st.write(f"- Count: {details.get('mentions', 'N/A')}")
            st.write(f"- Percentage: {details.get('percentage', 'N/A')}%")
            for insight in details.get("insights", []):
                st.write(f"  - {insight}")

        # Notable Themes
        st.write("### Notable Themes")
        notable_themes = report_data.get("notable_themes", [])
        if notable_themes:
            for theme in notable_themes:
                st.write(f"- **{theme.get('theme', 'Unnamed Theme')}**: {theme.get('description', 'No description available')}")
        else:
            st.write("No notable themes available.")

        # Notable Posts
        st.write("### Notable Posts")
        notable_posts = report_data.get("notable_posts", [])
        if notable_posts:
            for post in notable_posts:
                title = post.get('title', 'No Title')
                link = post.get('link', '#')
                snippet = post.get('snippet', 'No snippet available')
                st.write(f"- **[{title}]({link})**: {snippet}")
        else:
            st.write("No notable posts available.")

        # Recommendations
        st.write("### Recommendations")
        recommendations = report_data.get("recommendations", [])
        if recommendations:
            for rec in recommendations:
                st.write(f"- {rec}")
        else:
            st.write("No recommendations available.")

    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        st.error("Error parsing the JSON-formatted report. Please ensure the data format is correct.")
        st.write(f"Details: {str(e)}")  # Optional: Show error details for debugging

# Streamlit app interface
st.title("Online and Sentiment Analysis Report")
st.write("Analyze a brand or topic with integrated online monitoring, sentiment analysis, and report generation.")

# User input for brand or topic
brand_name = st.text_input("Enter the Brand or Topic Name")

# Run the analysis on button click
if st.button("Start Analysis"):
    if brand_name:
        st.write("Starting online monitoring and sentiment analysis...")
        result = run_social_media_monitoring(brand_name)
        
        if result:
            display_formatted_report(brand_name, result)
        else:
            st.error("Failed to generate the report. Please try again.")
    else:
        st.error("Please enter a brand or topic name to proceed.")
