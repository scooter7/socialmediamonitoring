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

# Enhanced function to fetch online mentions with error handling
def fetch_mentions(brand_name):
    sources = ["Twitter", "Facebook", "Reddit", "Quora", "News"]
    mentions = {}
    for source in sources:
        try:
            result = search_tool.search(brand_name)
            mentions[source] = parse_tool_output(result) if result else []
        except Exception as e:
            st.warning(f"Could not retrieve data from {source}. Error: {e}")
            mentions[source] = []  # Store an empty list if an error occurs
    return mentions

# Parse tool output to extract structured data
def parse_tool_output(tool_output):
    entries = re.findall(r"Title: (.+?)\nLink: (.+?)\nSnippet: (.+?)(?=\n---|\Z)", tool_output, re.DOTALL)
    parsed_results = [{"title": title.strip(), "link": link.strip(), "snippet": snippet.strip()} for title, link, snippet in entries]
    return parsed_results

# Create agents with CrewAI for research and analysis
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

# Display formatted report based on task outputs
def display_formatted_report(brand_name, result):
    st.header(f"Online and Sentiment Analysis Report for {brand_name}")
    st.write("---")

    # Extract task outputs
    task_outputs = result.tasks_output

    # Section 1: Research Findings
    st.subheader("1. Research Findings")
    research_output = task_outputs[0].raw if task_outputs[0] else "No data available"
    st.write(research_output)

    # Section 2: Online Mentions
    st.subheader("2. Online Mentions")
    mentions_output = task_outputs[1].raw if task_outputs[1] else "No mentions data available"
    parsed_mentions = parse_tool_output(mentions_output)
    if parsed_mentions:
        for mention in parsed_mentions:
            st.markdown(f"**Title:** {mention['title']}")
            st.markdown(f"**Link:** [Read more]({mention['link']})")
            st.markdown(f"**Snippet:** {mention['snippet']}")
            st.markdown("---")
    else:
        st.write("No online mentions available.")

    # Section 3: Sentiment Analysis
    st.subheader("3. Sentiment Analysis")
    sentiment_output = task_outputs[2].raw if task_outputs[2] else "No sentiment data available"
    st.write(sentiment_output)

    # Section 4: Key Themes and Recommendations
    st.subheader("4. Key Themes and Recommendations")
    report_output = task_outputs[3].raw if task_outputs[3] else "No report data available"

    try:
        # Clean JSON output and parse it
        report_output_cleaned = re.sub(r'```(?:json)?\n|\n```', '', report_output)
        report_data = json.loads(report_output_cleaned)

        # Access the 'report' section in the JSON data
        report = report_data.get("report", {})

        # Sentiment Distribution
        st.write("**Sentiment Distribution**")
        sentiment_analysis = report.get("sentiment_analysis", {})
        st.write(f"- Positive Mentions: {sentiment_analysis.get('positive_mentions', 'N/A')}")
        st.write(f"- Neutral Mentions: {sentiment_analysis.get('neutral_mentions', 'N/A')}")
        st.write(f"- Negative Mentions: {sentiment_analysis.get('negative_mentions', 'N/A')}")
        st.write(f"- Overall Sentiment: {sentiment_analysis.get('overall_sentiment', 'N/A')}")

        # Key Insights
        st.write("**Key Insights**")
        key_insights = report.get("key_insights", {})
        if key_insights:
            for key, insight in key_insights.items():
                st.write(f"- **{key.replace('_', ' ').title()}**")
                st.write(f"  - Description: {insight.get('description', 'No description available')}")
                st.write(f"  - Feedback: {insight.get('feedback', 'No feedback available')}")
        else:
            st.write("No key insights available.")

        # Recommendations
        st.write("**Recommendations**")
        recommendations = report.get("recommendations", [])
        if recommendations:
            for recommendation in recommendations:
                st.write(f"- **{recommendation.get('action', 'No action specified')}**")
                st.write(f"  - {recommendation.get('details', 'No additional details provided')}")
        else:
            st.write("No recommendations available.")

    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        st.error("Error parsing the JSON-formatted report. Please check the JSON structure.")
        st.write("Debugging Information:")
        st.write(f"Raw report output: {report_output}")
        st.write(f"Cleaned report output: {report_output_cleaned}")
        st.write(f"Error details: {str(e)}")

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
