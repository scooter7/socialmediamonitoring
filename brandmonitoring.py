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
            # mentions[source] will now store parsed mentions directly
            mentions[source] = parse_tool_output(result) if result else []
        except Exception as e:
            st.warning(f"Could not retrieve data from {source}. Error: {e}")
            mentions[source] = []  # Store an empty list if an error occurs
    return mentions

# Parse tool output to extract structured data
def parse_tool_output(tool_output):
    # Adjust regex to capture mentions if the format has changed
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
            return result.tasks_output  # Return only the tasks output for display
        except Exception as e:
            st.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                st.write("Retrying...")
                time.sleep(5)
            else:
                st.error("Max retries reached. Unable to complete the task.")
                return None

# Display formatted report based on task outputs
def display_formatted_report(brand_name, task_outputs):
    st.header(f"Online and Sentiment Analysis Report for {brand_name}")
    st.write("---")

    # Section 1: Research Findings
    st.subheader("1. Research Findings")
    research_output = task_outputs[0].raw if task_outputs and len(task_outputs) > 0 else "No data available"
    st.write(research_output)

    # Section 2: Online Mentions
    st.subheader("2. Online Mentions")
    mentions_output = task_outputs[1].raw if task_outputs and len(task_outputs) > 1 else "No mentions data available"
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
    sentiment_output = task_outputs[2].raw if task_outputs and len(task_outputs) > 2 else "No sentiment data available"
    st.write(sentiment_output)

    # Section 4: Key Themes and Recommendations
    st.subheader("4. Key Themes and Recommendations")
    report_output = task_outputs[3].raw if task_outputs and len(task_outputs) > 3 else "No report data available"

    # Display raw output for debugging purposes
    st.write("**Raw Report Output**", report_output)

    try:
        # Clean JSON output and parse it
        report_output_cleaned = report_output.strip('```json\n').strip('\n```')
        report_data = json.loads(report_output_cleaned)

        # Display structured information if available
        st.write("**Sentiment Distribution**")
        sentiment_distribution = report_data.get("sentiment_distribution", {})
        if sentiment_distribution:
            st.write(f"- Positive Mentions: {sentiment_distribution.get('positive', {}).get('percentage', 'Data unavailable')}%")
            st.write(f"- Negative Mentions: {sentiment_distribution.get('negative', {}).get('percentage', 'Data unavailable')}%")
            st.write(f"- Neutral Mentions: {sentiment_distribution.get('neutral', {}).get('percentage', 'Data unavailable')}%")

            # Display insights for each sentiment type
            for sentiment, data in sentiment_distribution.items():
                st.write(f"**{sentiment.capitalize()} Mentions**")
                for insight in data.get("key_insights", []):
                    st.write(f"  - {insight}")

        else:
            st.write("Sentiment distribution data is unavailable.")

        st.write("**Notable Themes**")
        notable_themes = report_data.get("notable_themes", [])
        for theme in notable_themes:
            st.write(f"- **{theme.get('theme', 'Unnamed Theme')}**")
            st.write(f"  - Description: {theme.get('description', 'No description available')}")
            if theme.get("hashtags"):
                st.write(f"  - Hashtags: {', '.join(theme['hashtags'])}")

        st.write("**Conclusion**")
        conclusion = report_data.get("conclusion", {})
        st.write(f"- Summary: {conclusion.get('summary', 'No summary available')}")
        st.write(f"- Polarized View: {conclusion.get('polarized_view', 'No implications available')}")

        st.write("**Recommendations**")
        recommendations = report_data.get("recommendations", [])
        if recommendations:
            for recommendation in recommendations:
                st.write(f"- {recommendation}")
        else:
            st.write("No recommendations available.")

    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        st.error(f"Error parsing the JSON-formatted report: {e}")

# Streamlit app interface
st.title("Online and Sentiment Analysis Report")
st.write("Analyze a brand or topic with integrated online monitoring, sentiment analysis, and report generation.")

# User input for brand or topic
brand_name = st.text_input("Enter the Brand or Topic Name")

# Run the analysis on button click
if st.button("Start Analysis"):
    if brand_name:
        st.write("Starting online monitoring and sentiment analysis...")
        task_outputs = run_social_media_monitoring(brand_name)
        
        if task_outputs:
            display_formatted_report(brand_name, task_outputs)
        else:
            st.error("Failed to generate the report. Please try again.")
    else:
        st.error("Please enter a brand or topic name to proceed.")

