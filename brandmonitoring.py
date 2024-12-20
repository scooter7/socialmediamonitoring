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

st.markdown(
    """
    <style>
    .st-emotion-cache-12fmjuu.ezrtsby2 {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 20px;
    }
    .logo-container img {
        width: 300px;
    }
    .app-container {
        border-left: 5px solid #58258b;
        border-right: 5px solid #58258b;
        padding-left: 15px;
        padding-right: 15px;
    }
    .stTextArea, .stTextInput, .stMultiSelect, .stSlider {
        color: #42145f;
    }
    .stButton button {
        background-color: #fec923;
        color: #42145f;
    }
    .stButton button:hover {
        background-color: #42145f;
        color: #fec923;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="logo-container">
        <img src="https://www.carnegiehighered.com/wp-content/uploads/2021/11/Twitter-Image-2-2021.png" alt="Logo">
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="app-container">', unsafe_allow_html=True)

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

# Capture and concatenate raw tool output
from datetime import datetime, timedelta

# Capture and concatenate raw tool output with date filtering
from datetime import datetime, timedelta

# Capture and concatenate raw tool output with date filtering
def fetch_mentions(brand_name):
    try:
        # Fetch raw tool output
        result = search_tool.search(brand_name)

        # Define two months ago as the cutoff date
        cutoff_date = datetime.now() - timedelta(days=60)

        # Extract and filter results by date if the output contains dates
        filtered_result = []
        for entry in result.split("\n---\n"):
            # Extract potential date using regex or text parsing
            date_match = re.search(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b \d{1,2}, \d{4}", entry)
            if date_match:
                # Parse the date and compare it with the cutoff date
                mention_date = datetime.strptime(date_match.group(), "%b %d, %Y")
                if mention_date >= cutoff_date:
                    filtered_result.append(entry)
        
        # Return filtered results or a message if none are valid
        return "\n---\n".join(filtered_result) if filtered_result else "No mentions from the last two months."
    except Exception as e:
        st.warning(f"Error fetching mentions: {e}")
        return ""

# Function to parse tool output for structured data
def parse_tool_output(tool_output):
    st.write("Debug - Raw Tool Output in parse_tool_output:", tool_output)  # Debugging log
    
    # Extract structured entries with regex
    entries = re.findall(r"Title: (.+?)\nLink: (.+?)\nSnippet: (.+?)(?=\n---|\Z)", tool_output, re.DOTALL)
    
    # Return structured results
    return [{"title": title.strip(), "link": link.strip(), "snippet": snippet.strip()} for title, link, snippet in entries]

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
        goal=f"Monitor social media platforms for mentions of {brand_name} and retain raw tool output (title, link, snippet).",
        backstory="You are an experienced social media analyst tasked with extracting exact mentions (verbatim tool output) for the report.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=create_llm(),
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
        description=f"Monitor social media platforms for mentions of '{brand_name}' and provide verbatim tool outputs (title, link, snippet) in the report.",
        agent=agents[1],
        expected_output="Verbatim tool outputs showing title, link, and snippet for mentions of the brand."
    )

    sentiment_analysis_task = Task(
        description=f"Analyze the sentiment of the social media mentions about {brand_name}. Categorize them as positive, negative, or neutral, and provide key insights based on observed themes.",
        agent=agents[2],
        expected_output="Detailed sentiment distribution and key insights into themes such as community engagement, student life, or reputation."
    )

    report_generation_task = Task(
        description=f"Generate a JSON-formatted report for {brand_name} based on findings, including a section for recommendations to improve sentiment and engagement.",
        agent=agents[3],
        expected_output="Comprehensive report in JSON format including key insights and actionable recommendations based on sentiment analysis and observed themes."
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

# Display the report, showing exact tool output for mentions
def display_formatted_report(brand_name, result):
    st.header(f"Online and Sentiment Analysis Report for {brand_name}")
    st.write("---")

    # Section 1: Research Findings
    st.subheader("1. Research Findings")
    research_output = result.tasks_output[0].raw if result.tasks_output[0] else "No data available"
    st.write(research_output)

    # Section 2: Online Mentions
    st.subheader("2. Online Mentions")
    mentions_output = result.tasks_output[1].raw if result.tasks_output[1] else "No mentions data available"

    if mentions_output:
        st.write("## Verbatim Mentions:")

        # Parse the tool output to extract structured mentions
        parsed_mentions = parse_tool_output(mentions_output)

        # Display each mention in markdown format
        if parsed_mentions:
            for mention in parsed_mentions:
                st.markdown(
                    f"**Title:** [{mention['title']}]({mention['link']})\n\n"
                    f"**Snippet:** {mention['snippet']}\n\n---"
                )
            
            # Include a summary of the mentions after displaying verbatim mentions
            st.write("## Summary of Mentions:")
            summarize_mentions(parsed_mentions)
        else:
            st.write("No structured mentions available from tool output.")
    else:
        st.write("No online mentions available.")

    # Section 3: Sentiment Analysis
    st.subheader("3. Sentiment Analysis")
    sentiment_output = result.tasks_output[2].raw if result.tasks_output[2] else "No sentiment data available"
    st.write(sentiment_output)


# Function to summarize mentions
def summarize_mentions(parsed_mentions):
    platforms = {}
    notable_mentions = []

    for mention in parsed_mentions:
        # Extract domain to identify platforms
        domain = mention['link'].split('/')[2].replace('www.', '')
        if domain not in platforms:
            platforms[domain] = 0
        platforms[domain] += 1

        # Collect notable mentions
        notable_mentions.append(f"- **{mention['title']}**: {mention['snippet']}")

    # Summarize platforms
    st.write("### Platforms Mentioned:")
    for platform, count in platforms.items():
        st.markdown(f"- **{platform}**: {count} mention(s)")

    # Summarize notable mentions
    st.write("### Notable Mentions:")
    for note in notable_mentions:
        st.markdown(note)

# Streamlit app interface
st.title("Online Presence Monitor")
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
