import json
import streamlit as st
import time
import os
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
        backstory="You are an expert researcher who quickly finds relevant information across various online sources.",
        verbose=True,
        tools=[search_tool],
        llm=llm,
        max_iter=15
    )

    social_media_monitor = Agent(
        role="Social Media Monitor",
        goal=f"Monitor social media platforms for mentions of {brand_name}.",
        backstory="You are an experienced social media analyst with a focus on trends, mentions, and sentiment analysis.",
        verbose=True,
        tools=[search_tool],
        llm=llm,
        max_iter=15
    )

    sentiment_analyzer = Agent(
        role="Sentiment Analyzer",
        goal=f"Analyze the sentiment of social media mentions about {brand_name}.",
        backstory="You are skilled in natural language processing and specialize in identifying sentiment trends.",
        verbose=True,
        llm=llm,
        max_iter=15
    )

    report_generator = Agent(
        role="Report Generator",
        goal=f"Generate comprehensive reports based on the analysis of {brand_name}.",
        backstory="You are a data analyst and report writer, skilled at presenting insights clearly and concisely.",
        verbose=True,
        llm=llm,
        max_iter=15
    )

    return [researcher, social_media_monitor, sentiment_analyzer, report_generator]

# Define tasks for each agent with CrewAI
def create_tasks(brand_name, agents):
    return [
        Task(
            description=f"Research {brand_name} and provide a summary of their online presence, key information, and recent activities.",
            agent=agents[0],
            expected_output="A structured summary with key insights on recent activities, platform presence, and notable mentions."
        ),
        Task(
            description=f"Monitor social media platforms for mentions of '{brand_name}'. Provide a summary of the mentions.",
            agent=agents[1],
            expected_output="Summary of mentions including counts, platforms, notable mentions, and hashtags."
        ),
        Task(
            description=f"Analyze the sentiment of the social media mentions about {brand_name}. Categorize them as positive, negative, or neutral.",
            agent=agents[2],
            expected_output="Sentiment distribution and notable themes."
        ),
        Task(
            description=f"Generate a JSON-formatted report for {brand_name} based on findings.",
            agent=agents[3],
            expected_output="Comprehensive report in JSON format including key insights and recommendations."
        )
    ]

# Run online monitoring and sentiment analysis workflow
def run_social_media_monitoring(brand_name, max_retries=3):
    llm = create_llm()
    agents = create_agents(brand_name, llm)
    tasks = create_tasks(brand_name, agents)
    crew = Crew(agents=agents, tasks=tasks, verbose=True)

    for attempt in range(max_retries):
        try:
            result = crew.kickoff()
            st.write("Debug: Raw result from Crew:", result)  # Debugging line to inspect raw result
            
            # Remove ```json and ``` from the result string
            cleaned_result = result.raw.replace("```json", "").replace("```", "").strip()
            
            # Attempt to parse JSON
            return json.loads(cleaned_result)
        except json.JSONDecodeError as e:
            st.error(f"JSON decoding error: {e}")
            st.write("Debug: Cleaned JSON string:", cleaned_result)
            if attempt < max_retries - 1:
                st.write("Retrying...")
                time.sleep(5)
            else:
                st.error("Max retries reached. Unable to complete the task.")
                return None

# Function to display mentions in Section 2
def display_mentions(mentions_data):
    if mentions_data:
        for mention in mentions_data:
            title = mention.get("title", "No title available")
            link = mention.get("link", "No link available")
            snippet = mention.get("snippet", "No snippet available")
            st.write(f"**Title:** {title}\n**Link:** {link}\n**Snippet:** {snippet}\n---")
    else:
        st.write("No online mentions or sentiment data available.")

# Function to parse and display the report content
def parse_and_display_report(report_output):
    try:
        # Directly use the parsed JSON data
        report_data = report_output

        # Display Online Mentions and Sentiment Analysis
        st.subheader("2. Online Mentions and Sentiment Analysis")
        mentions_data = report_data.get("tool_outputs", [])
        display_mentions(mentions_data)
    except Exception as e:
        st.error(f"Error displaying report: {e}")

# Main display function
def display_formatted_report(brand_name, report_output):
    st.header(f"Online and Sentiment Analysis Report for {brand_name}")
    st.write("---")

    st.subheader("1. Research Findings")
    st.write("Summary of recent activities, online presence, etc.")

    # 2. Online Mentions and Sentiment Analysis
    parse_and_display_report(report_output)

    st.subheader("3. Recommendations")
    recommendations = report_output.get("DMACC_Report", {}).get("Recommendations", "No recommendations available.")
    st.write(recommendations)

# Streamlit app interface
st.title("Online and Sentiment Analysis Report")
brand_name = st.text_input("Enter the Brand or Topic Name")

if st.button("Start Analysis"):
    if brand_name:
        st.write("Starting analysis...")
        result = run_social_media_monitoring(brand_name)
        if result:
            display_formatted_report(brand_name, result)
        else:
            st.error("Failed to generate the report.")
    else:
        st.error("Please enter a brand or topic name.")
