import os
import time
import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
import openai

# Load environment and set up Streamlit Secrets for API keys
load_dotenv()

# Directly set environment variables and API keys from Streamlit secrets
os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Initialize SerperDevTool for search and set up LLM selection
search_tool = SerperDevTool()

def create_llm(use_gpt=True):
    return ChatOpenAI(model="gpt-4o-mini") if use_gpt else Ollama(model="llama3.1")

def create_agents(topic_or_brand, llm):
    research_specialist = Agent(
        role="Research Specialist",
        goal=f"Conduct thorough research on {topic_or_brand}",
        backstory="Experienced researcher with expertise in finding and synthesizing information.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
        max_iter=15
    )

    social_media_monitor = Agent(
        role="Social Media Monitor",
        goal=f"Monitor social media platforms for mentions of {topic_or_brand}",
        backstory="Experienced social media analyst with keen eyes for trends and mentions.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
        max_iter=15
    )

    sentiment_analyzer = Agent(
        role="Sentiment Analyzer",
        goal=f"Analyze the sentiment of social media mentions about {topic_or_brand}",
        backstory="Expert in natural language processing and sentiment analysis.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=15
    )

    report_generator = Agent(
        role="Report Generator",
        goal=f"Generate a comprehensive report on {topic_or_brand} covering research, social media, and sentiment analysis.",
        backstory="Skilled data analyst and report writer, adept at presenting insights clearly.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=15
    )

    return [research_specialist, social_media_monitor, sentiment_analyzer, report_generator]

def create_integrated_tasks(topic_or_brand, agents):
    research_task = Task(
        description=f"Research {topic_or_brand} and provide a summary.",
        agent=agents[0],
        expected_output="Background research with key points and insights on {topic_or_brand}."
    )

    monitoring_task = Task(
        description=f"Monitor social media for mentions of '{topic_or_brand}'",
        agent=agents[1],
        expected_output="Summary of recent social media mentions and activity for {topic_or_brand}."
    )

    sentiment_analysis_task = Task(
        description=f"Analyze sentiment of the social media mentions for {topic_or_brand}",
        agent=agents[2],
        expected_output="Sentiment analysis report with positive, negative, and neutral breakdown."
    )

    report_generation_task = Task(
        description=f"Generate a comprehensive report for {topic_or_brand}",
        agent=agents[3],
        expected_output="Structured report including research, social media analysis, sentiment, and insights."
    )

    return [research_task, monitoring_task, sentiment_analysis_task, report_generation_task]

def run_integrated_analysis(topic_or_brand, use_gpt=True, max_retries=3):
    llm = create_llm(use_gpt)
    agents = create_agents(topic_or_brand, llm)
    tasks = create_integrated_tasks(topic_or_brand, agents)
    
    crew = Crew(agents=agents, tasks=tasks, verbose=True)

    for attempt in range(max_retries):
        try:
            result = crew.kickoff()
            return result
        except Exception as e:
            st.write(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                st.write("Retrying...")
                time.sleep(5)
            else:
                st.write("Max retries reached. Unable to complete the task.")
                return None

# Streamlit UI Setup
st.title("Unified Brand and Topic Analysis")
st.write("Analyze a brand or topic with a cohesive approach, integrating research, social media monitoring, sentiment analysis, and comprehensive reporting.")

# User input for brand or topic and model selection
topic_or_brand = st.text_input("Enter the Brand or Topic Name")
use_gpt = st.checkbox("Use GPT-4o-mini model (uncheck for Llama)")

# Run the integrated analysis on button click
if st.button("Start Integrated Analysis"):
    if topic_or_brand:
        result = run_integrated_analysis(topic_or_brand, use_gpt)
        if result:
            st.write("Final Comprehensive Report:")
            st.write(result)
        else:
            st.error("Failed to generate the report. Please try again later.")
    else:
        st.error("Please enter a brand or topic name to proceed.")
