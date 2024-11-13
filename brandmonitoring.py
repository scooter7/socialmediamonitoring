import os
import time
import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
import openai
import matplotlib.pyplot as plt

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
        verbose=True,
        tools=[search_tool],
        llm=llm,
    )
    social_media_monitor = Agent(
        role="Social Media Monitor",
        goal=f"Monitor social media platforms for mentions of {brand_name}",
        verbose=True,
        tools=[search_tool],
        llm=llm,
    )
    sentiment_analyzer = Agent(
        role="Sentiment Analyzer",
        goal=f"Analyze the sentiment of social media mentions about {brand_name}",
        verbose=True,
        llm=llm,
    )
    return [researcher, social_media_monitor, sentiment_analyzer]

# Define tasks with crewai
def create_tasks(brand_name, agents):
    research_task = Task(
        description=f"Research {brand_name} and provide a summary of their online presence.",
        agent=agents[0],
        expected_output="Structured research summary with key insights on recent activities and notable mentions.",
    )
    monitoring_task = Task(
        description=f"Monitor social media platforms for mentions of '{brand_name}' and summarize the mentions.",
        agent=agents[1],
        expected_output="Summary of mentions, including counts and notable mentions.",
    )
    sentiment_analysis_task = Task(
        description=f"Analyze the sentiment of the social media mentions about {brand_name}.",
        agent=agents[2],
        expected_output="Sentiment distribution across platforms.",
    )
    return [research_task, monitoring_task, sentiment_analysis_task]

# Run social media monitoring and sentiment analysis workflow
def run_social_media_monitoring(brand_name, max_retries=3):
    llm = create_llm()
    agents = create_agents(brand_name, llm)
    tasks = create_tasks(brand_name, agents)
    
    crew = Crew(agents=agents, tasks=tasks, verbose=True)
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

# Generate report and present in a structured format
def present_results(results):
    st.markdown("### Analysis Report")
    if "research" in results:
        st.subheader("Research Summary")
        st.write(results["research"])
    if "monitoring" in results:
        st.subheader("Social Media Mentions Summary")
        st.write(results["monitoring"])
    if "sentiment_analysis" in results:
        st.subheader("Sentiment Analysis Summary")
        st.write(results["sentiment_analysis"])

# Plot sentiment analysis by platform
def plot_sentiment_by_platform(sentiment_data):
    platforms = [entry["platform"] for entry in sentiment_data]
    sentiments = ["Positive", "Neutral", "Negative"]
    sentiment_counts = {
        sentiment: [entry.get(sentiment.lower(), 0) for entry in sentiment_data]
        for sentiment in sentiments
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    width = 0.2
    for i, sentiment in enumerate(sentiments):
        ax.bar(
            [x + i * width for x in range(len(platforms))],
            sentiment_counts[sentiment],
            width,
            label=sentiment,
        )

    ax.set_xlabel("Social Media Platforms")
    ax.set_ylabel("Count of Mentions")
    ax.set_title("Sentiment Analysis by Platform")
    ax.set_xticks([x + width for x in range(len(platforms))])
    ax.set_xticklabels(platforms)
    ax.legend()
    st.pyplot(fig)

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
            st.write("Analysis completed successfully.")
            present_results(result)
            
            # Assume `result` contains sentiment data per platform in a structured way
            if "sentiment_data" in result:
                plot_sentiment_by_platform(result["sentiment_data"])
        else:
            st.error("Failed to generate the report. Please try again.")
    else:
        st.error("Please enter a brand or topic name to proceed.")
