# Required Libraries
import sys
import pysqlite3 as sqlite3  # Forcing pysqlite3 as sqlite3
sys.modules["sqlite3"] = sqlite3

import os
import streamlit as st
import openai
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
import matplotlib.pyplot as plt
import pandas as pd

# Load environment and set up Streamlit Secrets for API keys
load_dotenv()
os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Set up the social media monitoring tool
search_tool = SerperDevTool()

# Function to create LLM using GPT-4o-mini
def create_llm():
    return ChatOpenAI(model="gpt-4o-mini")

# Function to create crew agents
def create_agents(brand_name, llm):
    researcher = Agent(
        role="Social Media Researcher",
        goal=f"Research and gather information about {brand_name} from various sources",
        tools=[search_tool],
        llm=llm,
        verbose=True,
        max_iter=10
    )

    social_media_monitor = Agent(
        role="Social Media Monitor",
        goal=f"Monitor social media platforms for mentions of {brand_name}",
        tools=[search_tool],
        llm=llm,
        verbose=True,
        max_iter=10
    )

    sentiment_analyzer = Agent(
        role="Sentiment Analyzer",
        goal=f"Analyze the sentiment of social media mentions about {brand_name}",
        llm=llm,
        verbose=True,
        max_iter=10
    )

    return [researcher, social_media_monitor, sentiment_analyzer]

# Function to create tasks for the agents
def create_tasks(brand_name, agents):
    research_task = Task(
        description=f"Research {brand_name} and provide a summary of online presence, platforms, and activities.",
        agent=agents[0],
        expected_output="Summary with key insights, online platforms, recent activities"
    )

    monitoring_task = Task(
        description=f"Monitor social media for mentions of '{brand_name}' in the last 24 hours.",
        agent=agents[1],
        expected_output="Platform-wise summary of social media mentions"
    )

    sentiment_analysis_task = Task(
        description=f"Analyze the sentiment of the social media mentions about {brand_name}. Classify them as Positive, Negative, or Neutral.",
        agent=agents[2],
        expected_output="Sentiment breakdown for each platform"
    )

    return [research_task, monitoring_task, sentiment_analysis_task]

# Run the social media monitoring tasks
def run_social_media_monitoring(brand_name):
    llm = create_llm()
    agents = create_agents(brand_name, llm)
    tasks = create_tasks(brand_name, agents)
    
    crew = Crew(agents=agents, tasks=tasks)
    result = crew.kickoff()
    return result

# Function to plot sentiment analysis
def plot_sentiment(sentiments, platform):
    sentiment_df = pd.DataFrame(list(sentiments.items()), columns=["Sentiment", "Count"])
    fig, ax = plt.subplots()
    ax.bar(sentiment_df["Sentiment"], sentiment_df["Count"])
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Mentions")
    plt.title(f"Sentiment Distribution on {platform}")
    st.pyplot(fig)

# Streamlit UI Setup
st.title("Social Media Monitoring and Sentiment Analysis")
st.write("Analyze a brand or topic with integrated social media monitoring, sentiment analysis, and report generation.")

# User input for brand or topic
topic_or_brand = st.text_input("Enter the Brand or Topic Name")

# Run the analysis on button click
if st.button("Start Analysis"):
    if topic_or_brand:
        st.write("Running social media monitoring and sentiment analysis...")
        
        # Run the crew tasks
        result = run_social_media_monitoring(topic_or_brand)

        # Extract results
        research_summary = result.get("tasks")[0]["output"]
        social_media_summary = result.get("tasks")[1]["output"]
        sentiment_data = result.get("tasks")[2]["output"]

        # Display Research Summary
        st.header("Research Summary")
        st.write(research_summary)

        # Display Social Media Summary
        st.header("Social Media Summary")
        st.write(social_media_summary)

        # Display and Plot Sentiment Analysis by Platform
        st.header("Sentiment Analysis by Platform")
        platform_sentiments = {"Twitter": {"Positive": 10, "Negative": 5, "Neutral": 3},  # Placeholder values
                               "Instagram": {"Positive": 8, "Negative": 2, "Neutral": 4},
                               "Facebook": {"Positive": 12, "Negative": 6, "Neutral": 1}}
        
        for platform, sentiments in platform_sentiments.items():
            st.subheader(f"{platform} Sentiment")
            plot_sentiment(sentiments, platform)

        st.success("Analysis Complete")
    else:
        st.error("Please enter a brand or topic name to proceed.")
