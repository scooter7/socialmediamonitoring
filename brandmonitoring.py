import os
import streamlit as st
import openai
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
import requests
import json
import base64

# Load environment and set up Streamlit Secrets for API keys
load_dotenv()

# Set environment variables directly from Streamlit secrets
os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Initialize search tool
search_tool = SerperDevTool()

# Function to create LLM using GPT-4o-mini only
def create_llm(use_gpt=True):
    return ChatOpenAI(model="gpt-4o-mini")

# Function to create agents for social media monitoring and sentiment analysis
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

    sentiment_analyzer = Agent(
        role="Sentiment Analyzer",
        goal=f"Analyze the sentiment of social media mentions about {brand_name}",
        backstory="You are an expert in natural language processing and sentiment analysis.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=15
    )

    return [researcher, sentiment_analyzer]

# Function to create tasks for agents
def create_tasks(brand_name, agents):
    research_task = Task(
        description=f"Research {brand_name} and provide a summary of recent social media activity.",
        agent=agents[0],
        expected_output="A summary containing recent mentions, platform details, and any notable hashtags or posts."
    )

    sentiment_analysis_task = Task(
        description=f"Analyze the sentiment of the social media mentions about {brand_name}.",
        agent=agents[1],
        expected_output="A report categorizing mentions as positive, negative, or neutral, with key themes."
    )

    return [research_task, sentiment_analysis_task]

# Function to run social media monitoring and sentiment analysis using CrewAI
def run_social_media_monitoring(brand_name, use_gpt=True):
    llm = create_llm(use_gpt)
    agents = create_agents(brand_name, llm)
    tasks = create_tasks(brand_name, agents)
    
    crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=True
    )

    try:
        result = crew.kickoff()
        return result
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

# Function to compile and save the report to a CSV file in GitHub
def generate_report(brand_name, result):
    # GitHub repository details
    repo_owner = "scooter7"  # Replace with your GitHub username
    repo_name = "socialmediamonitoring"       # Replace with your GitHub repository name
    file_path = "report.csv"     # Path in the repository where the file will be saved
    github_token = st.secrets["GITHUB_TOKEN"]  # Store your GitHub token in Streamlit secrets

    # Prepare CSV content
    csv_content = "Brand,Research Summary,Sentiment Analysis\n"
    csv_content += f'"{brand_name}","{result[0]}","{result[1]}"\n'
    encoded_content = base64.b64encode(csv_content.encode()).decode()

    # GitHub API URL to create or update a file
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}"
    
    # Prepare the request payload
    payload = {
        "message": "Add report entry",
        "content": encoded_content,
        "branch": "main"  # Update if you want to save to a different branch
    }

    # Make a GET request to check if the file already exists to get the `sha`
    response = requests.get(url, headers={"Authorization": f"token {github_token}"})
    if response.status_code == 200:
        # File exists, add 'sha' to payload for update
        payload["sha"] = response.json()["sha"]

    # Make the PUT request to create or update the file
    response = requests.put(url, headers={"Authorization": f"token {github_token}"}, data=json.dumps(payload))

    if response.status_code in [200, 201]:
        st.success("Report generated and saved to GitHub.")
    else:
        st.error(f"Failed to save report to GitHub. Status code: {response.status_code}")
        st.error(f"Response message: {response.json().get('message', 'No message available')}")

# Streamlit UI Setup
st.title("Social Media Monitoring and Sentiment Analysis with CrewAI")
st.write("Analyze a brand or topic with integrated social media monitoring, sentiment analysis, and report generation.")

# User input for brand or topic and model selection
brand_name = st.text_input("Enter the Brand or Topic Name")
use_gpt = st.checkbox("Use GPT-4o-mini model")

# Run the analysis on button click
if st.button("Start Analysis"):
    if brand_name:
        st.write("Running social media monitoring and sentiment analysis...")
        result = run_social_media_monitoring(brand_name, use_gpt)
        
        if result:
            st.write("Research Summary:", result[0])
            st.write("Sentiment Analysis:", result[1])
            
            # Generate final report
            st.write("Generating Report...")
            generate_report(brand_name, result)
    else:
        st.error("Please enter a brand or topic name to proceed.")
