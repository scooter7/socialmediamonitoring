import streamlit as st
import csv
import pandas as pd
from twitterscraper import scrape_twitter
from redditscraper import scrape_reddit
from quorascraper import scrape_quora
from facebookscraper import scrape_facebook
from sentiment import analyze_sentiment
import os

# File path for storing scraped data
DATA_FILE = "scraped_data.csv"

# Function to initialize CSV file with headers if it doesn't exist
def initialize_csv():
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Platform", "Data"])

# Function to save data to CSV file
def save_to_csv(platform, data):
    with open(DATA_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        for item in data:
            writer.writerow([platform, item])

# Display scraped data from CSV file
def display_csv_data():
    if os.path.exists(DATA_FILE):
        data = pd.read_csv(DATA_FILE)
        st.write(data)
    else:
        st.write("No data available yet.")

# Initialize the CSV file
initialize_csv()

# Set up the app layout
st.sidebar.title("Navigation")
options = st.sidebar.radio("Choose a section", ["Twitter", "Reddit", "Quora", "Facebook", "Sentiment Analysis", "View Saved Data"])

# Twitter Scraper Section
if options == "Twitter":
    st.title("Twitter Scraper")
    keyword = st.text_input("Enter keyword to search on Twitter:")
    if st.button("Scrape Twitter"):
        data = scrape_twitter(keyword)
        st.write(data)  # Display scraped data
        save_to_csv("Twitter", data)  # Save to CSV file

# Reddit Scraper Section
elif options == "Reddit":
    st.title("Reddit Scraper")
    subreddit = st.text_input("Enter subreddit to scrape:")
    if st.button("Scrape Reddit"):
        data = scrape_reddit(subreddit)
        st.write(data)  # Display scraped data
        save_to_csv("Reddit", data)  # Save to CSV file

# Quora Scraper Section
elif options == "Quora":
    st.title("Quora Scraper")
    topic = st.text_input("Enter topic to scrape on Quora:")
    if st.button("Scrape Quora"):
        data = scrape_quora(topic)
        st.write(data)  # Display scraped data
        save_to_csv("Quora", data)  # Save to CSV file

# Facebook Scraper Section
elif options == "Facebook":
    st.title("Facebook Scraper")
    page_name = st.text_input("Enter Facebook page name:")
    if st.button("Scrape Facebook"):
        data = scrape_facebook(page_name)
        st.write(data)  # Display scraped data
        save_to_csv("Facebook", data)  # Save to CSV file

# Sentiment Analysis Section
elif options == "Sentiment Analysis":
    st.title("Sentiment Analysis")
    text = st.text_area("Enter text for sentiment analysis:")
    if st.button("Analyze Sentiment"):
        result = analyze_sentiment(text)
        st.write(result)  # Display sentiment analysis result

# View Saved Data Section
elif options == "View Saved Data":
    st.title("Saved Scraped Data")
    display_csv_data()
