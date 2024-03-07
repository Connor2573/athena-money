import requests
from datetime import datetime, timedelta
from newspaper import Article, ArticleException
import pandas as pd
from tqdm import tqdm
import os
import glob

def extract_content(url):
    article = Article(url)
    try:
        article.download()
        article.parse()
        return article.text
    except ArticleException:
        print(f"Failed to download article: {url}")
        return ""
    
def classify_text(text, pipe):
    # Split the text into chunks of 512 tokens or less
    chunks = [text[i:i + 512] for i in range(0, len(text), 512)]

    # Classify each chunk and sum the results
    positive_score = 0
    neutral_score = 0
    negative_score = 0
    for chunk in chunks:
        sentiment = pipe(chunk)[0]
        #print(sentiment)
        if sentiment['label'] == 'positive':
            positive_score += sentiment['score']
        elif sentiment['label'] == 'neutral':
            neutral_score += sentiment['score']
        else:
            negative_score += sentiment['score']

    # Average the scores
    positive_score /= len(chunks)
    neutral_score /= len(chunks)
    negative_score /= len(chunks)

    # Return the average sentiment scores
    return positive_score, neutral_score, negative_score

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-classification", model="ProsusAI/finbert")

csv_files = glob.glob('./data/news/*.csv')

for file in csv_files:
# Read the CSV file
    df = pd.read_csv(file)

    # Extract the dates
    dates = df['published_utc'].tolist()

    # Initialize an empty DataFrame
    processed_data_df = pd.DataFrame(columns=['Date', 'Title1', 'Pos1', 'Neu1', 'Neg1', 'Title2', 'Pos2', 'Neu2', 'Neg2', 'Title3', 'Pos3', 'Neu3', 'Neg3'])
    #processed_data_df = pd.DataFrame(columns=['Date', 'Title1', 'Label1', 'Score1', 'Title2', 'Label2', 'Score2', 'Title3', 'Label3', 'Score3'])

    pbar = tqdm(dates)
    done_dates = set()
    for date in pbar:
        date = date.split('T')[0]  # Remove the time portion
        utc_time = datetime.strptime(date, '%Y-%m-%d')
        date = utc_time.strftime('%Y-%m-%d')
        
        if date in done_dates:
            continue
        else:
            done_dates.add(date)
        
        top_stories = df[df['published_utc'].str.contains(date)]
        row = [date]
        for index, story in top_stories.iterrows():
            content = story['title']#extract_content(story['url'])
            if content and type(content) == str:
                pos, neu, neg = classify_text(content, pipe)
                #print(f"Positive Score: {pos:.2f}, Neutral Score: {neu:.2f}, Negative Score: {neg:.2f}")
                row.extend([story['title'], pos, neu, neg])
                #result = pipe(content)[0]
                #label = result['label']
                #score = result['score']
                #row.extend([story['title'], label, score])
            else:
                print("No content available")
        while len(row) < 13:
            row.extend(['', 0, 0, 0])
        row = pd.Series(row, index=processed_data_df.columns)
        processed_data_df = processed_data_df._append(row, ignore_index=True)
        pbar.set_description(f"File: {file}, Date: {date}")
                
    # Write the DataFrame to a CSV file
    processed_data_df.to_csv(file.replace('news', 'processed_news'), index=False)