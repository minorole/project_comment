# Basic data handling and processing
import os
# Set environment variable to avoid parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np  # Added for numerical operations not covered by pandas
import logging

# Google API
import googleapiclient.discovery
from googleapiclient.errors import HttpError

# File and system operations
import shutil
from datetime import datetime
import gc  # Garbage Collector interface

# Text and natural language processing
import re
import emoji
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora, models
import gensim
from gensim.utils import simple_preprocess

# Machine learning and deep learning
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pyLDAvis.gensim

'''# Attempt to dynamically determine the correct pyLDAvis module for gensim compatibility
try:
    import pyLDAvis.gensim_models as gensimvis
    logging.info("Using pyLDAvis.gensim_models for compatibility with newer gensim versions.")
except ImportError:
    import pyLDAvis.gensim as gensimvis
    logging.warning("Falling back to pyLDAvis.gensim, consider updating pyLDAvis if using newer gensim versions.")
'''

# Visualization setup
plt.style.use("dark_background")
sns.set_palette("deep")

# Streamlining comment fetching with a generator to minimize memory usage
def get_comments(video_id, api_key, max_results=1000000):
    youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=api_key)
    next_page_token = None
    count = 0

    while count < max_results:
        try:
            request = youtube.commentThreads().list(
                part='snippet,replies',
                videoId=video_id,
                maxResults=100, # youtube API max is 100, don't increase 
                pageToken=next_page_token,
                textFormat='plainText'
            )
            response = request.execute()

            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                reply_count = item['snippet']['totalReplyCount']
                yield [
                    comment['authorDisplayName'],
                    comment['textDisplay'],
                    comment['likeCount'],
                    comment['publishedAt'],
                    reply_count,
                ]

                count += 1
                if count >= max_results:
                    break

            if 'nextPageToken' in response:
                next_page_token = response['nextPageToken']
            else:
                break
        except HttpError as e:
            logging.error(f"An HTTP error {e.resp.status} occurred:\n{e.content}")
            break

# Function to clean comment text
def clean_comment(comment):
    comment = re.sub(r'[^\x00-\x7F]+', ' ', comment)
    comment = emoji.demojize(comment)
    return comment

# Using partial function for batch processing to reduce memory footprint
def process_batch(batch, tokenizer, model):
    batch = [clean_comment(comment) for comment in batch]
    inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits.softmax(1)
    sentiments = probs.argmax(dim=1).tolist()
    label_map = {0: 'Negative', 1: 'Somewhat Negative', 2: 'Neutral', 3: 'Somewhat Positive', 4: 'Positive'}
    return [label_map[s] for s in sentiments]

# Function to process DataFrame in chunks and optimize memory usage
def process_dataframe_in_chunks(df, column_name='Comment', chunksize=100):
    model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    results = []
    for start_row in range(0, len(df), chunksize):
        end_row = min(start_row + chunksize, len(df))
        chunk = df.iloc[start_row:end_row]
        sentiments = process_batch(chunk[column_name].tolist(), tokenizer, model)
        results.extend(sentiments)

    df['Sentiment'] = results
    return df

# Function to create a Word Cloud
def create_word_cloud(text):
    wordcloud = WordCloud(background_color='black', width=800, height=400).generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig('word_cloud.png')
    plt.close()

# Preprocess text for topic modeling
def preprocess_text(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(token)
    return result
def create_bow_corpus(processed_docs):
    dictionary = gensim.corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=20, no_above=0.3, keep_n=100000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    return bow_corpus, dictionary

if __name__ == '__main__':
    comments = get_comments(video_id='7ESeQBeikKs', api_key='AIzaSyAKG1n4lPh2lY9KB15drpHOlnBBdmUqJnQ', max_results=1000000)
    df = pd.DataFrame(comments, columns=['Author', 'Comment', 'Likes', 'Published At', 'Replies'])
    # Processing comments for sentiment without loading the entire dataset into memory
    df_result = process_dataframe_in_chunks(df)
    df_result.to_csv('comments_with_sentiment.csv', index=False)
   
    # Analysis 1: Topic modeling
    processed_docs = df['Comment'].map(preprocess_text)
    bow_corpus, dictionary = create_bow_corpus(processed_docs)
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=20, id2word=dictionary, passes=20, workers=20)
     # Save topic modeling results to a text file
    with open('topic_modeling_results.txt', 'w') as file:
        for idx, topic in lda_model.print_topics(-1):
            file.write('Topic: {} \nWords: {}\n\n'.format(idx, topic))
    
    # Save LDA visualization to an HTML file
    vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dictionary)
    pyLDAvis.save_html(vis, 'lda_visualization.html')

    # Converting 'Published At' to datetime
    df['Published At'] = pd.to_datetime(df['Published At'])
    df['Published Date'] = df['Published At'].dt.date

    # Add engagement score for analysis, weighing more towards replies
    df['Engagement Score'] = df['Likes'] + (df['Replies'] * 2)

    # Analysis 2: Sentiment Analysis
    sentiment_count = df['Sentiment'].value_counts()

    # Analysis 3: Comment Frequency Over Time
    comments_over_time = df.groupby('Published Date').size()

    # Prepare data for visualization
    plt.figure(figsize=(15, 10))
    comments_over_time.plot(kind='line', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Number of Comments')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('comment_frequency_over_time_line.png')
    plt.close()

    # Analysis 4: Top Commenters based on engagement score and Their Impact
    top_engaging_comments = df.sort_values(by='Engagement Score', ascending=False).head(10)

    # Combine Author, Likes, and Replies into a single string for each of the top engaging comments
    top_engaging_comments['Label'] = top_engaging_comments.apply(lambda x: f"{x['Author']} (Likes: {x['Likes']}, Replies: {x['Replies']})", axis=1)

    # Visualization for Top Commenters
    plt.figure(figsize=(15, 10))
    ordered_data = top_engaging_comments.sort_values(by='Engagement Score', ascending=True)
    sns.barplot(x='Engagement Score', y='Label', data=ordered_data, palette='Blues_r')
    plt.title('Top 10 Engaging Comments with Author, Likes, and Replies')
    plt.xlabel('Engagement Score')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('top_engaging_comments.png')
    plt.close()

    # Analysis 5: Sentiment over time
    sentiment_over_time = df.groupby(['Published Date', 'Sentiment']).size().unstack(fill_value=0)

    plt.figure(figsize=(15, 10))
    for sentiment in sentiment_over_time.columns:
        plt.plot(sentiment_over_time.index, sentiment_over_time[sentiment], label=sentiment)
    plt.title('Sentiment Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Comments')
    plt.xticks(rotation=45)
    plt.legend(title='Sentiment')
    plt.tight_layout()
    plt.savefig('sentiment_over_time.png')
    plt.close()

    # Analysis 6: Word Cloud for Common Terms
    comment_text = " ".join(comment for comment in df['Comment'] if isinstance(comment, str))
    create_word_cloud(comment_text)

    # Saving top 10 comments to a file
    top_engaging_comments[['Author', 'Comment', 'Likes', 'Replies', 'Engagement Score']].to_csv('top_10_engaging_comments.csv', index=False)

    # Sentiment Analysis Chart (Pie Chart)
    plt.figure(figsize=(15, 10))
    colors = sns.color_palette("deep", len(sentiment_count))
    plt.pie(sentiment_count, labels=sentiment_count.index, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('sentiment_analysis_pie.png')
    plt.close()
