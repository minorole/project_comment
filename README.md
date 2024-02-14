## README.md for YouTube Comment Analysis Project

### Overview
This project is designed to perform an in-depth analysis of YouTube comments. It covers various aspects, including sentiment analysis, topic modeling, engagement scoring, and more. The script utilizes the YouTube API to fetch comments and applies natural language processing techniques to derive insights.

### Prerequisites
- Python 3.6 or later
- Google API key with YouTube Data API v3 enabled
- Required Python libraries: `pandas`, `numpy`, `googleapiclient`, `nltk`, `gensim`, `torch`, `transformers`, `matplotlib`, `seaborn`, `wordcloud`, `pyLDAvis`

### Installation
1. Clone the repository to your local machine.
2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Obtain a Google API key and enable the YouTube Data API v3.

### Configuration
- **API Key**: Replace `api_key` in the `get_comments` function with your Google API key.
  ```python
  comments = get_comments(video_id='VIDEO_ID_HERE', api_key='YOUR_API_KEY_HERE', max_results=1000000)
  ```
- **Video ID**: Change `video_id` in the `get_comments` function to the ID of the YouTube video you want to analyze.
- **LDA Model Topics**: Adjust the `num_topics` parameter in the `gensim.models.LdaMulticore` function to control the number of topics for the LDA model.
  ```python
  lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=20, id2word=dictionary, passes=20, workers=20)
  ```

### Usage
Run the script using the following command:
```bash
python youtube_comment_analysis.py
```
The script will perform the following analyses:
- Sentiment analysis of comments
- Topic modeling using LDA
- Calculation of engagement scores
- Analysis of comment frequency over time
- Identification of top commenters based on engagement score
- Visualization of sentiment over time
- Generation of a word cloud for common terms

### Output
The script generates several output files, including:
- `comments_with_sentiment.csv`: Comments with their corresponding sentiment analysis.
- `topic_modeling_results.txt`: Results of the topic modeling analysis.
- `lda_visualization.html`: Interactive HTML visualization of the LDA model.
- Various plots in PNG format (e.g., `comment_frequency_over_time_line.png`, `top_engaging_comments.png`, `sentiment_over_time.png`, `sentiment_analysis_pie.png`).
- `top_10_engaging_comments.csv`: Top 10 engaging comments based on the engagement score.

### License
This project is licensed under the MIT License - see the LICENSE file for details.
