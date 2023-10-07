import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the VADER lexicon (required for sentiment analysis)
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Sample text for sentiment analysis
text = "I love this product! It's amazing."

# Analyze sentiment
sentiment_scores = sia.polarity_scores(text)

# Interpret the sentiment scores
compound_score = sentiment_scores['compound']

if compound_score >= 0.05:
    sentiment = "Positive"
elif compound_score <= -0.05:
    sentiment = "Negative"
else:
    sentiment = "Neutral"

# Print results
print("Sentiment Analysis Results:")
print(f"Text: {text}")
print(f"Compound Score: {compound_score}")
print(f"Sentiment: {sentiment}")
