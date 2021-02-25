# twitter-stock-sentiment
This tool trains a NLTK Naive Bayes classifier then collects, cleans, and classifies the sentiments of live and trending tweets.

# Modes

## Top tweets

Collects the specified number of tweets for the specified ticker symbol to get the overall sentiment for the stock at the given moment.

## Stream

Uses Tweepy stream to collect and calculate the sentiments for live tweets for a specified ticker and plots them against a normalized price curve to see the relationship of twitters sentiment on the overall price of the stock over time. Could be used to find potential good times to invest.
