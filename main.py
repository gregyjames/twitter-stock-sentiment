# MIT License
#
# Copyright (c) 2021 Greg James
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# RESOURCES USED:
# https://towardsdatascience.com/text-normalization-for-natural-language-processing-nlp-70a314bfa646
# https://stackoverflow.com/questions/8376691/how-to-remove-hashtag-user-link-of-a-tweet-using-regular-expression
# https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
# https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range
# https://stackoverflow.com/questions/48966176/tweepy-truncated-tweets-when-using-tweet-mode-extended
# https://stackoverflow.com/questions/4270301/matplotlib-multiple-datasets-on-the-same-scatter-plot
# https://docs.tweepy.org/en/latest/streaming_how_to.html
# https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk
# http://www.nltk.org/howto/twitter.html
# https://docs.python.org/3/library/datetime.html#timedelta-objects
# https://pandas.pydata.org/pandas-docs/stable/reference/index.html
# https://learn.sparkfun.com/tutorials/graph-sensor-data-with-python-and-matplotlib/update-a-graph-in-real-time
# https://www.r-bloggers.com/2018/07/how-to-get-live-stock-prices-with-python/
#
# LIBRARIES USED:
# https://github.com/tweepy/tweepy
# https://www.nltk.org/
# https://matplotlib.org/stable/index.html
# https://pypi.org/project/yahoo-fin/
# https://pandas.pydata.org/

import tweepy
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import twitter_samples
from nltk import classify
from nltk import NaiveBayesClassifier
from collections import Counter
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import pandas_datareader.data as web
import pandas as pd
from yahoo_fin import stock_info as si

#twitter auth info MODIFY THIS WITH YOUR TOKENS
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

#tweepy api object
api = tweepy.API(auth)

#Load the positive and negative datasets from nltk
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

#contractions dictionary for replacing them in tweets
contractions_dict = { 
"ain't": "am not / are not / is not / has not / have not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is / how does",
"I'd": "I had / I would",
"I'd've": "I would have",
"I'll": "I shall / I will",
"I'll've": "I shall have / I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}

contractions_re = re.compile('(%s)'%'|'.join(contractions_dict.keys()))

#function to remove contractions from tweets
def expand_contractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)

#function to clean, tokenize, and lemmatize tweets
def clean(text):
    #remove the contractions
    unclean = expand_contractions(text)

    #remove http urls
    tweet = re.sub(r"http\S+", "", unclean)
    
    #remove https urls
    tweet = re.sub(r"https\S+", "", unclean)
    
    #remove hashtags
    tweet = re.sub(r"#(\w+)", ' ', tweet, flags=re.MULTILINE)
    
    #remove @ mentions
    tweet = re.sub(r"@(\w+)", ' ', tweet, flags=re.MULTILINE)
    
    #remove stock symbols from tweets
    tweet = re.sub(r"\$(\w+)", ' ', tweet, flags=re.MULTILINE)
    
    #remove digits from tweets
    tweet = re.sub(r"\d", "", tweet)
    
    #remove all emojis and punctuation from tweets
    tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet).split())
    
    #converts tweets to lowercase
    tweet = tweet.lower()
    
    #tokenize the normalized tweets
    sent = word_tokenize(tweet)
    
    #lemmatize the tokens to get the word stems
    sentence = lemmatize_sentence(sent)
    return sentence

def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []

    #Tag parts of speech
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence

#text for positive tweets
positive_tweet_tokens = twitter_samples.strings('positive_tweets.json')
#text for negative tweets
negative_tweet_tokens = twitter_samples.strings('negative_tweets.json')

positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []

#tokens for the cleaned positive tweets
for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(clean(tokens))

#tokens for cleaned negative tweets
for tokens in negative_tweet_tokens:
    negative_cleaned_tokens_list.append(clean(tokens))

#add stock specific positive tokens
positive_cleaned_tokens_list.append(["up", "bull", "bullish", "high"])

#add stock specific negative tokens
negative_cleaned_tokens_list.append(["down", "fall", "bear", "bearish", "low"])

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

#positive data set with label positive
positive_dataset = [(tweet_dict, "Positive")
                     for tweet_dict in positive_tokens_for_model]

#negative data set with label negative
negative_dataset = [(tweet_dict, "Negative")
                     for tweet_dict in negative_tokens_for_model]

#combined positive and negative dataset
dataset = positive_dataset + negative_dataset

#shuffle the dataset
random.shuffle(dataset)

#train test split from the combined dataset
train_data = dataset[:7000]
test_data = dataset[7000:]

#train the classifier on the train data
classifier = NaiveBayesClassifier.train(train_data)

#find accuracy based on the test data
print("Accuracy is:", classify.accuracy(classifier, test_data))

#show most informative features for the model
print(classifier.show_most_informative_features(10))

sentiments = []

# sentiment analysis for only the top tweets for a ticker
# ticker: the ticker to look up
# mode: 'popular','recent', or 'mixed'
# num: the number of tweets to get
def topOnly(ticker, mode, num):
    #search for all the top tweets for a ticker
    public_tweets = api.search(q="$"+ticker + " -filter:retweets",lang="en",result_type=mode,count=num,tweet_mode='extended')
    for tweet in public_tweets:
        #clean the tweet
        cleaned = clean(tweet.full_text)
        #find the sentiments from the model
        sentiments.append(classifier.classify(dict([token, True] for token in cleaned)))
    #count the number of positive tweets
    pos = sentiments.count("Positive")
    #count the number of negative tweets
    neg = sentiments.count("Negative")
    #find the overall score for the ticker
    score = ((pos * 1) + (neg * -1))/(pos+neg)
    print(score)

#list to hold the dates and times
xs = []

#hold the avg of the scores 
ys = []

#array to hold normalized prices
prices = []

#list to hold the scores
scores = []

#live stream listening
class MyStreamListener(tweepy.StreamListener):
    #when a new tweet is recieved
    def on_status(self, status):
        #clean the tweet
        cleaned = clean(status.text)
        
        #calculate the sentiment for the tweet
        #sentiments.append(classifier.classify(dict([token, True] for token in cleaned)))
        
        #calculate the score for the tweet
        #pos = sentiments.count("Positive")
        #neg = sentiments.count("Negative")
        #score = ((pos * 1) + (neg * -1))/(pos+neg)
        
        #add the score to the array
        sentiment = classifier.classify(dict([token, True] for token in cleaned))
        
        if sentiment == "Positive":
            scores.append(1)
        else:
            scores.append(-1)

    def on_error(self, status_code):
        #stop the stream on error
        return False

# stream data live for a ticker
# ticker: the ticker to stream
# interval: how often to update the graph (in milliseconds)
# numpoints: number of points to display on the graph
# weeksback: how far back to go for the high/low to normalize stock data
def stream(ticker, interval, numpoints, weeksback):
    #get 52 week high and low for normalizizing
    start = dt.datetime.now() - dt.timedelta(weeks=weeksback)
    end = dt.datetime.now()
    
    df = web.DataReader(ticker, 'yahoo', start, end)
    close_px = df['Adj Close']

    high = close_px.max()
    low = close_px.min()

    #matplotlib figure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    #clear the arrays
    sentiments = []
    
    #start the stream
    myStreamListener = MyStreamListener()
    myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener)
    
    #filter for the specified ticker
    myStream.filter(track=["$"+ticker], is_async=True)
    
    #animate the graphs
    def animate(i, xs, ys, scores, prices):
        #add the date to the array
        if len(scores) != 0:
            xs.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
            avgscore = sum(scores)/len(scores) 
            ys.append((avgscore-min(scores))/(max(scores)-min(scores)))
            price = si.get_live_price(ticker)
            prices.append((price-low)/(high-low))

        # Limit x and y lists to 20 items
        xs = xs[-numpoints:]
        ys = ys[-numpoints:]
        prices = prices[-numpoints:]

        #clear scores array
        #scores = []

        # Draw x and y lists
        ax.clear()
        ax.plot(xs, ys, linestyle='--', marker='o', color='b')
        ax.plot(xs, prices, linestyle='--', marker='x', color='r')
        ax.fill_between(xs, ys, prices, alpha=0.7)

        # Format plot
        plt.xticks(rotation=45, ha='right')
        plt.subplots_adjust(bottom=0.30)
        plt.title(ticker.upper() + ' sentiment over time')
        plt.ylabel('Sentiment')
        plt.xlabel('Time')

    # Set up plot to call animate() function periodically
    ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys, scores, prices), interval=interval)
    plt.show()

#topOnly("tsla", "popular", 100)
stream("tsla", 60000, 60, 1)