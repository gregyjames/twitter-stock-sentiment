# twitter-stock-sentiment
This tool trains a NLTK Naive Bayes classifier then collects, cleans, and classifies the sentiments of live and trending tweets.

## Modes

### Top tweets

Collects the specified number of tweets for the specified ticker symbol to get the overall sentiment for the stock at the given moment.

### Stream

Uses Tweepy stream to collect and calculate the sentiments for live tweets for a specified ticker and plots them against a normalized price curve to see the relationship of twitters sentiment on the overall price of the stock over time. Could be used to find potential good times to invest.

## License
MIT License

Copyright (c) 2021 Greg James

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
