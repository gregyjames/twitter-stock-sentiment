[![CodeFactor](https://www.codefactor.io/repository/github/gregyjames/twitter-stock-sentiment/badge)](https://www.codefactor.io/repository/github/gregyjames/twitter-stock-sentiment)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/4100948574bb40888686954b355f44fd)](https://www.codacy.com/gh/gregyjames/twitter-stock-sentiment/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=gregyjames/twitter-stock-sentiment&amp;utm_campaign=Badge_Grade)
# twitter-stock-sentiment
This tool trains a NLTK Naive Bayes classifier on positive and negative tweet data then collects, cleans, and classifies the sentiments of live and trending tweets about specified stock tickers to gauge overall market sentiment and its effects on stock price.

## Modes

### Top tweets
`topOnly("tsla", "popular", 100)`

Collects the specified number of tweets for the specified ticker symbol to get the overall sentiment for the stock at the given moment.

#### Options:
`ticker`: the ticker to look up

`mode`: 'popular','recent', or 'mixed'

`num`: the number of tweets to get

### Stream
![Example Graph](https://i.imgur.com/Xn8906P.png)
`stream("tsla", 60000, 60, 1)`

Uses Tweepy stream to collect and calculate the sentiments for live tweets for a specified ticker and plots them against a normalized price curve to see the relationship of twitters sentiment on the overall price of the stock over time. Could be used to find potential good times to invest.

### Options:
`ticker`: the ticker to stream

`interval`: how often to update the graph (in milliseconds)

`numpoints`: number of points to display on the graph

`weeksback`: how far back to go for the high/low to normalize stock data

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
