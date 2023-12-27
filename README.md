
# Financial News Sentiment Analysis

## Overview

This project performs sentiment analysis on financial news headlines related to big technology companies, specifically Google, Amazon, Netflix, and Apple. The techniques , KMeans clustering, LSTM RNN and NLTK library, are employed to predict sentiments (positive, negative, and neutral).


### Data Collection

Financial news data is scraped in real-time from *finviz* using Python's BeautifulSoup library. The headlines are then cleaned and preprocessed for analysis.

### Preprocessing

The headlines undergo preprocessing steps, including lemmatization and removal of stopwords, to prepare the data for sentiment analysis.


### Analysis Techniques


#### *KMeans Clustering* :

The CountVectorizer is used to convert the text data into a matrix of token counts.
A pre-trained KMeans model is loaded, and sentiment predictions are made based on the clusters.

#### *NLTK Library* :

The NLTK library, specifically the VADER sentiment analyzer, is used for sentiment analysis.
Sentiment scores are calculated, and headlines are classified as positive, negative, or neutral.



#### LSTM Recurrent Neural Network (RNN):

An LSTM RNN model is trained on the preprocessed data for more advanced sentiment analysis. The model predicts sentiments as positive, negative, or neutral based on the learned patterns in the text data.





## Demo


https://financialsentimentprediction.onrender.com/






