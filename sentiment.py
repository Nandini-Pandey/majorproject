import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer







# Data scraping from finviz
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup

finviz_url = 'https://finviz.com/quote.ashx?t='
tickers = ['AMZN', 'GOOG', 'NFLX','AAPL']  # ticker is a unique series of letters assigned to a security for trading purposes

news_tables = {}
for ticker in tickers:
    url = finviz_url + ticker

    req = Request(url=url, headers={'user-agent': 'my-app'})

    try:
        response = urlopen(req)
        html = BeautifulSoup(response, 'html')
        news_table = html.find(id='news-table')
        news_tables[ticker] = news_table
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")



# Data cleaning
parsed_data = []
for ticker, news_table in news_tables.items():
    for row in news_table.findAll('tr'):
        if row.a is not None:
            title = row.a.get_text()
            date_data = row.td.text.split(' ')
            date_data = list(filter(lambda x: x.strip() != '', date_data))  # Remove empty strings
            if len(date_data) == 1:
                time = date_data[0].strip()
            else:
                date = date_data[0].strip()
                time = date_data[1].strip()

            parsed_data.append([ticker, date, time, title])
        else:
            title = "Default value"
            parsed_data.append([ticker, "", "", title])

print(parsed_data)



# data creation

df=pd.DataFrame(parsed_data,columns=['ticker','date','time','title'])
print(df.head())
lemmatizer = WordNetLemmatizer()
print(df['title'])
corpus=[]






# Lemmatization
for i in range(0, len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['title'][i])
    review = review.lower()
    review = review.split()

    review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    print(review)  # Add this line to see the preprocessed review
    corpus.append(review)





from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()



# loading the saved model
loaded_model = pickle.load(open('model.sav', 'rb'))



# Perform K-Means clustering
def kmeanssentimentanalysis(X,loaded_model):
    k = 3
    kmeans = KMeans(n_clusters=k)
    cluster_labels = loaded_model.predict(X)
    cluster_sentiments = {
    0: "positive",
    1: "negative",
    2: "neutral"
    }
    # Assign sentiments to sentences based on cluster labels
    sentiment_predictions = [cluster_sentiments[label] for label in cluster_labels]
    return sentiment_predictions
    
    
    

#using nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

def nltksentimentanalysis(data):
    sid = SentimentIntensityAnalyzer()
    sentiment_predictions = []

    for sentence in data:
        ss = sid.polarity_scores(sentence)
        compound = ss['compound']
        
        if compound >= 0.05:
            sentiment = "Positive"
        elif compound <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        sentiment_predictions.append(sentiment)

    return sentiment_predictions



def main():
    st.markdown('<h1 style="color:dark blue;">Sentiment Analysis of financial news of big technology companies</h1>', unsafe_allow_html=True)
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/44/Investors_News_tips_-_Stock_Market_Quotes_%26_Financial_News.jpg", caption="Image from Wikimedia Commons")

    st.text("This project is about predicting the sentiments of financial news headlines viz. ")
    st.text(" positive(0), negative(1) and neutral(2).")
    st.text("The real timen financial news headlines of 4 tech giants Google,Amazon,Netflix ")
    st.text("and Apple were scraped from finviz using python library BeautifulSoup.")
    st.text("Sentiment analysis has been done using two techniques : KMeans and nltk library. ")
    st.text("User can choose the technique by which they want to see the prediction of sentiment. ") 

    if st.button("View dataframe"):
        df=pd.DataFrame(parsed_data,columns=['ticker','date','time','title'])
        st.dataframe(df[['ticker','date','title']])

    
    st.text("Choose the technique using which you wish to do the sentiment analysis --> ") 
    st.text("KMeans or nltk ")
    prediction=''
    if st.button("KMeans"):
        prediction= kmeanssentimentanalysis(X,loaded_model)
        st.success("Sentiment prediction")
        st.write(prediction)    

    elif (st.button("nltk")):
        prediction= nltksentimentanalysis(corpus)
        st.success('Sentiment Prediction')
        st.write(prediction)
        
    

        

    

    

    
if __name__=='__main__':
    main()
