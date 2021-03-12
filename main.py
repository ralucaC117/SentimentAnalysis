import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")


def cleanTxt(text):
    text = re.sub('@[A-Za-z0–9]+', '', text)  # Removing @mentions
    text = re.sub('#', '', text)  # Removing '#' hash tag
    text = re.sub('RT[\s]+', '', text)  # Removing RT
    text = re.sub('https?:\/\/\S+', '', text)  # Removing hyperlink

    return text


consumerKey = 'WtsSPsmlZu1vcOX9SU7M3C643'
consumerSecret = 'vxiyXonyG72ao2Zi2zhaMZDUVVrFVYIC0RRMInAVwLTZNd13QP'
accessToken = '1332267131384291330-XEC4N6m7R6zq0Qz4UjpEHpYPZycWdh'
accessTokenSecret = 'Ws7VoWUCOiKFPdP3zb5HkKI9UOoTL7Nvtz8xlMo7SkrIO'

authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret)
authenticate.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(authenticate, wait_on_rate_limit=True)

posts = api.user_timeline(screen_name="BillGates", count=100, language="en", tweet_mode="extended")
# i = 1
# print("Show the 5 recent tweets")
# for tweet in posts[0:5]:
#   print(str(i) + ') ' +tweet.full_text + '\n')
#   i = i + 1

df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])
# df.head()

df['Tweets'] = df['Tweets'].apply(cleanTxt)


# print(df)


def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
    return TextBlob(text).sentiment.polarity


df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
df['Polarity'] = df['Tweets'].apply(getPolarity)


# print(df)


def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'


df['Analysis'] = df['Polarity'].apply(getAnalysis)
print(df)

# print('Printing positive tweets:\n')
# j = 1
# sortedDF = df.sort_values(by=['Polarity'])  # Sort the tweets
# for i in range(0, sortedDF.shape[0]):
#     if (sortedDF['Analysis'][i] == 'Positive'):
#         print(str(j) + ') ' + sortedDF['Tweets'][i])
#         print()
#         j = j + 1
#
# print('Printing negative tweets:\n')
# j = 1
# sortedDF = df.sort_values(by=['Polarity'], ascending=False)  # Sort the tweets
# for i in range(0, sortedDF.shape[0]):
#     if (sortedDF['Analysis'][i] == 'Negative'):
#         print(str(j) + ') ' + sortedDF['Tweets'][i])
#         print()
#         j = j + 1

plt.figure(figsize=(8,6))
for i in range(0, df.shape[0]):
  plt.scatter(df["Polarity"][i], df["Subjectivity"][i], color='Blue')
plt.title('Sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.show()

#print(df['Analysis'].value_counts())

# plt.title('Sentiment Analysis')
# plt.xlabel('Sentiment')
# plt.ylabel('Counts')
# df['Analysis'].value_counts().plot(kind='bar')
# plt.show()
