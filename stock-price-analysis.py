#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
import time
import datetime
from datetime import date
get_ipython().run_line_magic('matplotlib', 'inline')
import mplfinance as mpf


# In[2]:


headers = {'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36'}


# In[3]:


today = datetime.datetime.now().replace(hour=11, minute=0, second=0, microsecond=0)
one_year = today  - datetime.timedelta(days=8000)
period1 = int(time.mktime(one_year.timetuple()))
period2 = int(time.mktime(today.timetuple()))


# In[4]:


def soup(symbol):
    url = f'https://query1.finance.yahoo.com/v7/finance/download/{symbol}?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true'
    response = requests.get(url, headers=headers)
    print(response)
    df_symbol = pd.read_csv(url)
    df_symbol.set_index('Date',inplace=True)
    dates = list(range(0,int(len(df_symbol))))
    if df_symbol.isnull().sum().sum() == 0:
        return df_symbol
       
    
symbol = 'AAPL'
df = soup(symbol)
df


# In[5]:


def plot_graph(df):
    df.plot.line(y="Close", use_index=True)

plot_graph(df)


# In[6]:


df["Tomorrow"] = df["Close"].shift(-1)
df


# In[7]:


df["Target"] = (df["Tomorrow"]>df["Close"]).astype(int)
df


# In[8]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 250, min_samples_split = 100, random_state=1)
train = df.iloc[:-100]
test = df.iloc[-100:]
predictors=["Open", "High", "Low", "Close", "Volume"]
model.fit(train[predictors], train["Target"])


# In[9]:


from sklearn.metrics import precision_score

preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index, name="Predicted-target")
preds.info()


# In[10]:


check = pd.concat([test["Target"], preds], axis=1)
check


# In[11]:


precision_score(test["Target"], preds)


# In[12]:


check.plot()


# In[13]:


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predicted-target")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


# In[14]:


df.shape[0]


# In[15]:


def backtest(df, model, predictors, start=2300, incr=230):
    complete_preds=[]
    
    for i in range(start, df.shape[0], incr):
        train = df.iloc[0:i].copy()
        test = df.iloc[i:(i+incr)].copy()
        predictions = predict(train, test, predictors, model)
        complete_preds.append(predictions)
        
    return pd.concat(complete_preds)


# In[16]:


predictions = backtest(df, model, predictors)
predictions


# In[17]:


predictions["Predicted-target"].value_counts()


# In[18]:


precision_score(predictions["Target"], predictions["Predicted-target"] )


# In[19]:


predictions["Target"].value_counts()/predictions.shape[0]


# In[20]:


bi =df.rolling(2).mean()
weekly = df.rolling(5).mean()
quarterly = df.rolling(90).mean()

df["Bi_mean"] = bi["Close"]/df["Close"]
df["weekly_mean"] = weekly["Close"]/df["Close"]
df["quarterly_mean"] = quarterly["Close"]/df["Close"]
df["open_close_ratio"] = df["Open"] / df["Close"]
df["high_close_ratio"] = df["High"] / df["Close"]
df["low_close_ratio"] = df["Low"] / df["Close"]


# In[21]:


new_predictors = ["Bi_mean","weekly_mean","quarterly_mean", "open_close_ratio", "high_close_ratio",  "low_close_ratio"  ]


# In[22]:


df


# In[23]:


df=df.dropna()


# In[24]:


new_predictors


# In[25]:


model = RandomForestClassifier(n_estimators=200, min_samples_split=100, random_state=1)


# In[26]:


model


# In[27]:


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >=.6] = 1
    preds[preds <.6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


# In[28]:


predictions = backtest(df, model, new_predictors)


# In[29]:


predictions["Predictions"].value_counts()


# In[30]:


precision_score(predictions["Target"], predictions["Predictions"])


# In[31]:


predictions


# In[ ]:




