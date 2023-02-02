#importing relevant libraries to use in the project
import pytrends
from pytrends.request import TrendReq
import pandas as pd
#check for trending stocks

def getTrendingStocks(timeFrame = 'now 1-H'):
    #code for finding the frequencies of term in google trends and related queries
    trending_terms = TrendReq(hl='en-US', tz=360)
    keywords = ['share price','stock price']
    trending_terms.build_payload(
          kw_list=keywords,
          cat=0,
          timeframe='now 1-H',
          geo='US',
          gprop='')
    term_interest_over_time = trending_terms.interest_over_time()
    related_queries= trending_terms.related_queries()

    # Code for cleaning related queries to find top searched queries and rising queries
    top_queries=[]
    rising_queries=[]
    for key, value in related_queries.items():
        for k1, v1 in value.items():
            if(k1=="top"):
                top_queries.append(v1)
            elif(k1=="rising"):
                rising_queries.append(v1)
    top_searched=pd.DataFrame(top_queries[1])
    print(top_searched)
    rising_searched=pd.DataFrame(rising_queries[1])
    print(rising_searched)

getTrendingStocks()