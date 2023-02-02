import praw
import datetime
import requests

def get_posts_for_time_period(sub, beginning, end=int(datetime.datetime.now().timestamp())):
    """
    Gets posts from the given subreddit for the given time period
    :param sub: the subreddit to retrieve posts from
    :param beginning: The unix timestamp of when the posts should begin
    :param end: The unix timestamp of when the posts should end (defaults to right now)
    :return:
    """
    print("Querying pushshift")
    url = "https://apiv2.pushshift.io/reddit/submission/search/" \
               "?subreddit={0}" \
               "&limit=500" \
               "&after={1}" \
               "&before={2}".format(sub, beginning, end)
         
    response = requests.get(url)
    response.raise_for_status()  # raises exception when not a 2xx response
    if response.status_code != 204:
        return response.json()
    else:
        print('bad response')


def getRecentHots():

    reddit = praw.Reddit(client_id='l0vYqYMpPR7fwqrEh2ET1w', client_secret='PqjHAMaSTT4UizII24zyXeHr5eLR1Q', user_agent='stockBot')

    hot_posts = reddit.subreddit('worldnews').hot(limit=50)
    return hot_posts