import pandas as pd
import tweepy as tw

accounts = [
    ['consumer_key',
    'consumer_secret',
    'access_token_key',
    'access_token_secret'],
    ['consumer_key_2',
    'consumer_secret_2',
    'access_token_key_2',
    'access_token_secret_2']
    ] # add as many accounts as you have access to (around 7-8 accounts is when you hit 100% uptime)
    
def init(i=0,wait=True):
    # Input: 
      # i: Integer. The index of the list 'accounts' used to select different accounts
      # wait: Boolean. True = wait the remaining time to pull a tweet (max 15 minutes)
    consumer_key = accounts[i][0]
    consumer_secret = accounts[i][1]
    access_token_key = accounts[i][2]
    access_token_secret = accounts[i][3]
    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token_key, access_token_secret)
    api = tw.API(auth,wait_on_rate_limit=wait,wait_on_rate_limit_notify=wait)
    return api

# tweets = tweepy cursor
def extract_tweets(tweets,region= None):
    li =[]
    for tweet in tweets:  
        # Add or remove fields to capture or remove different meta data in a tweet JSON
        data=[tweet.user.screen_name, tweet.user.id, str(tweet.full_text),
             tweet.id,
             tweet.retweet_count,
             tweet.favorite_count,
             tweet.user.followers_count,
             tweet.user.friends_count,
             tweet.user.verified,
             (tweet.created_at - tweet.user.created_at).days,
             tweet.created_at,
             tweet.metadata['result_type'],
             tweet.user.geo_enabled,
             tweet.place,
             tweet.coordinates,
             tweet.user.location,
             region]
        try: # If it's a retweet, get some metadata
            tweet.retweeted_status # Immediately fails if the tweet is not a retweet
            data.append(1)
            data.append(tweet.retweeted_status.user.location)
            data.append(tweet.retweeted_status.created_at)
            data.append(tweet.retweeted_status.id)
        except: # If it's not, then fill nothing
            data.append(0) # this is retweet field
            data.append(None) # this is retweet user field
            data.append(0) # original_date
            data.append(0) # original_tweet_id
        li.append(data)
        
    # If any fields are updated, make sure their position correspond to the columns in this dataframe
    
    df = pd.DataFrame(data=li,columns = ['screen_name','user_id','full_text','tweet_id',
                                           'retweet_count','favorite_count','followers_count','friends_count',
                                           'verified','days_since_creation','tweet_date','result_type','geo_enabled',
                                           'place','coordinates','user_location','region','retweet','original_location',
                                           'original_date','original_tweet_id'])
    return df

# read region text file (check file for the format)
def extract_locations_from_txt(coordinates_path):
    region = []
    coordinates = []
    temp=[]
    next_is_region = True
    with open(coordinates_path) as f:
        for line in f:
            line = line.strip()
            if next_is_region:
                region.append(line)
                next_is_region = False
            elif line == "":
                next_is_region = True
                coordinates.append(temp)
                temp=[]
            else:
                temp.append(line)
        coordinates.append(temp) # covers the last set of circles
    df = pd.DataFrame(region,columns=['region'])
    df['circles'] = coordinates
    return df
