from datetime import datetime,timedelta

# These are the main helper functions used from converting between date times, utc timestamps, and tweet IDs

# max_id VS tweet_id:
    # max_id = maximum value a tweet could be given the remaining 22 bits (upper bound of range)
    # tweet_id = minimum value a tweet could be given the remaining 22 bits (lower bound of range)

# 1288834974657 is the start of twitters snowflake implementation in UTC millisecond timestamp (the epoch)
EPOCH = 1288834974657

# return a millisecond timestamp FROM a tweet ID
def utcms_from_tweetId(id_):
    return ((id_ >> 22) + EPOCH)

# return a tweet ID FROM a millisecond timestamp
def tweetId_from_utcms(utc_ms):
    return ((utc_ms - EPOCH) << 22) + (2**22 -1)

def tweetId_from_utc(utc_s): # timestamp given in seconds
    return ((int(utc_s * 1000) - EPOCH) << 22 ) + (2**22 -1)

def datetime_from_utcms(utc_ms):
    return datetime.utcfromtimestamp(utc_ms/1000)

def datetime_from_tweetId(id_):
    return datetime_from_utcms(utcms_from_tweetId(id_))

def utc_from_datetime(date):
    return (date - datetime(1970, 1, 1))// timedelta(seconds=1)

def utcms_from_datetime(date):
    return (date - datetime(1970, 1, 1))// timedelta(microseconds=1000)

def tweetId_from_datetime(date, ms = False):
    if ms:
        return tweetId_from_utcms(utcms_from_datetime(date))
    return tweetId_from_utc(utc_from_datetime(date))

# date_string in format "year-month-day hour:minute:second". Making this a function for ease of use
def date_from_string(date_string, ms = False):
    if ms:
        return datetime.strptime(date_string,'%Y-%m-%d %H:%M:%S.%f')
    return datetime.strptime(date_string,'%Y-%m-%d %H:%M:%S')

def tweetId_from_datetimeStr(date_string, ms = False):
    return tweetId_from_datetime(date_from_string(date_string,ms),ms)

# date_string "year-day-month"
def untilDateStr_from_datetimeStr(date_string):
    return (datetime.strptime(date_string,'%Y-%m-%d') + timedelta(days=1)).date().strftime("%Y-%m-%d")

# given an until_date as string (if you want tweets from the 5th your until date = the 6th) 
    # until_date format: "year-month-day" or datetime.date
# returns tweet_id for the first second of the day
def sinceId_from_untilDate(until_dateStr):
    date = datetime.strptime(until_dateStr,'%Y-%m-%d') - timedelta(days=1)
    return tweetId_from_datetime(date) - (2**22 -1) # We want the minimum number possibly representable

# ---------------------------------------------------------------- #
# Misc utility 

# provide tweet from my DF, return URL of that tweet
def url_builder(tweet):
    return "https://twitter.com/"+ tweet['screen_name'] +"/status/"+ str(tweet['tweet_id'])

def bin64_from_int64(number): 
    return "{0:064b}".format(number)

def datacenter_from_tweetId(id_):
    return int(bin64_from_int64(id_)[42:47],2) # grab the corresponding bits
