import pandas as pd
from datetime import datetime, timedelta
import random
import tweetId_functions as twtID
import tweepy_utils as twpUtil

# This is a simplified version of velocitiy_utils.py, where I'm including only the important functions for VBEST
# Hopefully this makes it easier to extend from our work

# SPP = Size Per Pull
    # The maximum amount of tweets you want returned from a query, max = 100
    # Always set to 100, but left as a parameter in case Twitter fixes their API limitation 
        # (currently Twitter restricts only by # of queries, but not # of tweets like they state they do)

def dt2sec(date):
    return float((date.hour * 60 + date.minute) * 60 + date.second + (date.microsecond / 1000000))
    
def fake_sample(df,upper_id,lower_id, spp = 100):
    # Sample the already collected corpus of tweets, as if we were collecting them live
    return df.loc[(df['tweet_id'] > lower_id) & (df['tweet_id'] <= upper_id)].head(spp)

def sec2ids(date, time_list):
    # Take a list of timestamps for a given date and convert them into their corresponding tweet IDs 
    datet = datetime.strptime(date,'%Y-%m-%d') + timedelta(days=0)
    sample_ids = [tweetId_from_datetime(timedelta(seconds = sec) + datet) for sec in time_list]
    return sample_ids

def real_sample(twitter,circle,region,until_date, max_id,since_id,keyword="-filter:retweets",spp=100):
    # sample_id = 1 element in sample_intervals
    li = []
    previous_id = max_id
    for i in range(0,spp//100):
        tweets = twpUtil.build_cursor_range(twitter,until_date, since_id = since_id,max_id=previous_id, location= circle, keyword=keyword)
        li.append(twpUtil.extract_tweets(tweets, region))
        previous_id = li[-1].iloc[-1].tweet_id -1
    temp_df = pd.concat(li)
    temp_df['seconds']=temp_df.tweet_id.apply(twtID.datetime_from_tweetId).apply(dt2sec)
    return temp_df

def time_sampling(date_string, queries= 360, start_range_max = None,set_random_number=0):
    # returns time sample intervals and their corresponding tweet IDs, which lets us sample Twitter easily
    # start_range_max lets you define a maximum time to start at, which lets you be able to avoid selecting the last few seconds of a day for sampling 
        # If a uniform sample was to be started at the last 5 seconds of the day, that gives 5 seconds to collect 100 tweets, which usually didn't happen
        # This let's us make sure we sample just far enough from the end of the day to get at least 100 tweets
            # (Usually start_range_max = (length of time interval in minutes -2.5 minutes)
            
    date = datetime.strptime(date_string,'%Y-%m-%d') + timedelta(days=1)
    if start_range_max is None:
        start_range_max = 1440/queries # 1440 = number of minutes in a day
    constant = 1440/queries # to always take steps in the correct length of time
    # we are 1 ms over what tweets could possibly show up, but it doesn't matter because we CAN'T recieve tweets from the next day, since we can't recieve tweets from that day anyways
    id_list = []
    date_list = [] # not needed, but nice to see what's going on
    start = date - timedelta(minutes= random_number)
    for i in range(0, int(queries)):
        time = start - timedelta(minutes = constant * i)
        date_list.append(time)
        id_list.append(twtID.tweetId_from_datetime(time))
    if debug:
    return id_list, date_list

#%% Compute Velocity from sample intervals returned by time_sampling


def get_velocity(df, date, sample_intervals, spp,query_limits):
    until_date = twtID.untilDateStr_from_datetimeStr(date) # we first generate the until dates we want w/ random sampling
    since_id = twtID.sinceId_from_untilDate(until_date)
        
    dfs = pd.DataFrame([])
    velocities = [] # number_tweets/(time in seconds spanned to get them), So we are getting time required to get 100 tweets
    min_range = []
    max_range = []
    lengths = []
    for i in range(len(sample_intervals)):
        if i == len(sample_intervals)-1:
            temp_df = fake_sample(df,sample_intervals[i],since_id,spp)
        else:
            # temp_df = fake_sample(df,sample_intervals[i],sample_intervals[i+1],spp) # this prevents overlap, but isn't how we do it on twitter
            temp_df = fake_sample(df,sample_intervals[i],since_id,spp)
        lengths.append(len(temp_df))
        
        if len(temp_df)==0: 
            velocities.append(0)
            min_range.append(0)
            max_range.append(0)
            continue
        
        dfs = dfs.append(temp_df)
        min_range.append(temp_df.iloc[-1].tweet_id)
        max_range.append(temp_df.iloc[0].tweet_id)
    
        subtract = abs(twtID.datetime_from_tweetId(temp_df.iloc[0].tweet_id) - twtID.datetime_from_tweetId(temp_df.iloc[-1].tweet_id))
        time_diff = subtract.seconds + (subtract.microseconds*10e-7)
         # if time diff is 0, then add .0001 s (as its the lowest amount of time that COULD have transpired)
        if time_diff == 0:
            velocities.append(.0001)
        else:
            velocities.append(len(temp_df)/time_diff)
        
        query_limits -= 1 
    return velocities,min_range,max_range,dfs,query_limits,lengths

# Velocity computation using Dr. Buskirks suggestion
    # Velocity is [given interval time - time last tweet occured]
        # Thus velocity is a measurement of a time interval rather than a query
def get_velocity_buskirk(df, date, sample_intervals, spp,query_limits):
    until_date = twtID.untilDateStr_from_datetimeStr(date) # we first generate the until dates we want w/ random sampling
    since_id = twtID.sinceId_from_untilDate(until_date)
        
    dfs = pd.DataFrame([])
    velocities = [] # number_tweets/(time in seconds spanned to get them), So we are getting time required to get 100 tweets
    min_range = []
    max_range = []
    lengths = []
    for i in range(len(sample_intervals)):
        if i == len(sample_intervals)-1:
            temp_df = fake_sample(df,sample_intervals[i],since_id,spp)
        else:
            # temp_df = fake_sample(df,sample_intervals[i],sample_intervals[i+1],spp) # this prevents overlap, but isn't how we do it on twitter
            temp_df = fake_sample(df,sample_intervals[i],since_id,spp)
        lengths.append(len(temp_df))
        
        if len(temp_df)==0: 
            velocities.append(0)
            min_range.append(0)
            max_range.append(0)
            continue
        
        # if time diff is 0, then add .0001 s
        
        #print(str(time_intervals[i])[11:],":",str(time_intervals[i+1])[11:], f"| time covered in {spp} tweets:",time_diff, "seconds")
        dfs = dfs.append(temp_df)
        min_range.append(temp_df.iloc[-1].tweet_id)
        max_range.append(temp_df.iloc[0].tweet_id)
        # instead of subtracting from first timepoint found, do the start of the time point we are considering
        subtract = abs(twtID.datetime_from_tweetId(sample_intervals[i]) - twtID.datetime_from_tweetId(temp_df.iloc[-1].tweet_id))
        time_diff = subtract.seconds + (subtract.microseconds*10e-7)
        #ignore an overlapping issue atm instead of preventing it
        if time_diff == 0:
            velocities.append(0)
        else:
            velocities.append(len(temp_df)/time_diff)
        
        query_limits -= 1 
    return velocities,min_range,max_range,dfs,query_limits,lengths

#twitter parameter not being used anymore, so we can be more efficent
def get_velocity_real(twitter,circle,region,date, sample_intervals, spp,query_limits,ii,keyword="-filter:retweets"):
    # This does NOT use the buskirk velocity suggestion
    
    accounts = twpUtil.accounts
    until_date = twtID.untilDateStr_from_datetimeStr(date) # we first generate the until dates we want w/ random sampling
    since_id = twtID.sinceId_from_untilDate(until_date)
        
    dfs = pd.DataFrame([])
    velocities = [] # number_tweets/(time in seconds spanned to get them), So we are getting time required to get 100 tweets
    min_range = []
    max_range = []
    sizes = []
    for i in range(len(sample_intervals)):
        twitter = twpUtil.init((ii // 180) % len(accounts))
        if i == len(sample_intervals)-1:
            temp_df = real_sample(twitter,circle,region,until_date,since_id=since_id,max_id=sample_intervals[i],spp=spp,keyword=keyword)
        else:
            temp_df = real_sample(twitter,circle,region,until_date,since_id=since_id,max_id=sample_intervals[i],spp=spp,keyword=keyword)
        sizes.append(len(temp_df))
        ii+= int(spp//100) # since we can go up by more than 1
        if len(temp_df)==0: 
            velocities.append(0)
            min_range.append(0)
            max_range.append(0)
            continue
        
        #print(str(time_intervals[i])[11:],":",str(time_intervals[i+1])[11:], f"| time covered in {spp} tweets:",time_diff, "seconds")
        dfs = dfs.append(temp_df)
        min_range.append(temp_df.iloc[-1].tweet_id)
        max_range.append(temp_df.iloc[0].tweet_id)
    
        subtract = abs(twtID.datetime_from_tweetId(temp_df.iloc[0].tweet_id) - twtID.datetime_from_tweetId(temp_df.iloc[-1].tweet_id))
        time_diff = subtract.seconds + (subtract.microseconds*10e-7)
            
        #ignore an overlapping issue atm instead of preventing it. Not changing this code even though prior If statement makes this redundent
        if time_diff == 0:
            velocities.append(0)
        else:
            velocities.append(len(temp_df)/time_diff)
        print(f"{i} : {len(sample_intervals)} | {len(temp_df)} | {str(temp_df.iloc[-1].tweet_date)} | Region: {region}")
        query_limits -= 1 
    return velocities,min_range,max_range,dfs,query_limits,sizes

#%% Create a consistent and easy to work with dataframe for the sampled velocity

def clean_scout(v1,min1,max1,dfs,query_limits,lengths,time_intervals):
    # used to clean and create consistent columns after computing velocity
    scout = pd.DataFrame(list(reversed(v1)),columns=['velocity'])
    scout['time'] = time_intervals[::-1]
    scout['min_id']=list(reversed(min1))
    scout['max_id'] = list(reversed(max1))
    scout['time_end']=scout['min_id'].apply(twtID.datetime_from_tweetId)
    scout['utc'] = scout.time.apply(twtID.utc_from_datetime)
    scout['seconds']=scout['time'].apply(dt2sec)
    scout['tpp'] = lengths
    return scout, dfs, query_limits

def velocity_cleaner(df,date,sample_intervals,time_intervals,spp,query_limits):
    return clean_scout(get_velocity(df,date,sample_intervals,spp,query_limits),time_intervals)

# for Dr. Buskirks velocity method
def velocity_cleaner_buskirk(df,date,sample_intervals,time_intervals,spp,query_limits):
    return clean_scout(get_velocity_buskirk(df,date,sample_intervals,spp,query_limits),time_intervals)

def velocity_cleaner_real(twitter,circle,region,date,sample_intervals,time_intervals,spp,query_limits,ii,keyword="-filter:retweets"):
    return clean_scout(get_velocity_real(twitter,circle,region,date,sample_intervals,spp,query_limits,ii,keyword = keyword),time_intervals)

#%% RPY2 STUFF
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import STAP

def r2df(rdf):
    with localconverter(robjects.default_converter + pandas2ri.converter):
        df = robjects.conversion.rpy2py(rdf)
    return df

# rdf.rx2('velocity') # to refer to a r df 
def df2r(df):
    with localconverter(robjects.default_converter + pandas2ri.converter):
        rdf = robjects.conversion.py2rpy(df)
    return rdf

def custom_func(file_name, file_path = r"G:\Programming\R_scripts"):
    with open(file_path+'\\'+file_name, 'r') as f:
        string = f.read()
    myfunc = STAP(string, "myfunc")
    return myfunc

#%% VBest

def vbest_test(scout,psu_n=624):
    # The VBEST algorithm which calls R code from Python
    # Used in the experiment of VBEST, could be simplified by removing unneeded samples 
    
    vbest = custom_func('VBESTfinalR.R')
    rdf = df2r(scout[['seconds','velocity']])
    res = vbest.VBESTsamp(rdf.rx2('seconds'),rdf.rx2('velocity'),Vbestn=psu_n)
    # get data out of R format
    vsample= [list(r2df(res[0][i])) for i in range(3)]
    ssample= [list(r2df(res[1][i])) for i in range(3)]
    isample = [list(r2df(res[2][i])) for i in range(3)]
    span = r2df(res[3])
    total_tweets = r2df(res[4])[0]
    npsu = r2df(res[5])[0]
    expected_tweets = r2df(res[6])
    over_tweets = r2df(res[7])
    inv_weights = [r2df(res[8][i]) for i in range(3)]
    size = r2df(res[9])
    seeds = r2df(res[10])
    # condense similar data into dataframe to condense outputs
    samples = []
    for i in range(3):
        sample = pd.DataFrame(vsample[i], columns=['vsample'])
        sample['ssample'] = ssample[i]
        sample['isample'] = isample[i]
        sample['inverse_weights'] = inv_weights[i]
        samples.append(sample)
    psu_stuff = pd.DataFrame(expected_tweets, columns=['expected'])
    psu_stuff['over'] = over_tweets
    
    return samples,psu_stuff,span,total_tweets,npsu, size, seeds

def vbest_short(scout,psu_n=624):
    # Simplified VBEST, only returns the loess model information. Used when comparing against other smoothing functions
    vbest = custom_func('VBESTfinalR.R')
    rdf = df2r(scout[['seconds','velocity']])
    res = vbest.VBESTshort(rdf.rx2('seconds'),rdf.rx2('velocity'),Vbestn=psu_n)
    yhats = r2df(res[0])
    span = r2df(res[1])[0]
    total_tweets = r2df(res[2])[0]
    seed = r2df(res[3])[0]
    return yhats,span,total_tweets,seed
