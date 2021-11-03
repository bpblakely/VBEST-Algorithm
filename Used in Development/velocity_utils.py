import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import tweetId_functions as twtID
import tweepy_utils as twpUtil
import matplotlib.pyplot as plt
import seaborn as sns
import locale

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import STAP

import matplotlib.gridspec as gridspec
import matplotlib.pylab as pl
from scipy import stats

from glob import glob

# This file contains all helper functions used during research, almost all of these functions are unneeded for VBEST


# Function naming convention
    # Real in name = changes to work with and collect LIVE twitter data
        # So these functions typically have twitter query parameters and tweepy stuff
    
    # Fake (or doesnt have the name real) = Work with a dataframe of already collected tweets
        # You don't have to wait for tweets to be collected or provide query parameters and stuff
        
    # Scout = The sampled velocities (n of them) in VBEST first step
        # Scout literally meaning to scout out the distribution
        
# I could have probably done it better, but this was sufficient

#%% Helper Functions for corpus data

all_regions = ['Atlanta-Sandy Springs-Roswell, GA','Phoenix-Mesa-Scottsdale, AZ','Chicago-Naperville-Elgin, IL-IN-WI','Columbus, OH',
           'Pittsburgh, PA','Baltimore-Columbia-Towson, MD']
lazy_regions = {'at':all_regions[0],'ga':all_regions[0],'atl':all_regions[0], 'az':all_regions[1], 'phx':all_regions[1],'il':all_regions[2],
                'in':all_regions[2],'wi':all_regions[2], 'chicago':all_regions[2], 'chic':all_regions[2], 'oh':all_regions[3],
                'col':all_regions[3],'columbus':all_regions[3],'pit':all_regions[4],'pitt':all_regions[4],'pa':all_regions[4],
                'burg':all_regions[4], 'md':all_regions[5],'balt':all_regions[5],'baltimore':all_regions[5]}

corpus_directory = r'E:\my_tweets'

# use to easily set the main corpus directory for future use
def set_corpus_dir(directory):
    global corpus_directory
    corpus_directory = directory

# given a region str, return the corresponding region
def get_region(region_str):
    if region_str in all_regions: return region_str
    region_str = region_str.lower()
    global lazy_regions
    if region_str in lazy_regions.keys(): 
        return lazy_regions[region_str]
    print(f"{region_str} is not a valid region or region abbreviation")
    print(f"Possible regions: \n {all_regions}")    
    print("\nPossible abbreviations:")
    for k,v in lazy_regions.items():
        print(f"{k}: {v}")
    raise ValueError('Incorrect Region')
    
# returns the file path for given date and region as string
def filename_builder(date, region, directory = None):
    if directory is None:
        global corpus_directory
        directory = corpus_directory
    return directory +'\\'+date+'\\all_'+get_region(region)+'_'+date+'.csv.gz'
    
def get_corpus_count(date,region, directory = None):
    return pd.read_csv(filename_builder(date,region,directory),usecols=['tweet_id']).size

#%% Preprocessing

# reads tweets csv and correctly formats to be used with other functions
def read_tweets(file,*args,**kwargs):
    df= pd.read_csv(file,parse_dates=['tweet_date'],*args,**kwargs)
    if 'seconds' not in df.columns:
        df['seconds']=df.tweet_id.apply(twtID.datetime_from_tweetId).apply(dt2sec)
    if 'quer' in df.columns:
        if len(df) > 0 and df.quer.iloc[0] == 0:
            df['quer'] = df['quer'] +1
    return df

# given a file name, get the corresponding velocity file for it
def get_velocity_df(file):
    temp = file.split('all')
    v_file = temp[0]+r'velocity\velocity'+temp[1][:-3]
    dfv = pd.read_csv(v_file,parse_dates=['time'])
    return dfv[::-1].reset_index(drop=True)

def append_queries(file):
    dfv = get_velocity_df(file)
    df = read_tweets(file)
    amounts = dfv.pull_amount.values
    queries = []
    for i in range(len(amounts)):
        queries.extend([i+1]*amounts[i])
    df['quer'] = queries[:len(df)] # can't name trhe columns 'query' because df.query is a function
    return df

#%% real sampling

# sample_id = 1 element in sample_intervals
def real_sample(twitter,circle,region,until_date, max_id,since_id,keyword="-filter:retweets",spp=100):
    li = []
    previous_id = max_id
    for i in range(0,spp//100):
        tweets = twpUtil.build_cursor_range(twitter,until_date, since_id = since_id,max_id=previous_id, location= circle, keyword=keyword)
        li.append(twpUtil.extract_tweets(tweets, region))
        previous_id = li[-1].iloc[-1].tweet_id -1
    temp_df = pd.concat(li)
    temp_df['seconds']=temp_df.tweet_id.apply(twtID.datetime_from_tweetId).apply(dt2sec)
    return temp_df

#%% fake sampling

# def dt2sec(date):
#     return float((date.hour * 60 + date.minute) * 60 + date.second + (date.microsecond*10e-7))

def dt2sec(date):
    return float((date.hour * 60 + date.minute) * 60 + date.second + (date.microsecond / 1000000))
    
# Sample the already collected corpus of tweets, as if we were collecting them live
def fake_sample(df,upper_id,lower_id, spp = 100):
    return df.loc[(df['tweet_id'] > lower_id) & (df['tweet_id'] <= upper_id)].head(spp)

def time_sampling(date_string, queries= 360, start_range_max = None,debug = False,set_random_number=0):
    date = datetime.strptime(date_string,'%Y-%m-%d') + timedelta(days=1)
    if start_range_max is None:
        start_range_max = 1440/queries # 1440 = number of minutes in a day
    constant = 1440/queries # to always take steps in the correct length of time
    print(start_range_max)
    # we are 1 ms over what tweets could possibly show up, but it doesn't matter because we CAN'T recieve tweets from the next day, since we can't recieve tweets from that day anyways
    id_list = []
    date_list = [] # not needed, but nice to see what's going on
    if debug and set_random_number > 0:
        random_number = set_random_number
    else:
        random_number = random.uniform(0,start_range_max)
    start = date - timedelta(minutes= random_number)
    # print(random_number)
    for i in range(0, int(queries)):
        time = start - timedelta(minutes = constant * i)
        date_list.append(time)
        id_list.append(twtID.tweetId_from_datetime(time))
    if debug:
        return id_list,date_list, random_number
    return id_list, date_list

def time_sampling_constrained(time1,time2, queries, start_range_min = 0):
    if queries == 0:
        return [],[]
    if queries == 1:
        time = time1 + timedelta(seconds=abs(time2 - time1).seconds/2)
        return [twtID.tweetId_from_datetime(time)],[time]
    minutes = -((time2 - time1).seconds // -60)
    start_range_max = minutes/queries
    # now lets try to get some samples from a time interval
    id_list =[]
    date_list = [] # not needed, but nice to see what's going on
    random_number = random.uniform(start_range_min, start_range_max)
    start = time2 - timedelta(minutes= random_number)
    #print(random_number)
    for i in range(0, int(queries)):
        time = start - timedelta(minutes = start_range_max * i)
        date_list.append(time)
        id_list.append(twtID.tweetId_from_datetime(time))
    return id_list, date_list

#given the list of times points to sample (in seconds)
def secList_sampling(df,date,secList,spp=200):
    until_date = twtID.untilDateStr_from_datetimeStr(date) # we first generate the until dates we want w/ random sampling
    since_id = twtID.sinceId_from_untilDate(until_date)
    date = datetime.strptime(date,'%Y-%m-%d') + timedelta(days=0)
    
    twit_ids = [twtID.tweetId_from_datetime(timedelta(seconds = sec) + date) for sec in secList]
    twit_ids = list(reversed(twit_ids)) # reverse it so we are now going from end of the day to the start
    
    dfs = pd.DataFrame([])
    for i in range(len(twit_ids)):
        temp_df = fake_sample(df,twit_ids[i],since_id,spp)
        temp_df['query']=i
        dfs = dfs.append(temp_df)
    return dfs
#%% 

# Actual Computation of velocity from a tweet

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
         # if time diff is 0, then add .0001 s
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

#%% Cleaning scout data

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

#%% Different Velocity Handling to hopefully help stabilize velocity

# just make the new dataframe for boths versions
def new_velocity_helper(velocities,seconds,time):
    scout = pd.DataFrame(velocities,columns=['velocity'])
    scout['seconds'] = seconds
    scout['time'] = time
    return scout

def velocity_pooling(scout,n):
    # pool n sampled velocities together
    velocities =[]
    seconds_list = [] # the seconds the period starts at
    time_list = [] # the timestamp the period starts at
    for i in range(0,len(scout),n):
        subset = scout.iloc[i:(i+n)]
        seconds = subset.tpp.div(subset.velocity) # if we get a zero in a velocity

        # Take the midpoint to be the new timepoint
        if n % 2 == 1: # if odd, we should take the middle time point
            seconds_list.append(subset.iloc[n//2].seconds)
            time_list.append(subset.iloc[n//2].time)
        else: # Otherwise, find the middle by averaging the timestamps
            seconds_list.append(subset.seconds.mean())
            time_list.append(subset.time.mean())

        if seconds.sum() == 0:
            velocities.append(0)
        else:
            velocities.append(subset.tpp.sum()/seconds.sum())
    return new_velocity_helper(velocities,seconds_list,time_list)

def velocity_avg(scout,n):
    # average n sampled velocities together
    velocities =[]
    seconds_list = [] # the seconds the period starts at
    time_list = [] # the timestamp the period starts at
    for i in range(0,len(scout),n):
        subset = scout.iloc[i:(i+n)]
        
        # Take the midpoint to be the new timepoint
        if n % 2 == 1: # if odd, we should take the middle time point
            seconds_list.append(subset.iloc[n//2].seconds)
            time_list.append(subset.iloc[n//2].time)
        else: # Otherwise, find the middle by averaging the timestamps
            seconds_list.append(subset.seconds.mean())
            time_list.append(subset.time.mean())

        velocities.append(subset.velocity.mean())
    return new_velocity_helper(velocities,seconds_list,time_list)


#%% Error

def get_kde_values(df,sample_points=24000):
    # x = np.linspace(0, 86400, num= sample_points)
    x = list(range(0,86400))
    kde = stats.gaussian_kde(df.seconds)
    y = kde.pdf(x)
    return x,y

def bin_density(df,nbins=1000):
    sec_constant = 3600/nbins # get the number of seconds assigned to each bin
    sec_bins = pd.DataFrame([])
    for j in range(0,24):
        sec_bins['bin_'+str(j)]= [j*3600 + sec_constant*i for i in range(0,nbins)]
    labels = [i for i in range(0,nbins-1)] # label each column with it's corresponding i*sec_constant
    df_bins = pd.DataFrame([])
    for j in range(0,24):
        groups = df[df.tbin==j].groupby(['tbin', pd.cut(df[df.tbin==j].seconds, sec_bins['bin_'+str(j)],labels=labels)]).size().unstack()
        df_bins=df_bins.append(groups)
    return df_bins

# to significantly optimize runtime: provide a precomputed y1, since it's constant value
def relative_error(df,dfs,y1=None):
    if y1 is None:
        _, y1 = get_kde_values(df)
    _, y2 = get_kde_values(dfs)
    return np.sqrt((np.square(y1 - y2)).mean(axis=0))

# proportion = True means look at the proportional difference between the 2
def error_by_hour(real_bin,sampled_bin,proportion = False):
    err = []
    errp = []
    for i in range(0,24):
        if not proportion:
            err.append(np.sqrt((np.square(real_bin.iloc[i] - sampled_bin.iloc[i])).mean(axis=0)))
        else:
            rp = np.array([r/sum(real_bin.iloc[i]) for r in real_bin.iloc[i]])
            sp = np.array([s/sum(sampled_bin.iloc[i]) for s in sampled_bin.iloc[i]])
            errp.append(np.sqrt((np.square(rp - sp)).mean(axis=0)))
    return err if not proportion else errp

# returns the root squared mean error between real densitity and sampled density
    # This is the error between the relative area under the curve
def relative_error_bins(real_bin,sampled_bin):
    # fix empty columns
    if not real_bin.shape[0] == sampled_bin.shape[0]:
        missing = list(set(real_bin.index) - set(sampled_bin.index))
        for miss in missing: # np.shape(s_bin)[1]
            s = sampled_bin.xs(list(set(np.arange(0,24)) & set(sampled_bin.index))[0]) # ensure we select valid index
            s.name = miss
            sampled_bin = sampled_bin.append(s)
            sampled_bin.xs(miss)== list(np.zeros(np.shape(sampled_bin)[1]))
        sampled_bin = sampled_bin.sort_index()
    s = np.array(sampled_bin.stack().tolist())
    r = np.array(real_bin.stack().tolist())
    return np.sqrt((np.square(r - s)).mean(axis=0))

#%% PLOTTING

def plot(df, date, min_interval,region, text= True):
    plt.rc('font', size=6)
    li = pd.date_range(date, periods=24*(60/min_interval), freq=str(min_interval)+'Min')
    li = li.append(pd.Index([li[-1]+timedelta(minutes=min_interval)])) 
    
    temp = []
    for i in range(0,len(li)-1):
        temp.append([len(df.loc[(df['tweet_date']>=li[i]) & (df['tweet_date']<li[i+1])]),li[i]])  # greater or equal to time and smaller than largest time
    bucketed = pd.DataFrame(temp)
    x_axis = [x for x in np.arange(0,24,min_interval/60)] 
    plt.figure(figsize=(30,10))
    sns.barplot(x_axis,bucketed[0])
    if text:
        for index, value in enumerate(bucketed[0]): # this is just for the text above the bars
            plt.text(index-.29, value+(value*.006), locale.format_string("%d", value, grouping=True),fontweight='bold') # value = height text is displayed above bar, index controls how far left or right the text should be above the bar
    plt.xlabel(f'Time in {min_interval} minutes intervals')
    plt.ylabel('Number of tweets in interval')
    plt.title(f'Total Tweet Distribution on {date} for {region}, tweet count = {len(df)}')
    plt.show()

def plot_retweets(dfo,dfr,min_interval,region,date,kde=False):
    plt.rc('font', size=8)
    #plt.figure(figsize=(30,10))
    sns.distplot(dfo[['tweet_date']],bins=int(24*(60/min_interval)),kde=kde,label='Original Tweets',color='red')
    sns.distplot(dfr[['tweet_date']],bins=int(24*(60/min_interval)),kde=kde,label='Retweets',color='green')
    plt.legend()    
    plt.xlabel(f'Time in {min_interval} minutes intervals')
    plt.ylabel('Number of tweets in interval')
    plt.title(f'Total Tweet Distribution on {date} for {region}, tweet count = {len(dfo)+len(dfr)}')
    plt.show()

def plot_velocities(data_list,intervals):
    plt.rc('font', size=6)
    plt.figure(figsize=(10,7))
    sns.lineplot(intervals,data_list)
    plt.xlabel('Time')
    plt.ylabel('Tweets per second during interval')
    plt.title('Tweets per second across time')
    plt.show()

# must provide pandas dataframes, not r data frames
def plot_knots(predicted,selected_knots,date,region,legend='spline regression'):
    plt.plot(predicted['x'],predicted['pred'],label=legend)
    for xc in selected_knots:
        plt.axvline(x=xc,linestyle='dashed',linewidth=.5,color='black')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Tweets per second')
    plt.title(f'Velocity cubic regression with {len(selected_knots)} knots for {region} on {date}')
    plt.legend()
    plt.show()

# error_column= 'curve_err' or 'proj_err'
# only doing 3 groups because I'm too lazy to generalize my plotting function
def box_plot_groups(df,name,ngroups,error_column='curve_err'):
    df = pd.DataFrame(df,columns=['proj_err','curve_err'])
    df['spp']=pd.Series(df.index).apply(lambda x: ((x % ngroups)+1)*100)
    groups = list(df.groupby('spp'))
    
    gs = gridspec.GridSpec(2, 2)
    pl.figure().suptitle(name,fontsize=16)

    ax = pl.subplot(gs[0, 0]) # row 0, col 0
    ax.set_title(f'Samples Per Pull: {groups[0][0]}')
    pl.boxplot(groups[0][1].curve_err,vert=False)
    
    ax = pl.subplot(gs[0, 1]) # row 0, col 1
    ax.set_title(f'Samples Per Pull: {groups[1][0]}')
    pl.boxplot(groups[1][1].curve_err,vert=False)
    
    ax = pl.subplot(gs[1, :]) # row 1, span all columns
    ax.set_title(f'Samples Per Pull: {groups[2][0]}')
    pl.boxplot(groups[2][1].curve_err,vert=False)
    
def plot_densities(df_real,df_sampled,interval_min=30,title=None):
    sns.distplot(df_sampled['seconds'], hist=True, kde=True, 
                 bins=int(1440/interval_min), color = 'darkblue', 
                 hist_kws={'edgecolor':'black'},
                 kde_kws={'linewidth': 3}, label='sampled density')
    
    sns.distplot(df_real['seconds'], hist=True, kde=True, 
                 bins=int(1440/interval_min), color = 'darkred', 
                 hist_kws={'edgecolor':'black'},
                 kde_kws={'linewidth': 3}, label='real density')
    plt.legend()
    if title is None:
        plt.title("Real vs Sampled Density")
    else:
        plt.title("Real vs Sampled Density: "+title)
    plt.show()
    
def plot_multiple_days(df,dates,region,min_interval=30):
    days_spanned = len(dates)
    ax = sns.distplot(df.seconds,bins=int(24*(60/min_interval))*days_spanned)
    i=0
    y_point = max(ax.get_lines()[0]._y)
    for date in dates:
        plt.annotate(date, xy=((86400*i)+32000, y_point), xytext=(86400*i+32000, y_point))
        df_length = len(df.loc[(df.seconds>=86400*i) & (df.seconds<86400*(i+1))])
        plt.annotate(int_with_commas(df_length), xy=((86400*i)+32000, y_point*1.05), xytext=(86400*i+32000, y_point*1.05))
        plt.axvline(86400*i, linestyle='dashed', linewidth=1)
        # plt.text(86400*i + 100,.000025,date)
        i+=1
    plt.title(f'Tweet Distribution of {region} over {days_spanned} days')
    plt.show()
        
#%% Printing
def int_with_commas(integer):
    # return a string of an integer with commas for readability
    return ','.join([str(integer)[::-1][i:i+3] for i in range(0,len(str(integer)),3)])[::-1]
def error_printer(df,name,knots=True):
    print(f'{name} Approach:')
    print('Projection Abs Mean Error: ',np.mean(df.proj_err)*100,'%')
    if knots: print('Average Knot Count: ',np.mean(df.knots))
    # print(f'{name}: \n\tPercentage of error > 5%: {len(df[df.error > .05])}/{len(df)}\n\tMean of error > 5%: {(df[df.error > .05].error.mean())*100} %')
    print(f'Curve Mean Error: {df.curve_err.mean()}')
    print(30*'-','\n')
    
def error_printer_groups(df,name,ngroups):
    df = pd.DataFrame(df,columns=['proj_err','curve_err'])
    df['spp']=pd.Series(df.index).apply(lambda x: ((x % ngroups)+1)*100)
    pull_mean = df.groupby('spp').mean()
    for n in pull_mean.index.tolist():
        error_printer(pull_mean.xs(n),name+f' spp = {n}',False)

#%% RPY2 STUFF

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

#%% Working with output files from sampling 

# build a DF for all recent queries 
def build_recent(date, region, size = 720, directory = None):
    file = filename_builder(date,region,directory)
    df = append_queries(file)
    #amount = get_velocity_df(file).pull_amount.values[:size].sum()
    # return typically 3 recent results, but built for supporting more than 3 multiples of 180
    return [df.loc[df.quer <= (size - 180*i) ] for i in range((size // 180)-1)]

# build a DF for all recent queries, but also save it so I dont have to do it anymore
def build_recent_save(date,region,size,directory = None):
    if directory is None:
        directory = sample_directory
    if size in [360,540,720]: size -= 96
    
    recent_df = build_recent(date,region,size,directory)
    recent_dir = directory + '\\' + date + '\\recent'
    if not os.path.isdir(recent_dir):
        os.mkdir(recent_dir)
    file_name = recent_dir + '//all_'+get_region(region)+'_'+date+'_'+str(size)+'.csv.gz'
    recent_df.to_csv(file_name,index=False)    
    return len(recent_df), recent_dir # so we can call this function and build a log file quickly
    

#%% Working with multiple dates

def read_multiple_dates(region,dates,file_loc=corpus_directory,*args,**kwargs):
    li = []
    i=0
    for date in dates:
        df= read_tweets(file_loc+'\\'+date+'\\all_'+region+'_'+date+'.csv.gz',*args,**kwargs)
        df.seconds = df.seconds+86400*i
        li.append(df)
        i+=1
    df = pd.concat(li)
    return df

def generate_dates(start_date,days_spanned):
    # From start_date, generate n dates
    dates =[]
    for i in range(0,days_spanned):
        dates.append((datetime.strptime(start_date,"%Y-%m-%d")+timedelta(days=i)).strftime("%Y-%m-%d"))
    return dates

def read_multiple_dates_regions(region,dates,file_loc=r'F:\Twitter Data\my_tweets',concat = True,*args,**kwargs):
    li = []
    i=0
    for i in range(0,len(dates)):
        df= pd.read_csv(file_loc+'\\'+dates[i]+'\\all_'+region+'_'+dates[i]+'.csv.gz',*args,**kwargs)
        df.seconds = df.seconds+86400*i # This makes the seconds column contiously increase for each date we add
        li.append(df)
        i+=1
    if concat:
        return pd.concat(li)
    return li

# n = number of files to get (n random choices from all the files)
# concat: merge resulting data frames into 1 DF. Pass False if you want to keep the dataframes independent
def random_sample_files(file_loc,n,filt_key = '*all_*.csv.gz',concat=True,read=True,*args,**kwargs):
    li = [name for name in glob(file_loc+'\*\\'+filt_key)] # use *all_*.csv.gz to filter out the no_retweet data
    files = random.sample(li,n)
    dates = pd.Series(files).str[13:23].to_list()
    regions = pd.Series(files).str[28:-18].to_list()
    history = pd.DataFrame(files,columns=['file'])
    history['dates'] = dates
    history['regions'] = regions
    if not read:
        return history
    return read_multiple_dates_regions(regions,dates,file_loc,concat=concat,*args,**kwargs), history # returning li to save history 

#%% VBest

def vbest_test(scout,psu_n=624):
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
    samples= []
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
    vbest = custom_func('VBESTfinalR.R')
    rdf = df2r(scout[['seconds','velocity']])
    res = vbest.VBESTshort(rdf.rx2('seconds'),rdf.rx2('velocity'),Vbestn=psu_n)
    yhats = r2df(res[0])
    span = r2df(res[1])[0]
    total_tweets = r2df(res[2])[0]
    seed = r2df(res[3])[0]
    return yhats,span,total_tweets,seed

def aspline_area(scout,degree=3,k=20):
    func = custom_func('Collection_of_Functions.R')
    rdf = df2r(scout[['seconds','velocity']])
    res=func.Aspline_Area_Proportions(rdf.rx2('seconds'),rdf.rx2('velocity'),degree,k=k)
    if degree == 3:
        return r2df(res[7])[0],r2df(res[8]).reset_index().pred
    return r2df(res[5])[0],r2df(res[6]).reset_index().pred

#%% Sample directory functions

sample_directory = r'E:\Twitter Data\my_samples' # my file location
methods_map = {'inv': 'Inverse', 'inverse':'Inverse','srs':'Srs','vbest':'VBest','vb':'VBest','recent':'recent','uniform':'uniform'}

def set_sample_dir(directory):
    global sample_directory
    sample_directory = directory

def filename_sample_builder(date,region,method,size,directory = None):
    # simplify directory usage
    if directory is None: 
        global sample_directory
        directory = sample_directory
    # simplify method names to easily match their corresponding directory
    global methods_map
    method = method.lower()
    if method in methods_map.keys(): 
        method = methods_map[method]
        if method in ['recent','uniform'] and size in [264,444,624]: size += 96
        elif size in [360,540,720]: size -= 96
    if method == 'recent':
        return directory+'\\'+ date+ '\\' + method + '\\all_'+get_region(region)+'_'+str(size)+'.csv.gz' # recent files dont have dates, whoops
    if method in ['mixed','popular']:
        return directory+'\\'+ date+ '\\' + method + '\\all_'+get_region(region)+'_'+date+'.csv.gz'
    return directory+'\\'+ date+ '\\' + method + '\\all_'+get_region(region)+'_'+date+'_'+str(size)+'.csv.gz'

def filename_model_builder(date,region, directory = None):
    # simplify directory usage
    if directory is None: 
        global sample_directory
        directory = sample_directory
    return directory+'\\'+ date + '\\model_'+get_region(region)+'\\'

def filename_mixed_pop(date,region,method, directory = None):
    if directory is None: 
        global sample_directory
        directory = sample_directory
    return directory+'\\'+ date + '\\'+ method + '\\all_' + get_region(region)+'_'+date+'.csv.gz'

def sample_append_queries(date, region, method, size, directory = None):
    file = filename_sample_builder(date,region,method,size,directory)
    dfv = get_velocity_df(file)
    df = read_tweets(file)
    amounts = dfv.pull_amount.values
    queries = []
    for i in range(len(amounts)):
        queries.extend([i+1]*amounts[i])
    df['quer'] = queries[:len(df)] # can't name the columns 'query' because df.query is a function
    return df

def save_sample_appended_queries(date, region, method, size, directory = None):
    df = sample_append_queries(date, region, method, size, directory)
    file = filename_sample_builder(date,region,method,size,directory)
    ###
    # Insert function to create excel csv file either HERE or return the number of tweets
    ###
    df.to_csv(file,index=False)
    return len(df)

def get_model_volume(date,region, directory = None):
    file = filename_model_builder(date,region,directory)+'integer_meta.csv'
    return pd.read_csv(file).iloc[1].values[0]

def get_model_npsu(date,region,directory = None):
    file = filename_model_builder(date,region,directory)+'integer_meta.csv'
    return pd.read_csv(file).iloc[2].values[0]

def get_model_span(date,region,directory = None):
    file = filename_model_builder(date,region,directory)+'integer_meta.csv'
    return pd.read_csv(file).iloc[0].values[0]

def get_model_inserve_weights(date,region,size,directory = None):
    if size in [264,444,624]: size += 96 # data files labeled as: 1=264, 2 = 444, 3 = 720
    file_number = str((size//180)-1)
    file = filename_model_builder(date,region,directory)+file_number+'_samples_meta.csv'
    return pd.read_csv(file)['inverse_weights']

def get_model_samples_meta(date,region,size,directory = None):
    if size in [264,444,624]: size += 96 # data files labeled as: 1=264, 2 = 444, 3 = 720
    file_number = str((size//180)-1)
    file = filename_model_builder(date,region,directory)+file_number+'_samples_meta.csv'
    return pd.read_csv(file)

# provide df and weights to make this function easier
def sample_append_inverse_weights(df,weights):
    query_sizes = df.quer.value_counts().sort_index()
    w = []
    for i in range(len(query_sizes)):
        w.extend([weights[i]]*query_sizes.iloc[i])
    df['weights'] = w
    return df
    
#%% Meltwater data 

def get_meltwater_volume(date, region, retweets = False, keyword = None,directory=r'E:\Meltwater'):
    # date as string 'year-month-day'
    file = directory + '\\' + date + '\\volume-report\\'
    if retweets:
        if keyword is None or keyword == "":
            file += 'Region_Results.csv'
            meltwater = pd.read_csv(file)
            count = meltwater.loc[meltwater.Region == region]['Total Count']
        else:
            file += 'Region_Keyword_Results.csv'
            meltwater = pd.read_csv(file)
            count = meltwater.loc[meltwater.Region == region][keyword]
    else: # don't include retweets
        if keyword is None or keyword == "":
            file += 'Region_Results_Only_Direct.csv'
            meltwater = pd.read_csv(file)
            count = meltwater.loc[meltwater.Region == region]['Total Count']
        else:
            file += 'Region_Keyword_Results_Only_Direct.csv'
            meltwater = pd.read_csv(file)
            count = meltwater.loc[meltwater.Region == region][keyword]
    return count

