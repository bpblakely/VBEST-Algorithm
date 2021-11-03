import numpy as np
import pandas as pd
import warnings
import time
import re
from nltk.tokenize import TweetTokenizer
from functools import partial
from datetime import timedelta
import velocity_utils as vtils
import tweepy_utils as twpUtil

warnings.filterwarnings("ignore")

regions = ['Atlanta-Sandy Springs-Roswell, GA','Phoenix-Mesa-Scottsdale, AZ','Chicago-Naperville-Elgin, IL-IN-WI','Columbus, OH',
           'Pittsburgh, PA','Baltimore-Columbia-Towson, MD']

city_filters = {'Atlanta-Sandy Springs-Roswell, GA': 'alph|aretta|pharet|atl|lanta|mari|ietta|sand', 
                'Phoenix-Mesa-Scottsdale, AZ': 'Cas|asa|Gra|rand|ande|tem|tp|mpe|scotts|sdale|ttsd|Cha|ler|handle|Mes|esa|pho|phe|eno|phx|phi|enix', 
                'Chicago-Naperville-Elgin, IL-IN-WI': 'boli|ingbr|Chic|cago|wind|Des|keno|sha|keosh|Elgi|gary|sko|skie|umburg|schaum|Evan|van|Hof|ffman|naper', 
                'Columbus, OH': 'col|umb|bus', 
                'Pittsburgh, PA': 'pitts|burgh', 
                'Baltimore-Columbia-Towson, MD': 'balt|imore|blt|columbi|towso|twsn|wson'}
    ## see 'city_filter.py' for how these filters were determined

def filter_city(df,region):
    # returns the filtered dataframe
    return df.loc[df.user_location.str.contains(city_filters[region],regex=True,case=False)>0]

 # An example call of filter_city: filter_city(dataframe, city_filters[regions[0]])
    # city_filters[regions[0]] uses the dictonary to let you provide a region and match it with it's city filter

#%%
# Prepare the text of tweets to be filtered

def remove_url(string):
    return re.sub(r'http\S+', '', string)

def remove_special(string):
    return re.sub('[^A-Za-z0-9]+', ' ', string)

tknzr = TweetTokenizer(strip_handles=True,preserve_case=False) # prepare tokenizer
def prepare_text(string):
    string = ' '.join(tknzr.tokenize(string)) #
    string = remove_url(string) # remove all possible URL's
    string = remove_special(string) # remove all special characters such as #!@%
    # string = string.lower() # convert all text to lowercase, already done with tokenizer
    return string.strip() # remove spaces, if any

#%%
# Mimic Twitters query functionality as closely as possible
def star_match(word,keyword):
    return word.startswith(keyword) # stars with, so now we dont get random substring 
def full_match(word,keyword):
    return word == keyword

# use any() to use lazy evaluation
def full_search(string,keyword):
    return any((full_match(w,keyword) for w in string.split(' ')))
def star_search(string,keyword):
    return any((star_match(w,keyword) for w in string.split(' ')))

def phrase_search(string,phrase):
    # Accurate, but slow
    expression = re.compile(r'\b'+ r' '.join([('(' + k +')').replace('*)',')[a-zA-Z]*') for k in phrase.split(' ')]) + r'\b')
    return bool(expression.search(string))

    # Easier to read code
    # r = []
    # for k in "work* from home".split(' '):
    #     temp = '(' + k +')' # (work*)
    #     temp = temp.replace('*)',')[a-zA-Z]*') # (work)[a-zA-Z]*
    #     r.append(temp)
    # rr = r'\b'+ r' '.join(r) + r'\b'  # r'\b(work.*) (from) (home)\b'
    
    # re.compile(rr).search('I hate working from home') # True
    # re.compile(rr).search('work from home') # True
    # re.compile(rr).search('from home worker to president') # False
    
    
# phrase_search('i am hunkering down out here in alaska',"hunker* down")
# phrase_search('just got my stimulus checks its great','stimulus check*')

# near matching for: keyword1 near/n keyword2
def near_match(string,keyword1,keyword2,n):
    k1 = np.array([])
    k2 = np.array([])
    # handle matching
    if keyword1.find('*') >=0:
        k1check = star_match
        keyword1 = keyword1.replace('*','')
    else:
        k1check = full_match
    if keyword2.find('*') >=0:
        k2check = star_match
        keyword2 = keyword2.replace('*','')
    else:
        k2check = full_match   
    
    # construct word position lists
    for i,w in enumerate(string.split(' ')):
        if k1check(w,keyword1):
            k1 = np.append(k1,i) # track positions where keyword1 is true in the string
        elif k2check(w,keyword2):
            k2 = np.append(k2,i) # track positions where keyword2 is true in the string
    # compute distance matrix
    distance = np.abs(k1[:, np.newaxis] - k2)
    # return True if any distance in the distance matrix is less than or equal to n
    return np.any(distance <= n)

# split query into multiple filters by building a list of partial functions 
# When using the list of filters, use any() to ensure lazy evaluation
def build_filters(filter_string):
    filt = filter_string.lower().replace('"','').split(' or ') # remove quotations and split on different conditions
    filt_list = []
    # go through each filter and build a PARTIAL function for the corresponding logic to be used for filtering
    for f in filt:
        f = f.strip() # remove unwanted spaces 
        if "near/" in f: # near follows the format of: "(keyword1 near/n keyword2)"
            f0 = f.replace('(','').replace(')','').split(' ') # remove parathensis
            n = int(f0[1][-1])
            keyword1 = f0[0]
            keyword2 = f0[-1]
            filt_list.append(partial(near_match,keyword1 = keyword1, keyword2 = keyword2, n=n))
        elif f.find('*') >= 0:
            if len(f.split()) == 1:
                f = f.replace('*','')
                filt_list.append(partial(star_search,keyword=f))
            else:
                filt_list.append(partial(phrase_search,phrase = f))
        elif len(f.split()) > 1:
            filt_list.append(partial(phrase_search,phrase=f))
        else:
            filt_list.append(partial(full_search,keyword=f))
    return filt_list

# [func("I hate covid") for func in build_filters(""""covid*" OR "coronavirus" OR "rona" or "corona virus" or "cv19" """)]


#%%

# Query strings taken from Meltwater verbatim, BUT I optimize them for speed (getting rid of redundencies)

    # covid_string1 = """"covid" OR "covid-19" OR "covid19" OR "covid testing" OR "covid cases" OR "coronavirus" OR "rona" or "covid*" or "corona virus" or "cv19" """
    # general_string1 = """"virus" OR "flu" OR "pandemic" OR "sars" OR "pneumonia" OR "Fauci" """
    # mask_string1 = """"Face Mask*" OR "Mask*" OR "PPE" OR "N95" OR "Face Covering" OR "Face Shield" or (face near/3 cover*) or (face near/3 mask*)"""
    # sanitize_string1 = """"hand sanitizer" OR "disinfect" OR "disinfectant*" OR "lysol" OR "sanitize*" OR "sanitizing" OR "sanitizer" OR "hand wash" OR "hand washing" OR "bleach" or (hand near/3 wash*) or (hands near/3 wash*) or (hand near/3 sanitiz*) or (hands near/3 sanitiz*) or clorox"""
    # distancing_string1 = """"Social Distance" OR "Social Distancing" OR "Six feet apart" OR "6 ft apart" OR "6 feet apart" OR "hunker* down" OR "lockdown" OR "quarantine" OR "quarantining" or "lock down" or "quarant*" or "social distanc*" """
    # symp_string1 = """"Can't smell" OR "No Smell" OR "Can't taste" OR "No Taste" OR "Cough" OR "Fever" OR "Chills" OR "sore throat" OR "asymptomatic" or (loss near/3 smell) or (loss near/3 taste) or (sore near/5 throat) or  (lost near/3 smell) or (lost near/3 taste)"""
    # tests_string1 = """"antigen*" OR "antibodies" OR "seroprevalence" or "antibody*" """
    # treat_string1 = """"ventilator*" OR "remdesivir" OR "vaccine" OR "contact tracing" """
    # work_string1 = """"wfh" OR "working from home" OR "work* from home" OR "not working now" OR "furlough" OR "reopen*" OR "reopening" OR "stimulus check*" OR "remote work" OR "working remotely" OR "unemployed" or (work* near/3 home)"""

    # this sanitize_string is closer to what we care about and will include slightly more tweets than meltwater, but run WAY faster
    #sanitize_string = """"disinfect*" OR "sanitiz*" OR "bleach" or (hand near/3 wash*) or (hands near/3 wash*) or "clorox" OR "lysol" """

# Optimized to avoid redundent queries

covid_string = """"covid*" OR "coronavirus" OR "rona" or "corona virus" or "cv19" """
general_string = """"virus" OR "flu" OR "pandemic" OR "Fauci" OR "pneumonia" OR "sars" """
mask_string = """"Mask*" OR "PPE" OR "N95" OR "Face Shield" OR (face near/3 cover*)"""
sanitize_string = """"disinfect" OR "disinfectant*" OR "lysol" OR "sanitize*" OR "sanitizing" OR "bleach" or "clorox" or (hands near/3 wash*) or (hand near/3 wash*) or (hand near/3 sanitiz*) or (hands near/3 sanitiz*)"""
distancing_string = """"quarant*" OR "social distanc*" OR "lockdown" or "lock down" OR"hunker* down" OR "6 ft apart" OR "6 feet apart" OR "Six feet apart" """
symp_string = """"Fever" OR "Cough" OR "Chills" OR "asymptomatic" OR "Cant smell" OR "No Smell" OR "Cant taste" OR "No Taste" OR  (loss near/3 smell) or (loss near/3 taste) or (sore near/5 throat) or (lost near/3 smell) or (lost near/3 taste)"""
tests_string = """"antigen*" or "antibody*" OR "antibodies" OR "seroprevalence" """
treat_string = """"vaccine" OR "ventilator*" OR "contact tracing" OR "remdesivir" """
work_string = """"unemployed" OR "stimulus check*" OR "reopen*" OR (work* near/3 home) OR "wfh" OR "furlough" OR "remote work" OR "working remotely" OR "not working now" """


queries = [covid_string,general_string,mask_string,tests_string,treat_string,sanitize_string,distancing_string,symp_string,work_string]
names = ['covid','general_virus','masks','test','treatments','sanitizing','social_distancing','symptoms','working']

# buil names for other categories
proportion = []
corpus_proportion = []
for name in names:
    proportion.append(name+'P')
    corpus_proportion.append(name+'CP')

df_name_order = ['region','date','method','targetQuery','actualQuery','queryID','span','probSel','npsus','totTweets','totTweetsEst','totTweetsFilt',
                     'count_original','count'] + names + proportion + corpus_proportion

def preprocess_corpus(df,region, city_filt = True):
    if city_filt:
            df = filter_city(df,region).reset_index(drop=True)
    df['scrubbed_text'] = df['full_text'].apply(prepare_text)
    df = df.drop(['full_text','tweet_date','user_location','seconds'],axis=1)
    return df

# set preprocess = True if you havent filtered city
# Verbose used to test computation efficiency
def do_filter(df, preprocess = False, region = None,  city_filt = True, verbose=True):
    j = 0 # see comment below as to why this is here
    
    def evaluate(string):
        # same thing as lazy evaluation
        for p in build_filters(queries[j]):
            if p(string):
                return 1 
        return 0
    
    start = time.monotonic()
    
    if preprocess:
        if region is None and city_filt: 
            print("Region can't be None if you are going to filter the cities")
            return -1
        df = preprocess_corpus(df,region,city_filt)
    
    for i in range(len(names)):
        j = i # not sure if I should be able to refer to this i in the evaluate() function, so I'm being safe
        start_time = time.monotonic()
        df[names[i]] = df.scrubbed_text.apply(evaluate)
        if verbose: 
            print(names[i], timedelta(seconds=time.monotonic() - start_time))
        
    if verbose: 
        print("total filtering time", timedelta(seconds=time.monotonic() - start))
    
    return df.drop(['scrubbed_text'],axis=1)



# this code currently INCLUDES the changed user locations
def efficient_filter(sdf,df,region):
    #start_time = time.monotonic()
    sdf = filter_city(sdf,region).reset_index(drop=True) # filter all tweets by city, this is cheap
    extra = sdf[~sdf.tweet_id.isin(df.tweet_id)] # remove already filtered keywords
    extra = do_filter(extra, preprocess = True, city_filt = False,verbose = False) # do the filtering, no need to filter by city
    
    sdf = sdf[['tweet_id','quer']].merge(df.drop(['quer'],axis=1),on='tweet_id') # merge sdf with df, but keep sdf's queries
    sdf = sdf.append(extra) # add the extra data
    #print('total filtering time (efficient)',timedelta(seconds=time.monotonic() - start_time))
    return sdf

# check if any queries are missing and return a dataframe with them appended on and the tweet_id dropped
def fill_missing_queries(final_out, size):
    missing_queries = list(set([i+1 for i in range(size)]) - set(final_out.quer))
    if len(list(missing_queries)) == 0:
        return final_out
    li = []
    for query in missing_queries:
        li.append([query]+[0]*9) # I think we the data has 9 columns, so 9 is used to create a list of len = 9 with 0's
    missing = pd.DataFrame(li,columns=['quer']+names)
    if 'tweet_id' in final_out.columns:
        return final_out.append(missing) 
    return final_out.append(missing)

#%%  This is the final code used to filter the text from a sample of tweets

# Efficient Filtering steps

    # 1. Filter main corpus like I have been, no need for location filter
        # This lets us process the text in all the corpus tweets (which we need to compute anyways)
        
    # 2. Filter all cities from the sample file, this is a small computation, so its not a big deal
    
    # 3. Now only consider the tweet_ids not in the main corpus filtered and run the filtering on those
        # Avoids processing tweets we have already processed from Step 1
            # By comparing tweet IDs
        
    # 4. Combine the extra tweets with the already filtered ids in the main corpus

# sdf = sampled tweets data frame for specific date and location
# df = full corpus of tweets for the same date and location as sdf

def filter_sample(sdf,df,date,region,method,size,static):
    tweets_per_query_orig = sdf[['quer','tweet_id']].groupby(['quer']).count().rename(columns = {'tweet_id':'count'})
    
    if len(sdf) == 0: # if we didnt obtain any tweest from a query (happens with 'Popular')
        queries_used = 1 # we must have used ATLEAST 1 query to check if there was any information there
        start_times = -1
        end_times = 0
    else:
        queries_used = sdf.quer.iloc[-1]
        # manipulate dataframes to get the start and end times of each query in seconds.ms
        start_times = sdf[['quer','tweet_id']].groupby(['quer']).first().tweet_id.apply(twpUtil.datetime_from_tweetId).apply(vtils.dt2sec)
        end_times = sdf[['quer','tweet_id']].groupby(['quer']).last().tweet_id.apply(twpUtil.datetime_from_tweetId).apply(vtils.dt2sec)
    
    if method == 'recent' and size in [264,444,624]: size += 96 
    final_out = efficient_filter(sdf,df,region)
        
    tweets_per_query = final_out[['quer','tweet_id']].groupby(['quer']).count().rename(columns = {'tweet_id':'count'})
    
    final_out = fill_missing_queries(final_out,size)
    
    final_out = final_out.assign(duplicates = final_out.tweet_id.duplicated().astype(int)) # duplicates
    
    if 'tweet_id' in final_out.columns:
        final_out = final_out.drop('tweet_id',axis=1)
    final_out = final_out.astype(int)
    
    if method in ['popular','mixed']:
        by_query = final_out.groupby(['quer']).sum() # we use 1 query to check if there is any data there anyways
        by_query['targetQuery'] = -1
        by_query['actualQuery'] = queries_used
    else:
        by_query = final_out.groupby(['quer']).sum()
        by_query['targetQuery'] = size
        by_query['actualQuery'] = queries_used
    by_query['start'] = start_times
    by_query['duration'] = start_times - end_times
    for key,value in static.items():
        by_query[key] = value
    by_query['count'] = tweets_per_query
    by_query['count_original'] = tweets_per_query_orig
    by_query['method'] = method
    if method == 'Inverse':
        by_query['probSel'] = 1/vtils.get_model_inserve_weights(date,region,size)
    else:
        by_query['probSel'] = -1

    by_query['queryID'] = by_query.index
    
    by_query[proportion] = by_query[names].div(by_query['count'],axis=0)
    
    # totTweets is the number of tweets BEFORE city filtering. This is so we can have a better comparison of Loess Estimate
    # totTweetsFilt is number of tweets AFTER city filtering. This number is used in the corpus proportions calculations
    # count_original = total tweets pulled in the query
    # count = total tweets in the query AFTER filtering by region
    by_query = by_query[['region','date','method','targetQuery','actualQuery','queryID','span','probSel','npsus','totTweets','totTweetsEst','totTweetsFilt',
                          'start','duration','count_original','count','duplicates']+ names + proportion + corpus_proportion]
    # by_query = by_query[['region','date','method','targetQuery','actualQuery','queryID','span','probSel','npsus','totTweets','totTweetsEst','totTweetsFilt',
    #                      'start','duration','count_original','count']+ names + proportion + corpus_proportion]
    by_query = by_query.reset_index(drop=True) # reset the index so we can make this a valid, appendable DF
    return by_query

