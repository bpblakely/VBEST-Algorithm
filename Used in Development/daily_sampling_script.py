from tweepy_utils import *
from tweetId_functions import *
from velocity_utils import *
import os
import numpy as np
import pandas as pd
import tweepy as tw
from datetime import datetime, timedelta
from text_cell import text_my_cell
import brian_utils as btils
import time
import random

output_folder = r'E:\\my_samples'
inputs = r'C:\Users\bpb92\Documents\Python Scripts\locations.txt'
ii=0
locations = extract_locations_from_txt(inputs)
#locations = locations[-2:]

def sample_naive(sample_name,query_limits,date=None,result_type='recent'):
    global ii
    
    if date is None:
        date = (datetime.now().date()-timedelta(days=2)).strftime("%Y-%m-%d")
    
    until_date = untilDateStr_from_datetimeStr(date) # we first generate the until dates we want w/ random sampling
    s = datetime.now()
    total_tweet_count = 0
    pull_count = []
    velocity = []
    time_start = []
    time_end = []
    time_per_acc = []
    keyword = "-filter:retweets"
    # keyword = ""
    since_id = sinceId_from_untilDate(until_date)
    
    if not os.path.isdir(output_folder+'//'+date):
        os.mkdir(output_folder+'//'+date)
    os.mkdir(output_folder+'//'+date+'//'+sample_name)
    os.mkdir(output_folder+'//'+date+'//'+sample_name+'//velocity')
    
    logs = []
    region_counter = 1
    for region,circles in locations.values:
        circle = circles[0]
        data = pd.DataFrame([])
        i = 0
        max_id = sinceId_from_untilDate((datetime.strptime(date,'%Y-%m-%d') + timedelta(days=2)).date().strftime("%Y-%m-%d")) # last ID from the first run of this program
        s2=datetime.now()
        
        while i < query_limits:
            twitter = init((ii // 180) % len(accounts)) # use our accounts intelligently (only kind of), dont wait
            if i%180==0:
                time_per_acc.append((datetime.now()-s2).seconds/60)
                s2=datetime.now()
            try:
                if result_type == "popular":
                    tweets = tw.Cursor(twitter.search, q=keyword,lang='en',tweet_mode='extended', count = 100, max_id=max_id-1,
                         since = since_id, result_type = result_type, until=until_date, geocode=circle).items(query_limits*100)
                    temp_df = extract_tweets(tweets, region)
                else:
                    tweets = tw.Cursor(twitter.search, q=keyword,lang='en',tweet_mode='extended', count = 100, max_id=max_id-1,
                              since = since_id, result_type = result_type, until=until_date, geocode=circle).pages(1)
                    temp_df = extract_tweets(list(next(tweets)), region)
            except StopIteration:                    
                print("No more tweets after ",temp_df.iloc[-1].tweet_date)
                data = data.append(temp_df)
                pull_count.append(len(temp_df))
                velocity.append(0) # cannot compute velocity with only 1 tweet
                time_start.append(0)
                i+=1
                ii+=1
                break
            except: 
                i+=1
                ii+=1
                pass
            if len(temp_df)<=1:
                data = data.append(temp_df)
                pull_count.append(len(temp_df))
                velocity.append(0) # cannot compute velocity with only 1 tweet
                if len(temp_df)==1:
                    time_start.append(temp_df.tweet_date.iloc[0])
                else:
                    time_start.append(0)
                i+=1
                ii+=1
                break
            pull_count.append(len(temp_df))
            delta_t = abs(temp_df.tweet_date.iloc[0]-temp_df.tweet_date.iloc[-1]).seconds
            if delta_t == 0:
                delta_t = .001 # if zero say it's 1 milisecond
            velocity.append(len(temp_df)/delta_t)
            time_start.append(temp_df.tweet_date.iloc[0])
            time_end.append(temp_df.tweet_date.iloc[-1])
            print(f"{i} | {len(temp_df)} | {str(temp_df.iloc[-1].tweet_date)} | Region: {region}: {region_counter}/{len(locations)}")
            
            max_id = temp_df.iloc[-1].tweet_id
            data = data.append(temp_df)
            i+=1
            ii+=1
        
        correct = data.loc[data['tweet_date'] >= datetime.strptime(date+' 00:00:00','%Y-%m-%d %H:%M:%S')]
        over_count = len(data)-len(correct) # track number of tweets we pull from the prior day
        correct = correct.drop_duplicates('tweet_id')
        correct.to_csv(output_folder+'//'+date+'//'+sample_name+'//all_'+region+'_'+date+'.csv.gz',index=False)
        print(f"Completion Time: {datetime.now()-s} | Queries Used: {i+1} | Number of tweets collected: {len(data)} | Region: {region}")
        print(f"Pulls per Query: {np.mean(pull_count)}")
        print(f"avg time spent collecting 180 queries: {np.mean(time_per_acc)}")
        velocities = pd.DataFrame(data=list(reversed(time_start)),columns=['time'])
        velocities['velocity'] = list(reversed(velocity))
        velocities['pull_amount'] = list(reversed(pull_count))
        velocities.to_csv(output_folder+'//'+date+'//'+sample_name+'//velocity//velocity_'+region+'_'+date+'.csv',index=False)
        total_tweet_count += len(data)
        logs.append([region,len(data)])
        region_counter += 1
        pull_count = []
        velocity = []
        time_start = []
        time_end = []
        del correct
        with open(output_folder+'//'+date+'//'+sample_name+'//velocity//overCount_'+region+'_'+date+'.txt','w+') as f:
            f.write(f'{over_count}')
        
    runtime = datetime.now()-s
    print(f'Total queries used: {ii}')
    print(f'Total tweets collected: {total_tweet_count}')
   # text_my_cell(f'{sample_name} finished collecting tweets from {date} with a total of {btils.int_comma(total_tweet_count)} tweets collected\ntotal run time: {runtime}')
    with open(output_folder+'//'+date+'//'+sample_name+'//log.txt','w+') as f:
        f.write(f'{sample_name}\n')
        f.write(f'total tweet count: {total_tweet_count}\ntotal run time: {runtime}\n\n')
        f.write('Region: number_of_tweets\n')
        for i in range(0,len(logs)):
            f.write(f'{logs[i][0]}: {logs[i][1]}\n')

def sec2ids(date,time_list,sample_name,size=""):
    datet = datetime.strptime(date,'%Y-%m-%d') + timedelta(days=0)
    sample_ids = [tweetId_from_datetime(timedelta(seconds = sec) + datet) for sec in time_list]
    df_logger = pd.DataFrame(time_list,columns=['seconds'])
    df_logger['max_id'] = sample_ids
    df_logger.to_csv(output_folder+'//'+date+'//'+sample_name+'//'+size+'_psus_selected.csv')
    return sample_ids

def uniform_ids(date, query_limits):
    uIds,utimes = time_sampling(date,query_limits)
    time_list = [dt2sec(t) for t in utimes]
    return sec2ids(date,time_list,'uniform')

def regional_set(region,circle,date,sample_name,query_limits,time_list=None,keyword="-filter:retweets",psu_size = "0"):
    global ii
    until_date = untilDateStr_from_datetimeStr(date) # we first generate the until dates we want w/ random sampling
    since_id = sinceId_from_untilDate(until_date)
    pull_count = []
    velocity = []
    time_start = []
    time_end = []
    data = pd.DataFrame([])
    s = datetime.now()
    if sample_name == 'uniform': # default to uniform sampling
        sample_ids = uniform_ids(date,query_limits)
    else: # This is the case of the other 3 sampling algorithms
        sample_ids = sec2ids(date,time_list,sample_name,psu_size)
    query_limits = len(sample_ids)
    for i in range(query_limits):
        twitter = init((ii // 180) % len(accounts)) # use our accounts intelligently (only kind of), dont wait
        #temp_df=[]
        try:
            temp_df = real_sample(twitter, circle, region, until_date, sample_ids[i], since_id, keyword= keyword, spp=100)
            if len(temp_df)<=1:
                data = data.append(temp_df)
                break
        except StopIteration:
            print("No more tweets after ",temp_df.iloc[-1].tweet_date)
        except: 
            pass
        pull_count.append(len(temp_df))
        delta_t = abs(temp_df.tweet_date.iloc[0]-temp_df.tweet_date.iloc[-1]).seconds
        if delta_t == 0:
            delta_t = .001 # if zero say it's 1 milisecond
        velocity.append(len(temp_df)/delta_t)
        time_start.append(temp_df.tweet_date.iloc[0])
        time_end.append(temp_df.tweet_date.iloc[-1])
        print(f"{i} | {len(temp_df)} | {str(temp_df.iloc[-1].tweet_date)} | Region: {region}")
        
        data = data.append(temp_df)
        ii+=1
    
    correct = data.loc[data['tweet_date'] >= datetime.strptime(date+' 00:00:00','%Y-%m-%d %H:%M:%S')]
    over_count = len(data)-len(correct) # track number of tweets we pull from the prior day
    correct = correct.drop_duplicates('tweet_id')
    correct.to_csv(output_folder+'//'+date+'//'+sample_name+'//all_'+region+'_'+date+'_'+psu_size+'.csv.gz',index=False)
    print(f"Completion Time: {datetime.now()-s} | Queries Used: {i} | Number of tweets collected: {len(data)} | Region: {region}")
    print(f"Pulls per Query: {np.mean(pull_count)}")
    velocities = pd.DataFrame(data=list(reversed(time_start)),columns=['time'])
    velocities['velocity'] = list(reversed(velocity))
    velocities['pull_amount'] = list(reversed(pull_count))
    file_info = output_folder+'//'+date+'//'+sample_name+'//velocity//velocity_'+region+'_'+date 
    velocities.to_csv(file_info+'_'+psu_size+'.csv',index=False)
    with open(output_folder+'//'+date+'//'+sample_name+'//velocity//overCount_'+region+'_'+date+'_'+psu_size+'.txt','w+') as f:
        f.write(f'{over_count}')
    return len(data),[region,len(data)],ii

def make_directories(sample_names,date):
    for name in sample_names:
        if not os.path.isdir(output_folder+'//'+date+'//'+name):
            os.mkdir(output_folder+'//'+date+'//'+name)
            os.mkdir(output_folder+'//'+date+'//'+name+'//velocity')

#  this file structure kind of sucks and I'm too burnt out to improve it
def output_info(region,date,sample,psu,span,total_tweets,npsu,size,seeds,scout,seed):
    loc = output_folder+'//'+date+'//'+"model_"+region
    if not os.path.isdir(loc):
        os.mkdir(loc)
    for i in range(3):
        sample[i].to_csv(loc+'//'+str(3-i)+'_samples_meta.csv',index=False)
    psu.to_csv(loc+'//psu_meta.csv',index=False)
    li = [span[0],total_tweets,npsu,size[0],seed]
    a=pd.DataFrame(li,columns=['span,total_tweets,npsu,size,scout_seed'])
    a.to_csv(loc+'//integer_meta.csv',index=False)
    a=pd.DataFrame(seeds,columns=['seeds'])
    a.to_csv(loc+'//seeds_meta.csv',index=False)
    scout.to_csv(loc+'//scout.csv',index=False)
    
    
def sample2(sample_name,query_limits,date=None,spp=200,start_queries=96,result_type='recent'):
    global ii 
    if date is None:
        date = (datetime.now().date()-timedelta(days=2)).strftime("%Y-%m-%d")
        
    until_date = untilDateStr_from_datetimeStr(date) # we first generate the until dates we want w/ random sampling
    since_id = sinceId_from_untilDate(until_date)
    s = datetime.now()
    total_tweet_count = 0
    keyword = "-filter:retweets"
    # keyword = ""
    
    if not os.path.isdir(output_folder+'//'+date):
            os.mkdir(output_folder+'//'+date)
            
    if sample_name == 'uniform':
        if not os.path.isdir(output_folder+'//'+date+'//'+sample_name):
            os.mkdir(output_folder+'//'+date+'//'+sample_name)
            os.mkdir(output_folder+'//'+date+'//'+sample_name+'//velocity')
    else:
        names = ['VBest','Srs','Inverse']
        make_directories(names,date)
    logs = []
    logs_mult = {'VBest':[],'Srs':[],'Inverse':[]}
    total_tweets_mult = {'VBest':[],'Srs':[],'Inverse':[]}
    sizes=[]
    for region,circles in locations.values:
        circle = circles[0]
        if sample_name == 'vbest_test':
            twitter = init((ii // 180) % len(accounts))
            seed = int(random.uniform(1,1000000)) # get a random seed
            random.seed(seed)
            max_range = 1440/start_queries//(spp//100)
            # min RNG value must be set, since if we get -30 minutes, then our program will fail (start collecting tweets at the first second of the day)
            sample_intervals, time_intervals = time_sampling(date,queries = start_queries//(spp//100),start_range_max= max_range-2.5)
            #print(time_intervals[0])
            # create parameter to return _ (the dataframe containing tweets obtained from sampling)
            scout,_,_= velocity_cleaner_real(twitter,circle,region,date,sample_intervals,time_intervals,spp, query_limits,ii)
            ii += 96
            sample,psu,span,total_tweets,npsu,size,seeds = vbest_test(scout, query_limits-start_queries)
            output_info(region,date,sample,psu,span,total_tweets,npsu,size,seeds,scout,seed)
            sizes.append(size[0])
            for j in range(0,3): # 0 = 720, 1 = 540, 2 =  360
                init_queries = query_limits - 180*j
                count,log,ii = regional_set(region,circle,date,'VBest',init_queries,time_list=sample[j]['vsample'],keyword=keyword,psu_size=str(init_queries-start_queries))
                total_tweets_mult['VBest'].append(count)
                logs_mult['VBest'].append(log)
                
                count,log,ii = regional_set(region,circle,date,'Srs',init_queries,time_list=sample[j]['ssample'],keyword=keyword,psu_size=str(init_queries-start_queries))
                total_tweets_mult['Srs'].append(count)
                logs_mult['Srs'].append(log)
                
                count,log,ii = regional_set(region,circle,date,'Inverse',init_queries,time_list=sample[j]['isample'],keyword=keyword,psu_size=str(init_queries-start_queries))
                total_tweets_mult['Inverse'].append(count)
                logs_mult['Inverse'].append(log)
            
        else:
            count,log,ii = regional_set(region,circle,date,sample_name,query_limits,keyword=keyword,psu_size=str(query_limits))
            total_tweet_count += count
            logs.append(log)
        ii += 1
    if sample_name == 'uniform':
        runtime = datetime.now()-s
        print(f'Total queries used: {ii}')
        print(f'Total tweets collected: {total_tweet_count}')
        #text_my_cell(f'"{sample_name}" finished collecting tweets from {date} with a total of {btils.int_comma(total_tweet_count)} tweets collected\ntotal run time: {runtime}')
        with open(output_folder+'//'+date+'//'+sample_name+'//log.txt','w+') as f:
            f.write(f'{sample_name}\n')
            f.write(f'total tweet count: {total_tweet_count}\ntotal run time: {runtime}\n\n')
            f.write('Region: number_of_tweets\n')
            for i in range(0,len(logs)):
                f.write(f'{logs[i][0]}: {logs[i][1]}\n')
    else:
        runtime = datetime.now()-s
        for z in range(0,3): # alg name
            sample_name = names[z]
            for za in range(0,3): # 3 different runs
                print(f'Total tweets collected: {sum(total_tweets_mult[sample_name][za::3])}')
                #text_my_cell(f'"{sample_name}" finished collecting tweets from {date} with a total of {btils.int_comma(sum(total_tweets_mult[sample_name][za::3]))} tweets collected\ntotal run time: {runtime}')
                with open(output_folder+'//'+date+'//'+sample_name+'//'+str(query_limits - 180*za)+'_log.txt','w+') as f:
                    f.write(f'{sample_name}\n')
                    f.write(f'total tweet count: {sum(total_tweets_mult[sample_name][za::3])}\ntotal run time: {runtime}\n\n')
                    f.write('Region: number_of_tweets\n')
                    logs = logs_mult[names[z]][za::3]
                    for i in range(0,len(logs)):
                        f.write(f'{logs[i][0]}: {logs[i][1]}\n')
        text_my_cell(f'"Multiple Algorithms finished collecting tweets from {date} with total run time: {runtime}')

#%% sampling normally
ii=0
initial_start = datetime.now()
query_limits = 720
date = (datetime.now().date()-timedelta(days=6)).strftime("%Y-%m-%d")

#date = "2020-11-28"
sample_naive('popular',query_limits,date = date,result_type = 'popular')
sample_naive('mixed',query_limits,date = date,result_type = 'mixed')

sample2('uniform',query_limits,date = date,spp=100)
sample2('uniform',query_limits-180,date = date,spp=100)
sample2('uniform',query_limits-360,date = date,spp=100)

sample2('vbest_test',query_limits,date = date,spp=200,start_queries=96)

final_end = datetime.now()

text_my_cell(f'"Sampling Script from {date} with total run time: {final_end-initial_start}\n Total Queries Used: {ii}',0)
