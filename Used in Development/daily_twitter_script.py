import tweepy_utils as twpUtil
import tweetId_functions as twtID
import os
import numpy as np
import pandas as pd
import tweepy as tw
from datetime import datetime, timedelta

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

inputs = r'locations.txt'
output_folder = r'E:\\my_tweets'
locations = extract_locations_from_txt(inputs)

accounts = twpUtil.accounts

total_tweet_count = 0

date = (datetime.now().date()-timedelta(days=2)).strftime("%Y-%m-%d")

until_date = twtID.untilDateStr_from_datetimeStr(date) # we first generate the until dates we want
since_id = twtID.sinceId_from_untilDate(until_date)
s = datetime.now() # used to track how long the process took
pull_count = []
velocity = []
time_start = []
time_end = []

keyword = "-filter:retweets"
# keyword = ""

os.mkdir(output_folder+'//'+date)
os.mkdir(output_folder+'//'+date+'//velocity')

logs = []
ii=0
region_counter = 1
for region,circles in locations.values:
    circle = circles[0]
    data = pd.DataFrame([])
    i = 0
    max_id = twtID.sinceId_from_untilDate((datetime.strptime(date,'%Y-%m-%d') + timedelta(days=2)).date().strftime("%Y-%m-%d")) # last ID from the first run of this program
    s2 = datetime.now()
    while max_id >= since_id:  #initial max = next days ID, then go until the last tweet pulled has a smaller tweet ID than the smallest ID possible
        twitter = twpUtil.init((ii // 180) % len(accounts)) # use our accounts intelligently (only kind of), dont wait
        
        if i % 180==0:
            s2=datetime.now()
            
        try:
            tweets = tw.Cursor(twitter.search, q=keyword,lang='en',tweet_mode='extended', count = 100, max_id=max_id-1,
                     result_type ='recent',until=until_date, geocode=circle).pages(1)
            temp_df = twpUtil.extract_tweets(list(next(tweets)), region)
            
        except StopIteration:
            print("No more tweets after ",temp_df.iloc[-1].tweet_date)
            break
        
        except: 
            pass
        
        # If we only get 1 tweet, computing the velocity will break the code, so just add it and move on
        if len(temp_df)<=1:
            data = data.append(temp_df)
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
    correct = correct.drop_duplicates('tweet_id')
    correct.to_csv(output_folder+'//'+date+'//all_'+region+'_'+date+'.csv.gz',index=False)
    
    print(f"Completion Time: {datetime.now()-s} | Queries Used: {i} | Number of tweets collected: {len(data)} | Region: {region}")
    print(f"Pulls per Query: {np.mean(pull_count)}")
    
    velocities = pd.DataFrame(data=list(reversed(time_start)),columns=['time'])
    velocities['velocity'] = list(reversed(velocity))
    velocities['pull_amount'] = list(reversed(pull_count))
    velocities.to_csv(output_folder+'//'+date+'//velocity//velocity_'+region+'_'+date+'.csv',index=False)
    
    total_tweet_count += len(data)
    logs.append([region,len(data)])
    
    region_counter += 1
    
    del correct
    
runtime = datetime.now()-s
print(f'Total queries used: {ii}')
print(f'Total tweets collected: {total_tweet_count}')
with open(output_folder+'//'+date+'//log.txt','w+') as f:
    f.write(f'Total Tweet Count: {total_tweet_count}\ntotal run time: {runtime}\n\n')
    f.write('Region: number_of_tweets\n')
    for i in range(0,len(logs)):
        f.write(f'{logs[i][0]}: {logs[i][1]}\n')
        