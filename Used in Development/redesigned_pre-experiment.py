import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from velocity_utils import *
from tweepy_utils import *
import ray
import os
from pathlib import Path
import random
from functools import partial
import rpy2
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import STAP
import brian_utils as btils

def old_aspline(scout,degree=3,k=20,folds=5):
    func = custom_func('Collection_of_Functions.R')
    rdf = df2r(scout[['seconds','velocity']])
    res=func.Aspline_Area_Proportions(rdf.rx2('seconds'),rdf.rx2('velocity'),degree,k=k)
    if degree == 3:
        return r2df(res[7])[0],r2df(res[8]).reset_index().pred,int(len(r2df(res[1])))
    return r2df(res[5])[0],r2df(res[6]).reset_index().pred,int(len(r2df(res[1])))

def aspline_area(scout,degree=3,k=20,folds=5):
    func = custom_func('aspline_final.R')
    rdf = df2r(scout[['seconds','velocity']])
    res=func.Aspline_Area_Proportions(rdf.rx2('seconds'),rdf.rx2('velocity'),degree,k=k)
    # res[0] = [list of knots selected],  res[1] = predicted total tweets, res[2] = predicted_curve
    return r2df(res[1])[0], r2df(res[2]).reset_index().pred, int(len(r2df(res[0])))
    # predicted total tweets, predicted curve, number of knots selected


def loess_area2(scout,folds=5,k=3,degree=3): # k and order are dummies so I can call loess and aspline in the same way
    func = custom_func('loess_final.R')
    rdf = df2r(scout[['seconds','velocity']])
    res = func.loess_stuff(rdf.rx2('seconds'),rdf.rx2('velocity'),folds=folds)
    return r2df(res[0])[0],r2df(r2df(res)[2]),r2df(res[3])[0] # [0]=trap, [1]=simp
    # trap_estimate, tweets estimated at each 86400 time points, span
    
def kde_r(x,n,kernel="epanechnikov",from_ = 0, to =86399):
    func = custom_func('kde_final.R')
    rdf = df2r(x)
    res = func.kde_function_custom(rdf,n,kernel=kernel, from_ = from_, to = to)
    kde,bw = r2df(res)
    kde = r2df(kde)
    bw = r2df(bw)[0]
    return kde,bw

def kde_r_bw(x,n,bw,kernel="epanechnikov",from_ = 0, to =86399):
    func = custom_func('kde_final.R')
    rdf = df2r(x)
    res = func.kde_function_custom_bw(rdf,n,rpy2.robjects.vectors.FloatVector([bw]),kernel=kernel, from_ = from_, to = to)
    kde = r2df(res)
    kde = r2df(kde)
    return kde

# absRelBias
def area_error(projected,n):
    return (abs(projected-n)/n)*100

def local_error(projected,actual):
    return ((projected-actual)**2).sum()/86400

# Use this to compare curve, MAB
def l1_norm(projected,actual):
    return (abs(projected-actual)).sum()/86400

# Use this too
# def l2_norm(projected,actual):
#     return np.sqrt(((projected-actual)**2).sum())/86400

# use this too
def rmsd(project,actual):
    return np.sqrt(((project-actual)**2).sum()/86400)

def kde_noScaling(x,kernel="epanechnikov",from_ = 0, to =86399):
    func = custom_func('kde_final.R')
    rdf = df2r(x)
    res = func.kde_function_custom_noScaling(rdf,kernel=kernel, from_ = from_, to = to)
    return r2df(res)

def ks_weighted(weights1,weights2):
    # sometimes have slightly negative data for aspline order 1, so we can just set it to 0 (the values are like -0.00002)
    if any(weights1 < 0): 
        if isinstance(weights1,(list,np.ndarray)):
            weights1 = pd.Series(weights1)
            
        weights1 = weights1.apply(lambda x: 0 if x <0 else x)
    # normalize
    weights1 = (weights1 / weights1.sum())
    weights2 = (weights2 /weights2.sum())
    
    x_values = list(range(0,86400))
    weights1r = rpy2.robjects.numpy2ri.numpy2rpy(weights1)
    weights2r = rpy2.robjects.numpy2ri.numpy2rpy(weights2)
    func = custom_func('ks_weighted.R')
    res = func.ks_weighted(x_values,x_values,weights1r,weights2r)
    return r2df(res)

def time_sampling_experiment(date_string, queries= 360, start_range_min = 0,max_rng=0,debug = False,random_number=0):
    date = datetime.strptime(date_string,'%Y-%m-%d') + timedelta(days=1)
    id_list = []
    date_list = [] # not needed, but nice to see what's going on
    start_range_max = 1440/queries
    start = date - timedelta(minutes= random_number)
    # print(random_number)
    for i in range(0, int(queries)):
        time = start - timedelta(minutes = start_range_max * i)
        date_list.append(time)
        id_list.append(tweetId_from_datetime(time))
    if debug:
        return id_list,date_list, random_number
    return id_list, date_list

# Need to generalize these methods for n spp (before we just average 2 together)

# n is size per pull
def velocity_pooling(scout,n):
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
    velocities =[]
    seconds_list = [] # the seconds the period starts at
    time_list = [] # the timestamp the period starts at
    for i in range(0,len(scout),n):
        subset = scout.iloc[i:(i+n)]
        seconds = subset.tpp.div(subset.velocity)
        
        # Take the midpoint to be the new timepoint
        if n % 2 == 1: # if odd, we should take the middle time point
            seconds_list.append(subset.iloc[n//2].seconds)
            time_list.append(subset.iloc[n//2].time)
        else: # Otherwise, find the middle by averaging the timestamps
            seconds_list.append(subset.seconds.mean())
            time_list.append(subset.time.mean())

        velocities.append(subset.velocity.mean())
    return new_velocity_helper(velocities,seconds_list,time_list)

def bin_by_minute(df,minutes=1):
    return df.assign(minute=(df.seconds// (60*minutes)).astype(int)).groupby('minute').tweet_id.count().rename("tweets")

def fix_kde(file,date,df,bw):
    date_dt = datetime.strptime(date,'%Y-%m-%d')
    bwd = int(bw) + 1 # round up
    temp_df = df.copy(deep=True)
    
    previous_day = (date_dt - timedelta(days=1)).strftime('%Y-%m-%d')
    prev_df = read_tweets(file.replace(date,previous_day),usecols=['tweet_id','tweet_date','retweet','region'])
    prev_df = prev_df.loc[prev_df['retweet'] == 0] # remove retweets
    prev_df = prev_df.loc[prev_df['seconds']> 86400-bw]
    # prev_df = prev_df.head(bwd)
    # rebase seconds
    base_sec = prev_df.iloc[-1].seconds
    prev_df['seconds'] = prev_df.seconds - base_sec
    temp_df['seconds'] = temp_df.seconds + prev_df.iloc[0].seconds
    #from_ = bwd
    from_ = temp_df.seconds.min()
    to = temp_df.seconds.max()
    
    temp_df = temp_df.append(prev_df)
    del prev_df
    
    next_day = (date_dt + timedelta(days=1)).strftime('%Y-%m-%d')
    next_df = read_tweets(file.replace(date,next_day),usecols=['tweet_id','tweet_date','retweet','region'])
    next_df = next_df.loc[next_df['retweet'] == 0] # remove retweets
    next_df = next_df.loc[next_df['seconds']< bw]
    #next_df = next_df.tail(bwd)
    
    next_df['seconds'] = next_df.seconds + temp_df.seconds.max() # df.iloc[0].seconds should be the same as df.seconds.max(), but let's not assume perfect order
    next_df = next_df.append(temp_df)
    del temp_df
    
    # adjusted kde based off the previous bandwidth
    kde_y = kde_r_bw(next_df.seconds,len(df),bw,kernel="epanechnikov",from_ = from_, to = to)
    
    return kde_y

MINUTES_FROM_END_OF_DAY = 2 # the number of minutes from the end of the day we should stay away

def clean_code(func,df,kde_y,scout,k=40,degree=3,folds=5):
    n = len(df)
    
    estimated_tweets,projected_curve,metadata = func(scout,k=k,degree=degree,folds=folds)
    absRel = area_error(estimated_tweets,n)
    mab = l1_norm(projected_curve,kde_y)
    rms = rmsd(projected_curve,kde_y)
    tstat, pval = ks_weighted(projected_curve,kde_y)
    return [estimated_tweets,metadata,absRel,mab,rms,tstat,pval]


# generate a new set of queries for every single different model and velocity computation
def estimation_methods(date,region,df,velocity_comp,scout,kde_y):
    cols = ['region','date','velComp','estMethod','totalTweets','estimatedTweets','span','knots','absRelBias','mab','rmsd','tstat','pval']
    cols2 = ['region','date','velComp','estMethod','span','knots','totalTweets','estimatedTweets','absRelBias','mab','rmsd','tstat','pval']
    
    errors = pd.DataFrame([],columns=cols2)
    constant_data = [region,date,velocity_comp]
    n = len(df)
    
    a20_1 = clean_code(aspline_area,df,kde_y,scout,k=20,degree=1)
    errors = errors.append(pd.DataFrame([constant_data +['aspline_20 | degree 1',-1,a20_1[1],n,a20_1[0]]+ a20_1[2:]],columns= cols2))
    
    a20 = clean_code(aspline_area,df,kde_y,scout,k=20,degree=3)
    errors = errors.append(pd.DataFrame([constant_data +['aspline_20 | degree 3',-1,a20[1],n,a20[0]]+ a20[2:]],columns= cols2))
    
    a40_1 = clean_code(aspline_area,df,kde_y,scout,k=40,degree=1)
    errors = errors.append(pd.DataFrame([constant_data +['aspline_40 | degree 1',-1,a40_1[1],n,a40_1[0]]+ a40_1[2:]],columns= cols2))
    
    a40 = clean_code(aspline_area,df,kde_y,scout,k=40,degree=3)
    errors = errors.append(pd.DataFrame([constant_data +['aspline_40 | degree 3',-1,a40[1],n,a40[0]]+ a40[2:]],columns= cols2))
    
    loess3 = clean_code(loess_area2,df,kde_y,scout,k=40,degree=3,folds=3)
    errors = errors.append(pd.DataFrame([constant_data +['loess | 3 fold',loess3[1],-1,n,loess3[0]]+ loess3[2:]],columns= cols2))
    
    loess5 = clean_code(loess_area2,df,kde_y,scout,k=40,degree=3,folds=5)
    errors = errors.append(pd.DataFrame([constant_data +['loess | 5 fold',loess5[1],-1,n,loess5[0]]+ loess5[2:]],columns= cols2))
    
    errors = errors[cols] # reorganize the column order
    return errors
    
# n = 96,48, 32
# currently
def get_scout(df,date,n,spp,method,old_method=False):
    old_spp = spp//100 # used for pooling and averaging
    if method in ['averaged','pooled']: # if in average or pooled method, we sample 96 queries and then do different stuff to condense the info
        n = 96
        spp = 100
        
    max_rng = int(1440/n) 
    max_rng = max_rng - MINUTES_FROM_END_OF_DAY # Make it so we are at most 2 minutes away from the end of the day

    start_number = random.uniform(0,max_rng)
    sample_intervals,time_intervals = time_sampling_experiment(date,queries = n,random_number = start_number)
    if old_method:
        scout,_,_ = velocity_cleaner(df,date,sample_intervals,time_intervals,spp, 540) # spp is always 100
    else:
        scout,_,_ = velocity_cleaner_buskirk(df,date,sample_intervals,time_intervals,spp, 540)
    if method == 'averaged':
        scout = velocity_avg(scout,old_spp)
    elif method == 'pooled':
        scout = velocity_pooling(scout,old_spp)
    return scout

#@ray.remote
def flexible_errors(date,region):
    file = filename_builder(date,region)
    print(file)
    df = read_tweets(file,usecols=['tweet_id','tweet_date','retweet','region'])
    df = df.loc[df['retweet'] == 0] # remove retweets
    
    _,bw = kde_r(df.seconds,len(df),kernel="epanechnikov")
    kde_y = fix_kde(file, date, df, bw)
    
    #binned_y = bin_by_minute(df,minutes = 1)
    
    # We need to generate the queries first (time points)
    vel_comps = {'96_direct':get_scout(df,date,96,100,"direct"), '48_direct':get_scout(df,date,48,200,"direct"),
                 '48_pooled':get_scout(df,date,48,200,"pooled"), '48_averaged':get_scout(df,date,48,200,'averaged'),
                 '32_direct':get_scout(df,date,32,300,"direct"), '32_pooled':get_scout(df,date,32,300,"pooled"),
                 '32_averaged':get_scout(df,date,32,300,'averaged')}
  
    error_dfs = []
    for velocity_comp, scout in vel_comps.items():
        error_dfs.append(estimation_methods(date,region,df,velocity_comp,scout,kde_y))
    error_df = pd.concat(error_dfs)
    print('finished',file)
    return error_df

# Estimation methods
    # Aspline order 1: 20 and 40 knots
    # Aspline order 3: 20 and 40 knots
    # Loess k=3
    # Loess k=5
    
# Velocity Computation Methods
    # 96 queries direct
    # 48 queries, spp= 200: direct, pooled, averaged
    # 32 queries, spp= 300: direct, pooled, averaged
    
    
# Test the idea of velocity starting timepoint being the start of the time point instead of when we get the first tweet
#%%
ray.init(num_cpus=15)
#%%
dates = btils.generate_dates_range('2020-10-08','2020-11-21')
regions = ['Atlanta-Sandy Springs-Roswell, GA','Phoenix-Mesa-Scottsdale, AZ','Chicago-Naperville-Elgin, IL-IN-WI','Baltimore-Columbia-Towson, MD']


# i = 0
# j = 0
# 11-23-2020 pheonix bad data
# 10-07 first day of non-retweet data
for z in [5,6]:
    result = [flexible_errors.remote(dates[i],regions[j]) for i in range(len(dates)) for j in range(len(regions))]
    r= ray.get(result)
    
    final_out = pd.concat(r)
    final_out.to_csv(f'final_out_new{z}.csv',index=False)
    # a = final_out.groupby(['velComp','estMethod']).mean()
    # a.to_csv(f'final_out{z}_summary.csv')
#%%
dates = btils.generate_dates_range('2020-10-08','2020-11-21')
regions = ['Atlanta-Sandy Springs-Roswell, GA','Phoenix-Mesa-Scottsdale, AZ','Chicago-Naperville-Elgin, IL-IN-WI','Baltimore-Columbia-Towson, MD']

for i in [5]:
    r = []
    for date in dates:
        for region in regions:
            r.append(flexible_errors(date,region))
            
    final_out = pd.concat(r)
    final_out.to_csv(f'final_out_new{i}.csv',index=False)
#%% Creating file for dash_testing.py

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

df1 = pd.read_csv(r"G:\Python File Saves\final_out_new1.csv")
df2 = pd.read_csv(r"G:\Python File Saves\final_out_new2.csv")
df3 = pd.read_csv(r"G:\Python File Saves\final_out_new3.csv")
df4 = pd.read_csv(r"G:\Python File Saves\final_out_new4.csv")
df5 = pd.read_csv(r"G:\Python File Saves\final_out_new5.csv")
df1['run'] = 1
df2['run'] = 2
df3['run'] = 3
df4['run'] = 4
df5['run'] = 5

df = pd.concat((df1,df2,df3,df4,df5))
b = df.groupby(["region","date",'velComp','estMethod'])
temp = []
for g,frame in b:
    temp.append(frame)
c = pd.concat(temp)
c = c[['region','date','velComp','estMethod','run','totalTweets', 'estimatedTweets', 'span', 'knots', 'absRelBias', 'mab',
         'rmsd', 'tstat', 'pval']]
c.to_csv("preExperimentData.csv")

# df = pd.concat((df1,df2,df3,df4,df5))
# df = pd.concat((df2,df3,df4))
# averaged = df.groupby(df.index).mean()

# df1[['region','date','velComp','estMethod']]
# averaged[['region','date','velComp','estMethod']] = df1[['region','date','velComp','estMethod']]
# averaged[['region','date','velComp','estMethod']] = df2[['region','date','velComp','estMethod']]
# averaged = averaged[['region','date','velComp','estMethod','totalTweets', 'estimatedTweets', 'span', 'knots', 'absRelBias', 'mab',
#         'rmsd', 'tstat', 'pval']]
# averaged.to_csv(r"G:\Python File Saves\averaged_experiment_data_8_27_2021.csv",index=False)

#%% try to get negative values

# tweet_folder = r'E:\my_tweets'
# num_files = 180
# files = random_sample_files(tweet_folder,n=num_files,concat=False,read=False)
# for i in range(num_files):
#     files.iloc[i].file,files.iloc[i].dates,files.iloc[i].regions

#%%# testing    
# old_method = pd.read_csv('final_out.csv')
# new_method = pd.read_csv('final_out_new_method.csv')

# a = old_method.groupby(['velComp','estMethod']).describe()
# new_method.groupby(['velComp','estMethod'])

#%%
# plt.plot(kde_r(df.seconds,len(df),kernel="epanechnikov"),label = 'Normal KDE')
# plt.plot(kde_y,label= 'Adjusted KDE')
# plt.legend()

# test_kde = kde_noScaling(next_df.seconds,kernel="epanechnikov",from_ = from_, to = to) 

# sns.distplot(df.seconds,bins=3000,label = 'Normal KDE')
# plt.plot(test_kde,label = 'Adjusted KDE')
# plt.legend()
# plt.show()

# kde_y,bw = kde_r(df.seconds,len(df),kernel="epanechnikov")
# kde_y = r2df(kde_y)
# bw = r2df(bw)[0]
# plt.plot(kde_y)

# kde_y2,bw2 = kde_r(next_df.seconds,len(df),kernel="epanechnikov",from_ = from_, to = to)
# kde_y2 = r2df(kde_y2)
# bw2 = r2df(bw2)[0]
# plt.plot(kde_y2)

# kde_y3 = kde_r_bw(next_df.seconds,len(df),round(bw),kernel="epanechnikov",from_ = from_, to = to)



# scout = get_scout(df,date,96,100,"direct")
# expected,curve, span = loess_area2(scout,folds=5)
# expected,curve, span = aspline_area(scout,folds=5)
# plt.plot(curve,label='new velocity')
# plt.plot(kde_y)
# plt.plot(kde_r(df.seconds,len(df),kernel="epanechnikov")[0])
# # plt.plot(kde_y,label='standard kde')
# # plt.plot(kde_y2,label ='adjusted kde, auto bw')
# plt.plot(kde_y3,label ='adjusted kde, set bw')
# #plt.plot(curve,label='loess')
# plt.legend()

# test_kde.sum()
# curve.sum()
# plt.plot(binned_y)

# ks_weighted(curve,kde_y)

# curve.sum()
# kde_y3.sum()

# len(df)

# from rpy2.robjects import numpy2ri
# numpy2ri.activate()

# scout,_,_=velocity_cleaner_buskirk(df,'2020-09-10',sample_intervals,time_intervals,100, 2000)
# scout2,_,_=velocity_cleaner(df,'2020-09-10',sample_intervals,time_intervals,100, 2000)

# scout.velocity.plot(label='new velocity')
# scout2.velocity.plot(label='old velocity')
# expected,curve, span = loess_area2(scout,folds=5)
# expected_old,curve_old, span_old = loess_area2(scout2,folds=5)

# plt.plot(curve,label='new velocity')
# plt.plot(kde_y)
# plt.plot(curve_old,label='old velocity')
# plt.legend()