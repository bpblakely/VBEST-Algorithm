# Python WIP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from velocity_utils import *
import brian_utils as btils
from multiprocessing import Pool, cpu_count
from functools import partial

#%% Step 1: CV Loess
import lowess
import random

loess_iterations = 3

def kfold_alternate(x,y,span_val,folds=3):
    # K fold cross validation where you alternate data by the number of folds
    # Works better than the standard chunking approach for Time Series Data
        # [1,2,3,4,5,6,7,8,9,10] with folds = 3
        # fold 1: [1,4,7,10]
        # fold 2: [2,5,8]
        # fold 3: [3,6,9]
    loess = lowess.Lowess()
    errors = []
    for i in range(folds):
        test_index = list(range(i,len(x),folds)) # generates alternating indicies
        train_index = [n for n in range(len(x)) if n not in test_index] # gets the rest
        
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        loess.fit(x_train ,y_train, frac= span_val, robust_iters = loess_iterations)
        yhat = loess.predict(x_test,return_pairs=False)
        
        errors.append(np.mean(abs(y_test-yhat)))
    return errors


def loess_wrapper(x,y,span_vals=np.around(np.arange(.2,.65,.05),decimals=2),folds = 3):
    errors = []
    
    shuffle_index = np.random.permutation(len(x))
    x_shuffled, y_shuffled = x[shuffle_index], y[shuffle_index]

    for span_val in span_vals:
        errors.append(np.mean(kfold_alternate(x_shuffled, y_shuffled, span_val, folds = folds)))
    
    best_span = span_vals[np.argmin(errors)]
    loess = lowess.Lowess()
    loess.fit(x,y, frac = best_span,robust_iters = loess_iterations)
    return loess # loess.frac to see the selected span value

def vbest_function(x,y,refine=.005,span=None):
    # if seed is None:
    #     seed = random.randint(0,10e7)
    # random.seed(seed)
    if span is None:
        model = loess_wrapper(x,y,folds=3)
    else:
        model = lowess.Lowess()
        model.fit(x,y, frac = span, robust_iters = loess_iterations)
    
    # Get model fit params for parallelzation usage
    span = model.frac
    weights = model.weighting_locs
    matrix = model.design_matrix    
    
    x_grid = np.around(np.arange(0,86400+refine,refine),decimals=5) # rounded to 5 decimal places

    cpus = cpu_count() 
    if cpus > 1: cpus -= 2 # Let's save 1 cpu for the user

    # Try to size our chunks to best distribute our data
    chunk_n = int((x_grid.shape[0] // cpus)//1.25)
    
    chunks = [x_grid[i:i+chunk_n] for i in range(0,x_grid.shape[0],chunk_n)]
    
    part_func = partial(lowess.loess_predict,design_matrix=matrix,frac=span,weighting_locs=weights,return_pairs=True)
    
    with Pool(cpus) as pool:
        result = pool.map(part_func,chunks)
    
    result = np.concatenate(result) # still should check np.trapz
    
    m = np.matrix([result[range(1,len(x_grid)-2,2)][:,1],
              result[range(2,len(x_grid)-1,2)][:,1],
              result[range(3,len(x_grid),2)][:,1]])
    
    coeffs = np.matrix([refine/2,refine,refine/2])
    loess_areas = coeffs @ m
    area = loess_areas.sum()
    
    
    # Now we do it the fast way and see the difference
    x_grid_normal = np.arange(0,86400)
    preds = model.predict(x_grid_normal,return_pairs=False)
    normal_area = preds.sum()
    normal_trapz = np.trapz(preds)
    
    return area, normal_area,normal_trapz,span


#%%
df = pd.read_csv(r'E:\Twitter Data\twitter_query_data\all_data.csv',usecols=['date','region','totTweets','totTweetsEst'])
df = df.drop_duplicates() 
#%%
areas = [] # area python full, area_python normal (less x_grid), python trapz of lesser, Rarea, total_area
dates = btils.generate_dates_range('2020-11-24','2021-01-01')

regions = ['Atlanta-Sandy Springs-Roswell, GA','Phoenix-Mesa-Scottsdale, AZ','Chicago-Naperville-Elgin, IL-IN-WI','Baltimore-Columbia-Towson, MD']

for region in regions:
    for date in dates:
        # data
        scout_file = filename_model_builder(date,region)+'scout.csv'
        scout = pd.read_csv(scout_file)
        x = scout['seconds'].to_numpy()
        y = scout['velocity'].to_numpy()
        total, estimated = df.loc[(df.date == date) & (df.region == region)][['totTweets','totTweetsEst']].iloc[0]
        spanR = get_model_span(date,region)
        
        # model
        area,normal_area,normal_trapz,spanPython = vbest_function(x,y)
        areas.append([region,date,spanPython,spanR,area,normal_area,normal_trapz,estimated,total])
        
data = pd.DataFrame(areas,columns=['region','date','spanP','spanR','areaP','area_ez','area_trapz','areaR','true'])

print(np.mean(abs(data['true'] - data['areaP'])/data['true']))
print(np.mean(abs(data['true'] - data['area_ez'])/data['true']))
print(np.mean(abs(data['true'] - data['area_trapz'])/data['true']))
print(np.mean(abs(data['true'] - data['areaR'])/data['true']))

#%% PSU Grid Generation

date = '2020-12-17'
region = 'Chicago-Naperville-Elgin, IL-IN-WI'
scout_file = filename_model_builder(date,region)+'scout.csv'
scout = pd.read_csv(scout_file)
x = scout['seconds'].to_numpy()
y = scout['velocity'].to_numpy()
total, estimated = df.loc[(df.date == date) & (df.region == region)][['totTweets','totTweetsEst']].iloc[0]

# model
model = loess_wrapper(x,y)
by = 1
x_grid = np.arange(0,86400,by)
y_hat = model.predict(x_grid)
y_hat[:,1] *= by 
# y_hat[:,1].sum()


#%% PSUs

psu = [] # (index start, index end) list of points of the index ranges
cumsum = 0
index_start = 0

for i,value in enumerate(y_hat[:,1]):
    cumsum += value
    if cumsum >= 99.9999999999999999:
        psu.append([cumsum,index_start,i])
        cumsum = value
        index_start = i
psu.append([cumsum,index_start,len(y_hat)-1]) # add the last PSU point
psus = pd.DataFrame(psu,columns=['Volume','ind_start','ind_end'])
psus['second_start'] = y_hat[:,0][list(psus['ind_start'])]
psus['second_end'] = y_hat[:,0][list(psus['ind_end'])]
psus = psus[:-1] # Drop last row so we can't select the undersampled timepoint

# We start our sample at second_end, since twitter gives data in reverse order


# psus['bin'] = pd.cut(psus['second_end'], bins = pd.interval_range(0,86400,freq= 100))
# psus['bin'] = psus['bin'].apply(lambda x: int(x.right))
# psus.groupby('bin').count()['second_end'].plot()
#%% Sampling
vbest = []
srs = []
inv = []
weights = []

random.seed(10)

for i in range(3):
    
    samplesize = 624 - 180*i
    if len(psu) < samplesize:
        samplesize = len(psu)
    # vbest
    
    sampint = len(psu) / samplesize
    randintX = np.random.choice(range(0,len(psu)),1)[0]
    
    samp = [randintX]
    for j in range(1,samplesize):
        samp.append(round(randintX + j*sampint) % len(psu))
    samp = sorted(samp)[::-1]
    vbest.append(psus.iloc[samp])
    
    # srs
    samp = sorted(np.random.choice(range(0,len(psu)),samplesize,replace=False))[::-1]
    srs.append(psus.iloc[samp])
    
    # inv
    weightvec = np.c_[y_hat[:,0], y_hat[:,1] / y_hat[:,1].sum()]
    samp = sorted(np.random.choice(len(y_hat[:,0]),samplesize,replace=True, p = weightvec[:,1]))[::-1]
    inv.append(y_hat[samp])
    weights.append(weightvec[samp])

# vbest['bin'] = pd.cut(vbest['second_end'], bins = pd.interval_range(0,86400,freq= 100))
# vbest['bin'] = vbest['bin'].apply(lambda x: int(x.right))
# vbest.groupby('bin').count()['second_end'].plot()
#%%
# sample data

# date = '2020-12-17'
# region = 'Chicago-Naperville-Elgin, IL-IN-WI'
# sampled_scout = filename_model_builder(date,region)+'scout.csv'

# scout = pd.read_csv(sampled_scout)
# x = scout['seconds'].to_numpy()
# y = scout['velocity'].to_numpy()

# corpus_scout = get_velocity_df(filename_builder(date,region))
# corpus_scout = corpus_scout.iloc[::-1].reset_index(drop=True)
# corpus_scout['time'] = corpus_scout['time'].apply(dt2sec)
#%%
# x_start = x.copy()
# y_start = y.copy()

# shuffle_index = np.random.permutation(len(x))
# x, y = x[shuffle_index], y[shuffle_index]

# test_index = list(range(i,len(x),folds)) # generates alternating indicies
# train_index = [n for n in range(len(x)) if n not in test_index] # gets the rest

# x_train, x_test = x[train_index], x[test_index]
# y_train, y_test = y[train_index], y[test_index]

# # loess = lowess.Lowess()
# # loess.fit(x_train ,y_train, frac= .25,robust_iters=3)
# # y_hat = loess.predict(x_test)
# # print(np.mean(abs(y_test-y_hat)))

# train_sort = x_train.argsort()
# test_sort = x_test.argsort()

# x_train = x_train[train_sort]
# x_test = x_test[test_sort]
# y_train = y_train[train_sort]
# y_test = y_test[test_sort]

# loess = lowess.Lowess()
# loess.fit(x_train ,y_train, frac= .25,robust_iters=3)
# y_hat = loess.predict(x_test)
# print(np.mean(abs(y_test-y_hat)))

# final_x = np.append(x_train,x_test)
# ind = final_x.argsort()
# final_x = final_x[ind]
# final_y = np.append(y_train,y_hat)[ind]

# plt.plot(final_x,final_y)
# plt.scatter(x_train,y_train,label = 'train',color='blue')
# plt.scatter(x_test,y_hat,label='predicted y',color='skyblue',marker='s')
# plt.scatter(x_test,y_test,label='true y',color='red',marker='x')
# # plt.plot(x_start,y_start,label = 'train',color='grey')

# plt.legend()
# plt.show()

# # This doesn't work well for our case
# def kfold_chaining(loess,x,y,span_val,folds=3):
#     # https://stats.stackexchange.com/questions/14099/using-k-fold-cross-validation-for-time-series-model-selection
#     # Works better with larger number of folds
#     errors = []
#     for i in range(folds-1): # We use 1 less fold since we are chaining
#         split_index = int(len(x)*(1/folds))*(i+1) # gives a fraction of the data, *(i+1) makes it scale
#         x_train, x_test = x[:split_index], x[split_index:split_index*2]
#         y_train, y_test = y[:split_index], y[split_index:split_index*2]
#         loess.fit(x_train ,y_train, frac= span_val)
#         yhat = loess.predict(x_test)
#         errors.append(np.mean(abs(y_test-yhat)))
#     return errors

# # seems to just be worse than kfold_alternate, which makes sense b/c we are just making the mean worse by adding smaller steps
# def kfold_alt_chain(loess,x,y,span_val,folds=3):
#     # https://stats.stackexchange.com/questions/14099/using-k-fold-cross-validation-for-time-series-model-selection
#     # Idea is to split data into chunks like normal, but we instead train on that data alternated
#     # The goal is to catch smaller patterns in the data, but using alternating to overall improve the results
#     errors = []
#     for i in range(folds): # At the last fold we are just doing 1 normal run of kfold_chaining
#         split_index = int(len(x)*(1/folds))*(i+1)
#         alt_errors = kfold_alternate(loess,x[:split_index],y[:split_index],span_val,folds=folds)
#         errors.append(np.mean(alt_errors))
#     return errors

#%%
# x = np.linspace(0, 20000, num=10000)
# y = np.sin(x)

# #loess = loess_wrapper(x,y)
# import timeit

# loess = lowess.Lowess()
# timeit.timeit(lambda: loess.fit(x,y,robust_iters=0,frac=.25),number=1)

# loessM = moepy.lowess.Lowess()
# timeit.timeit(lambda: loessM.fit(x,y,robust_iters=0,frac=.25),number=1)

# x_grid = np.arange(0,20000,1)

# a = loess.predict_fast(x_grid)
# b = loess.predict(x_grid)

# timeit.timeit(lambda: loess.predict(x_grid),number=1)
# timeit.timeit(lambda: loess.predict_fast(x_grid),number=1)

# timeit.timeit(lambda: loessM.predict(x_grid),number=1)

# x_grid = np.arange(0,86400,1) # 86400 is EXCLUDED 
# y_predict = loess.predict(x_grid)
# areaP = np.trapz(y_predict)

# loess_corpus = loess_wrapper(corpus_scout['time'].to_numpy(),corpus_scout['velocity'].to_numpy())
# y_predictC = loess_corpus.predict(x_grid)
# areaC = np.trapz(y_predictC)

# plt.scatter(x,y,color='red',marker='.',label='sampled velocities')
# #plt.scatter(corpus_scout['time'],corpus_scout['velocity'],color='black',label='Velocities on pull',alpha=.5)
# plt.plot(x_grid,y_predict,label='Python Loess')
# plt.plot(x_grid,y_hat,label='R Loess')
# #plt.plot(x_grid,y_predictC,label='Python Loess Corpus')

# plt.xlabel('seconds')
# plt.ylabel('velocity')
# plt.legend()

