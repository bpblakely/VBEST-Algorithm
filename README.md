## VBEST: A Velocity Based Estimates for Sampling Tweets

Check core_functions.py for the core functions of VBEST

Citation

`
Buskirk, T. & Blakely, B. (2021). Sweet Tweets! Investigating the Performance of a New Method for Probability-Based Twitter Sampling Trent D. Buskirk, Novak Family Distinguished Professor of Data Science, BGSU Brian Blakely, Computer Science and Mathematics Major, BGSU
`


<!-- ABOUT THE PROJECT -->
## About The Project

This research conducted at Bowling Green State University. 

### Motivation

Sampling Tweets from Twitter is a common task for researchers, but previous techniques focused on building a graph of connections, where each node represents some relationship between 2 users. 

This is useful, but for our use case we wanted to randomly sample Twitter for Covid related Tweets based on geography (certain cities in the U.S.). 
Using the previous techniques of building a graph of users doesn't dynamically scale as new people join Twitter, or visit a city for a weekend and make a Tweet while there.

This is why we chose to use [Twitters Search API](https://developer.twitter.com/en/docs/twitter-api/v1/tweets/search/api-reference/get-search-tweets), which let us sample 1 week back in time (with the free edition).

Using this API we can use query tweets using keywords, geography, and much more.

**However**, the issue with the version 1 of this API is that it allows you to select a date to sample from, but doesn't let you specify the time of that day to sample from (this is now a feature in v2). 
The only thing you could do is use the 'Recent method' in the search API, and slowly work backwords from the end of the day to the start. 
Depending on the query parameters selected, there could be hundreds of millions of Tweets in a day, which is impossible to collect with the API limitations.

Ideally, we'd want to take a random sample or uniform sample of Tweets, since this lets us stay within the API limitations and reduces bias that could be introduced by only sampling the last N Tweets in a day.

To achieve this, we used synthetically made Tweet IDs to leverage the max_tweet_id query search option to let us sample at any time in a day. 
With precise time control (down to the millisecond in the day), we could then create any sampling technique we wanted.

<p align="right">(<a href="#top">back to top</a>)</p>

### Applications

**This research is directly applicable to [Twitter's Full-Archive Search for Academic Research](https://blog.twitter.com/developer/en_us/topics/tools/2021/enabling-the-future-of-academic-research-with-the-twitter-api)!** 

With researchers having full archive access to Twitter's Tweet database there are going to be hundreds of billions of Tweets that you can look at, but you are limited to a few million each month.
With so much information available and only being able to access a fraction of it, it becomes important to plan how you spend your queries, so you can get the most out of your months worth of data.

VBEST models the distribution of Tweets, pertaining to your specific search queries, and returns a list of the best times that you should sample based on the distribution of Tweets.

<p align="right">(<a href="#top">back to top</a>)</p>

#### Ways to use VBEST in Full-Archive Research 

*This is useable for ANY of the parameters in the Search API, so filtering by keyword searches, and/or geography, and/or language, ect, all work just fine as long as you consistently use the same parameters.*

1. Estimate the number of Tweets for you search query while using minimal queries.
   * This estimation is simply the area under the modeled distribution curve.
   * You could estimate the keyword "Covid" for all of America over the last 2 years.
      * There would be billions of Tweets, but you don't have to spend billions of queries to find that out. You can simply use VBEST to find an accurate estimate of the total number of Tweets matching our search criteria.

2. Estimating keyword frequency.
   * Keywords can be estimated by first estimating a keyword as defined previously, then apply step 1 again, but using the keyword " * " which is a wildcard operater  allowing any Tweet to be considered.
   * Manually computing keyword frequency could be impossible if the number of Tweets you're considering is in the millions, so estimating it makes it not only possible, but easy

3. Efficient sampling
   * The main point of VBEST was to build an efficent sampling algorithm around the many caveats of Twitter and its API.
   * By roughly knowing where Tweets fall in a given day, you can spread your samples out to ensure you don't duplicate any data or waste any queries.

I'm sure there are many other ways to apply this research, but we value these three applications as being essential for any future full-archive research, given the scale of the data available and the API limitations in place.

Using velocity as a source of estimation will allow researchers to get much more out of their limited resources as possible.


### Built With

* [Python](https://www.python.org/)
* [R](https://www.r-project.org/)
* [Rpy2](https://rpy2.github.io/)
* [Tweepy](https://www.tweepy.org/)


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

In order to use the VBEST algorithm, you need to set up rpy2, so that you can use R's LOESS model. 

How to sample:

1. Call time_sampling("date_string", n_queries) 
    * This function is used for the initial uniform sampling
    * Where "date_string" in the form "year-month-day". This is the date you're sampling from.
    * Creates n_queries amount of uniform timestamps and their corresponding tweet IDs
    * Returns **tweet ID list**, **timestamp list**
        * This is used to find the velocities through out a day, use the tweet IDs to sample from twitter
        * Timestamps are just used as a reference
    * Currently this function only supports time intervals of 1 day, but it's easy to scale to n days.
   
2. Call velocity_cleaner_real(tweepy & query stuff, ...,  sample_intervals, time_intervals, ...)
    * Used for taking the uniform tweet ID's, sampling them from twitter, computing velocity, and unifies the data
      * *Or call velocity_cleaner if you are providing a DataFrame of all tweets for a given day*
    * **sample_intervals**: the **tweet ID list** returned from time_sampling()
    * **time_intervals**: the **timestamp list** returned from time_sampling()
    * **Returns a DataFrame** consiting of summaries of each query used for the uniform sampling
        * **Returned DataFrame** *(Each row represents a query, each row has 8 variables defined as below)*
            * query_1:
              * *Variable Name : Description*
              * **velocity** : duration_of_query_in_seconds / total_tweets_pulled (TPP) 
              * **time** : (start time, time of first tweet in query)
              * **min_id** : (tweet ID)
              * **max_id** : (tweet ID)
              * **time_end** : (time of last tweet in query)
              * **utc** : (time start in UTC)
              * **seconds** : (duration of query: time - time_end)
              * **tpp** : (Total Tweets Pulled, total amount of Tweets returned in the query)
 
3. Call vbest_test(DataFrame from step 2, number of tweets to sample)
   * Returns a lot of stuff, but the first return is a DataFrame of timepoints to sample from
   * There are 3 types of different samples and 3 different sample sizes in this DataFrame
      * This needs to be cleaned up to only return 1 output
   * Currently you need to turn the returned sampling timepoints into tweet IDs using sec2ids()
    
4. Take the tweet IDs and sample 

<p align="right">(<a href="#top">back to top</a>)</p>

## Hyper-parameter Testing

Pre-experiment data comparing hyper-parameters and other smoothing techniques such as spline fitting can be found here.

This mainly compares the number of queries you should use in the uniform sampling step as well as some possible techniques to reduce the variance of velocities.

https://pre-experiment.herokuapp.com/


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
