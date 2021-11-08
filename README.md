## VBEST: A Velocity Based Estimates for Sampling Tweets

Check core_functions.py for the core functions of VBEST

Citation

`
Buskirk, T. & Blakely, B. (2021). Sweet Tweets! Investigating the Performance of a New Method for Probability-Based Twitter Sampling Trent D. Buskirk, Novak Family Distinguished Professor of Data Science, BGSU Brian Blakely, Computer Science and Mathematics Major, BGSU
`

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This research conducted at Bowling Green State University. 

<p align="right">(<a href="#top">back to top</a>)</p>

### Motivation

Sampling Tweets from Twitter is a common task for researchers, but previous techniques focused on building a graph of connections, where each node represents some relationship between 2 users. This is useful, but for our use case we wanted to randomly sample Twitter for Covid related Tweets based on geography (certain cities in the U.S.). Using the previous techniques of building a graph of users doesn't dynamically scale as new people join Twitter, or visit a city for a weekend and make a Tweet while there. This is why we chose to use [Twitters Search API](https://developer.twitter.com/en/docs/twitter-api/v1/tweets/search/api-reference/get-search-tweets), which let us sample 1 weeks worth of a data using the free tier




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
    


# VBEST Algorithm
 The code used in the research and development of the VBEST algorithm for Twitter

Pre-experiment data comparing hyperparameters and other smoothing techniques such as spline fitting can be found here

https://pre-experiment.herokuapp.com/


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search)

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
