# Exploring the Factors that Affect Airbnb Prices in London 

## Focus Problem


![](Images/image10.jpg)
<p align="center"> What are the most signifcant factors influencing Airbnb prices in London? </p>

## Executive Summary 

Having reported more than 100 million bookings in the first quarter of 2022, the Airbnb business has resurged from the pandemic and is booming. ‘Air Bed and Breakfast’ or Airbnb is essentially an online marketplace whereby those who wish to rent out their property are connected to people who are searching for accommodation when visiting that location. Airbnb has over 5.6 million listings across 220 countries globally and 100,000 locations. Some of its main competitors include Booking.com, TUI and Tripadvisor. 
Living in London ourselves, the key question we wanted to consider is what causes the prices of the hundreds of listings in London to fluctuate.


![](Images/airbnb.png)
  
Above we see that a rental request for 4 adults in London for 3 nights brings about 2 significantly different results in terms of price per night. So what drives these variations? To help us answer this question, we extracted 66,000 of London’s listings from InsideAirbnb and used them to analyse the features a listing possessed and its effect on its price. We investigated features such as location, amenities and property type to identify possible correlation between these features and the listing’s price. We then extracted over 1 million reviews to understand the sentiment behind the most expensive listings. Furthermore, London is a world city as it is home to some of the best universities and it hosts a variety of international financial services thus there is high demand for property in this city. This also means that there is high competition for rental properties and small variations in price can make a significant difference in demand for the listing. If the price is too high, people will not be willing to pay and if the price is too low, the owners are losing potential income. Therefore through this project we used machine learning and regression analysis to help predict the base price for rental properties in London.

Through the use of tools such as Pandas, Matplotlib, Seaborn, and Python to analyse and visualise our data, we found that there is some correlation with prices and the factors we have chosen to investigate (location, Wifi), while some don’t seem to have any trend (availability 365). Also, we have learned how to use Natural Processing Language to analyse comments (unstructured textual data)  which will be helpful for the hosts to understand each  guest’s activities, opinions and feedback to successfully derive their rental services. Furthermore, we learnt that predicting prices with Random Forest Regression is better compared to other selected machine learning regression models. However, we have to keep in mind that there are price outliers in the listings data, making machine learning a bit inefficient.

Overall, we recognized the listings and reviews data is not perfect because the lead time at which they were gathered was 9th September 2021. Therefore, the factors may have a different relationship with the price today than the price from September. However, we see machine learning and python libraries have given the importance of reviews and price prediction to hosts and guests. This will encourage more research for the data analyst to help hosts to see if their Airbnb price could be sensitive to the change of such factors observed.

Finally, we were able to put together a list of recommendations to prospective guests and hosts in order to make the most value out of their next property rental. Two examples of these include:
- Location is an important factor that affects prices 
- the most expensive boroughs are in the centre of London (Kensington, Chelsea, City of London, Westminster)
- Top amenities an aspiring host should include: Wifi, smoke alarm, essentials, heating, long term stay allowed


## Motivation

Our initial motivation for this project stemmed from our awareness of Airbnb scams being presented on the news quite frequently. Thus, we initially wanted to base our project around identifying scam listings, however we decided to change the route of this project into something broader and useful to us as students. Since 2007, Airbnb has vastly expanded across many countries and has disrupted the hospitality sector as people move away from traditional hotel and resort bookings. We were really interested in exploring this topic as Airbnb has revolutionised the property rental market and has made it much easier and more affordable to travel thus making it a great way for students to explore countries. As we all study Economics, this topic was of particular relevance to us as we are aware that Airbnb capitalises upon the sharing economy thus it was intriguing to use the array of data points to find out how hosts could increase their revenue. Also, due to having an interest in financial markets, Airbnb is a stock that will continue to keep growing due to its disruptive nature therefore analysing its market structure and how they can reign in more revenue is particularly interesting.

With so many different properties in each location, it becomes difficult to find the perfect one that is good value for money, in a convenient location with tourist attractions and is equipped with the amenities we need. Furthermore, through a quick surf through Airbnb’s website, we see many listings that look very similar in photos but are priced very differently thus we wanted to delve deeper into the hidden discrepancies that may be causing these price variations. 


## Justification

Our project is relevant to all three stakeholders - rental property hosts, the firm and guests. Analysis of the factors that make a listing more expensive means that hosts can ensure their listing has all the essential features and amenities to charge a higher price. As well as this, guests are informed of what factors make a listing cheaper and that the property they have chosen includes features that they want. Although Airbnb is able to give general advice to help hosts price their listings, there are not many services that are able to use several data points to accurately predict the price of a listing thus we deemed it fit to include a machine learning model that could help forecast base prices.  

![](Images/justification.png)

Our project is particularly relevant now because there has been a bounce back in the holiday market. In fact, Airbnb bookings hit a record high in March of 2022 according to Airbnb’s financial results. Airbnb was rapidly growing in 2020, but after COVID-19 caused a strike to the travel industry, there was a severe fall in booking numbers. On a year on year basis, bookings fell by 72% and at one point there were more cancellations than bookings due to travel restrictions. This placed pressure on Airbnb and all the hosts who were losing revenue due to the significant decrease in bookings being made. There was resilience as people began to travel domestically, however now the travel industry has made a full comeback with bustling airports over the Jubilee Bank Holiday Weekend, for example. It is now more important than ever that hosts have accurately priced their listings and filled them with the most sought after amenities and facilities to avoid making further losses. Looking at reviews is also particularly important as reviews build up communication and trust between the listing owner and guests. This constructive criticism also allows property owners to refine their listing according to their guest’s desires. Thus we decided that conducting sentiment analysis on the reviews from the listings  that bring in the most revenue would be extremely important within our data analysis.


## Aims
The primary aim of this project is to explore the factors that influence Airbnb prices in London. The secondary aim of this project is to create a mechanism in order to predict prices of listings in London using machine learning.

Airbnb has become a disruptor in the market for property rentals thus has become a relatively easy way for people to earn extra income through investing in new properties and renting them out via Airbnb. Guests often find Airbnbs to be cheaper and tend to prefer them over hotels due to the ‘homier’ feel. However, as Airbnb rapidly expands with thousands of new listings being added everyday, we wanted to find out what type of properties a host should invest in and what features they should include in order to make the most rental revenue and to retain customers leading us to our first question: what are the most important factors influencing pricing? To address this we decided to look at the features that are common among the most expensive listings - location, amenities and so on.

We also wanted to analyse the impact of reviews and if the actual content of the written reviews is significant.

The key questions we aim to answer through our project are: 
1. What are the most important factors influencing pricing?
2. What is the correlation between these factors and prices?
3. Why is or isn’t there a correlation?
4. Do reviews actually matter?
5. How can we use machine learning to help us predict prices of rental properties?
 

## Data
### What data do we want?

To answer our questions within our aims, we need data on the reviews of the most demanded and expensive listings and we also need data on the different features that a listing possesses. For example, the number of guests it hosts, whether the owner is a superhost, which amenities the listing includes and the location of the property. Of course, with thousands of listings across London comes millions of reviews under each one so it was important to limit our dataset. In order to make it easier to analyse, we took a sample of the reviews but we made sure that we would still have sufficient data to analyse.

### Time period

COVID-19 created a significant setback to listings and revenues so we decided it would be appropriate to conduct our analysis on data after the severity of the pandemic reduced therefore the dataset we are using was compiled on 9 September 2021 in order to give us a clearer idea of the factors affecting a price and to create more accurate price forecasting using machine learning.

## Getting Our Data

![](insideairbnbgettingdata.png)

Initially we did face some challenges with finding a suitable dataset as sources such as Kaggle did not provide us with comprehensive enough information and missed out crucial points such as amenities.

The datasets that we have used for this project come from InsideAirbnb.com which is a mission-driven activist project that scrapes Airbnb reviews and listings across different cities globally. InsideAirbnb provided us with two relevant CSV files on both listings and reviews. We chose the detailed version of the dataset over the summarised version as it included a wider array of data points such as the number of bedrooms and the types of amenities (wifi, heating, free parking, hot water etc ). These datasets were initially formed to highlight illegal listings on the platform that Murray Cox believed were distorting the housing market. It delivers curated statistics to cities that seek to rein in the home-sharing market. The datasets from InsideAirbnb can be considered factual as it is scraped directly from the Airbnb website and the publisher does not alter the data. The data is unlikely to be biased as it is all quantitative and publicly available. 

We used 2 datasets shown below. The screenshots show a randomly selected sample of each raw dataset opened in Excel.

1. Listings.csv

This dataset contains information about Airbnb listings, including neighbourhood, property type and so on. There are 66000 rows and 74 columns. These columns were the factors we analysed against median prices. 

![](wholelistingscsv.png)

2. Reviews.csv

This dataset contains written comments by guests about their experiences in a particular Airbnb. There are over 1 million rows of reviews, with some listings (given by listing_id) having multiple reviews. 

![](wholereviewscsv.png)

Sample of a negative review: (polarity score < 0):

![](negativecommentpolarity.png)

![](negativecomment.png)

Sample of a neutral review (polarity score = 0):

![](neutralcommentpolarity.png)

![](neutralreview.png)

Sample of a positive review (polarity score > 0):

![](positivecommentpolarity.png)

![](postivecomment.png)

### Obtaining Data

In order to conduct our analysis and visualisations from our data we used Python to format the raw dataset and then organised and filtered out unwanted columns.

First 5 rows of the formatted dataframe in Jupyter notebook (df.head(n=5)): 

Listings
![](listingscsv.png)

Reviews
![](obtainingreviewsdata.png)

### Limitaitons of Dataset

1. There may be possible bias in data as it could lack context. For example, not all listings are ‘active’, which means that just because an apartment is listed and included in the data, does not mean it’s available to rent out. Also, the data does not take into consideration the fact that multiple listings may be advertising the same property. However, academics have argued that the creator still portrays a fair and accurate representation of Airbnb
2. In terms of limitations of our analysis, the data is based on advertised/sticker prices rather than what is actually paid so this slightly affected our results. There were some extreme values of prices that had an impact on some calculations and visualisations therefore we decided to use median prices instead of mean prices to limit the effect of outliers.

Example of outlier:
![](priceoutlier.png)

3. The sheer number of reviews (over a million) poses a technical shortcoming as it would be too difficult to conduct sentiment analysis on all of these. As a result of this, our analysis does not include all of the comments but instead a random sample of the top comments that were included in the most expensive listings. 
4. Furthermore, there is great subjectivity to what makes a positive or negative review. To combat this we used Textblob sentiment to create polarity scores for the reviews. A score greater than 0 indicated a positive review and a score less than 0 was taken to be a negative review. 

### Descriptive Statistics of dataset

Descriptive statistics are useful as they can help us understand the collective properties of the elements of a data sample to gauge a better idea of the overall picture a dataset gives us. 

Overall we were looking at both continuous and discrete numerical data in terms of pricing, number of guests etc and nominal categorical textual data in the reviews, locations and amenities.

Central Tendency

The median price of all listings was £85. We opted to focus on median prices over mean prices to avoid the problems caused by outliers and extreme values within the dataset. 

Dispersion

There was a wide range of prices across the listings ranging from ____ - our limit of £1000

## Methodology

We used Python and Jupyter Notebook to analyse the datasets. The first thing we focused on was cleaning the data to remove erroneous, irrelevant, incorrectly formatted or duplicated information that would negatively influence our analysis and exploration. The steps we followed are detailed in the code, but the main things we focused on were dropping unwanted columns and null values or inconsistent entries from certain columns, converting data types where applicable as well as limiting prices. We were careful not to manipulate data too much but instead ensured that the data was cleaned only to the extent that it could be more easily and appropriately analysed, which was especially important as our project relied heavily on graphs.

This was followed by data exploration and modelling using the matplotlib and seaborn packages through the plotting of graphs such as bar charts and and maps. We analysed each factor separately - these included the number of bedrooms, location, the number of listings per host, the availability of the airbnb and so on - by determining their effect on the median price of the Airbnb listing. While we originally plotted graphs of factors against mean prices, we eventually decided to use median prices, which we found to be a better measure because it is less skewed by extreme values. Even after reducing the maximum price from £16700 to £1000, the range of prices which starts from £7 is still very wide, and thus median prices provide a more faithful representation.

As delineated in our code, we chose graph types based on what we felt suited the data best and that enabled us to visualise it and determine a correlation in the most appropriate way, whether it was a map, bar or scatter plot.

In some graphs such as that of the minimum number of nights versus price, we identified extreme outliers which hindered our ability to identify a pattern. To avoid changing the data completely, we removed outliers using the interquartile method but still kept the original graph to compare the two. While we understood that ‘true outliers’ represent natural variations and should be kept in the dataset, in this case, we removed them to get a clearer idea of the distribution of the scatter points. However, in most of our other graphs, we did not change the results even if it did not match our predictions at all, because this would have reduced the accuracy and validity of our analysis.

In the reviews data, the comments were cleaned and preprocessed  by filtering out the empty comments, removing the unicode characters, changing it to lowercase, removing the windows to new lines, removing all the stop words, and replacing the amount of spaces to one space. Given that all of the comments were in sentences, they were split into words using the word_tokenize in NLTK library. The words were captured using the Count_Vectorizer() and was created as a separate dataframe to explore the amount of times those top 20 words occurred. This will make it easier to create WordCloud of the top words in the ‘comments’ column.

In comparison of the reviews_score_ratings and price, the prices were changed into floats and plotted the graph using matplotlib.

To analyse the names of properties with top 100 cheapest and expensive listings, we selected ‘names’ and ‘price’ from the listings data. Then, we create 2 dataframes for both the top 100 expensive and cheapest listings, which include the ‘names’. After preprocessing the dataframes using the same steps for cleaning the comments with NLTK, then WordCloud was created for both dataframes with the help of matplotlib.

To calculate the subjectivity and polarity of the 10000 comments in the reviews data, we used Textblob sentiment after cleaning the textual data.  Then, we analysed the polarity scores to see whether it is a negative comment as the score is less than zero, neutral comment as the score is equal to zero, or a positive comment as the score is greater than zero.

For the amenities, the data was cleaned by removing the NaN values and then the amenities section was changed into lists and was able to string replace the brackets and the apostrophes to create a set of all possible amenities.

Then, we grouped each amenity into a certain category, if they have the same words/meaning but are in different letter cases or words. Here is a screenshot of the process:

![](pythonforgroupingamenities.png)

Then, we used the train_test split model(70:30 split) and accuracy scores to calculate the Feature Importance using Decision Tree and Random Forest classifier, which are available in the “scikit-learn” library of the python-based open source data analytics platform and the plots were created using Matplotlib.

Then, the NaN values were replaced by zeros in the columns. We drop the infrequent columns by creating a list of the amenity columns that have fewer  than 10% of listings. This led to our remaining amenities that were analysed below for features importance.

For the top amenities, the data was cleaned by removing the NaN values and then the amenities were separated using the numpy concentrate to count the top 20 amenities. Then, we plotted this using matplotlib.

For predicting prices, the process is explained below. The sklearn package was used to calculate each model (Linear Regression, Multiple Regression, Random Forest, KNN and Decision Tree. First, exclude the properties with listed prices of zero values. After, we replaced null values with zero values for “reviews_per month” and “host_listing_count”, and then encoded categorical values as integers.  The dataset was horizontally split into training and test data sets and also split into features (=X_train and X_test) and target (y_train and y_test). 

## Results

### Background

Before starting the project, we did some research on how Airbnb prices are generally set. While Airbnb gives a few recommendations to hosts, such as to conduct market research, consider the location and amenities, ultimately the host has their own personal pricing strategy and decides the price based on what they think matters the most.

Airbnb also has a Smart Pricing tool, which is an algorithm that generates price tips from metrics related to the property and recommends an ‘ideal’ price. The tool takes into account local demand, amenities, the number and ratings of reviews, room type, calendar availability and so on.

The issue with our project is that the prices we analysed are only from a specific point in time (when the data was scraped), which might not be extremely accurate as prices are constantly fluctuating and being updated.

Some Airbnbs also had additional service or cleaning fees which we realised could have significant effects on prices, especially because they are a flat fee rather than a nightly cost.

*Prices above £1000 are due to Airbnb hosts not understanding how to set 'sticker' prices correctly. Hence, we set a maximum price limit of £1000. Due to limitations of the data obtained from Inside Airbnb, our model will analyse advertised prices rather than prices actually paid, as we are unable to obtain this information.  

### General overiview of datasets

**1. Distribution of Median Prices**

![](distributionofmedianprices.png)

Just to give a general picture of the prices of the Airbnb listings in the dataset, we plotted a graph to demonstrate the distribution. The vertical dotted line indicates the median price of all the listings (£85), which seems reasonable, and the distribution is skewed to the right with prices mostly under £200 a night. This is useful to us in the analysis of factors against price, as we were able to determine whether the results were more similar or different to what was expected and whether they were reasonable, especially given the fact that outliers sometimes altered results. 

**2. Heat Map**

![](heatmapcorrelation.png)

We plotted a heat map which we hoped would give us a general overview or indication of which factors affect prices the most, and found that the top few included the number of people it accommodates, bedrooms, beds and the total of host listings. The top 3, however, were expected as undoubtedly the price of an Airbnb for 5 people would normally be more expensive than that for 1 as it would be bigger, have more bedrooms and so on. Hence, we can also see strong positive correlations between accommodates, bedrooms and beds. A limitation of this is that it included factors with numerical values.

### Zooming in - Sub Problem 1: What is the effect of Individual Factors of a Property on Median Price?

**1. Is the host a superhost?**

![](hostissuperhost.png)

This was a surprising result. Being an Airbnb consumer ourselves, we tend to look at whether the host is a superhost or not before making a final decision. However, based on the graphs, we can see that being a superhost does not actually have a positive effect on the prices of the Airbnb listings, but instead has a negative one.

**2. Is the host’s identity verified?**







## Conclusion

## Appendix

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

