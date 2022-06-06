
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
- Location is an important factor that affects prices - the most expensive boroughs are in the centre of London (Kensington, Chelsea, City of London, Westminster)
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

## Methodology
## Results

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

