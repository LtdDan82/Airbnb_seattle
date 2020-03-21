# -*- coding: utf-8 -*-
"""
Spyder Editor

Dies ist eine temporäre Skriptdatei.
"""


import pandas as pd
import numpy as np
import calendar
import re
import seaborn as sns

from sklearn.model_selection import train_test_split


data_calendar = r'https://raw.githubusercontent.com/LtdDan82/Airbnb_seattle/master/data_seattle/calendar.csv'
data_listings = r'https://raw.githubusercontent.com/LtdDan82/Airbnb_seattle/master/data_seattle/listings.csv'
data_reviews = r'https://raw.githubusercontent.com/LtdDan82/Airbnb_seattle/master/data_seattle/reviews.csv'
#%%
def get_csv_from_git(path_to_git):
    df = pd.read_csv(path_to_git)
    
    return df
#%%

#df_cal = get_csv_from_git(data_calendar)
#df_listings = get_csv_from_git(data_listings) 
#df_reviews = get_csv_from_git(data_reviews) 

#%%
def get_avg_price_per_month():
    '''
    # Possible Question: When was the most expensive month for AirBnb in Seattle in 2016?
    '''
    df_cal = get_csv_from_git(data_calendar)
    # 1./2. Business/Data Understanding:
    
    # This could be interesting for travelling to Seattle (vacation or business).
    # Or when it is most beneficial to offer your flat
    #
    # Aggregate these listings by average price per month
    # Generate a bar plot
    #
    # 3. Data Preparation
    # Generate datetime object from date in df_cal
    df = df_cal.copy(deep = True)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
    # Check for any 'NaT' values:
    if not np.count_nonzero(df['date']) == len(df['date']):
        print('Found NaT values')
    # Get month and year as separate columns:
    df['month'] = [x.month for x in df['date']]
    df['year'] = [x.year for x in df['date']]
    #Rename month to Jan, Feb, ...
    df['month'] = df['month'].apply(lambda x: calendar.month_abbr[x])
    # price is currently a string --> want a int/float for aggregation functions
    # first get only the numeric part of the string
    df['price_in_$'] = df['price'].str.extract('(\d+)')
    df['price_in_$'] = pd.to_numeric(df['price_in_$'], errors='coerce')
    # We only want to consider the year 2016 , so drop year 2017
    df = df[df['year'] == 2016]
    final_df = df[['listing_id', 'price_in_$', 'month']]
    # Drop rows with no price
    final_df = final_df.dropna(subset = ['price_in_$'], axis = 0)
    
    # 4.  Modeling (here just some statistics)
    # Get mean prices per month and sort by month
    monthly_mean = final_df.groupby('month')['price_in_$'].mean()
    
    # 5. Evaluation
    # Graphical evaluation for communicating with stakeholders (medium)
    cats = ['Jan', 'Feb', 'Mar',
            'Apr','May','Jun',
            'Jul', 'Aug','Sep',
            'Oct', 'Nov', 'Dec']
    
    monthly_mean.index = pd.CategoricalIndex(monthly_mean.index,
                                             categories=cats, ordered=True)
    monthly_mean = monthly_mean.sort_index()
    ax = monthly_mean.plot.bar(title = 'Seattle Average AirBnB prices per month',
                               color = 'gray')
    ax.set_ylabel('Price in $')
    # Calculate top three travelling months
    top_months = monthly_mean.nsmallest(3).index.tolist()
    print('The top three months to save money' + 
          'for your accomodation in Seattle are %s, %s and %s.'
          % (top_months[0], top_months[1], top_months[2]))

#%%
def get_host_review_corr():
    df_listings = get_csv_from_git(data_listings)     
    # 1./2. Business/Data Understanding
    '''
    Is there a correlation between the host profile and the review_scores_rating on AirBnB ?
    '''
    # All columns which you can influence easily by yourself when making an AirBnB-host account should partly influence your review score
    df = df_listings.copy(deep = True)
    df = df[['host_response_time', 'host_response_rate', 'host_is_superhost', 'host_has_profile_pic', 'review_scores_rating']]

    # 3. Data Preparation
    # 3.1 General
    #Remove rows from data where response value = NaN
    df = df.dropna(subset = ['review_scores_rating'], axis = 0)
    # 3.2 Host Response Rate
    # Transform the host_response_rate column from precentage of type "str" into decimals from 0 ... 1
    df['host_response_rate'] = df['host_response_rate'].str.extract('(\d+)')
    # Insert "0" for NaN values in column 'host_response_rate',
    # I assume that NaN means that the host never responded.
    # This information should not be lost
    df['host_response_rate'].fillna(value = str("0"), inplace = True)
    # Convert type "str" to decimals 0 ... 1
    df['host_response_rate'] = df['host_response_rate'].astype(float)/100
    #3.3 Superhost and has_pic
    # Map t/f values in is_superhost and has_pic column to Boolean True and False
    d = {'t': True, 'f': False}
    convert_cols = ['host_is_superhost', 'host_has_profile_pic']
    df[convert_cols] = df[convert_cols].apply(lambda x: x.map(d))
    # 4. Check the correlation    
    sns.heatmap(df.corr(), annot = True, fmt = '.2f')
    # get the correlation matrix values for review score rating
    rev_score_corr = df.corr()['review_scores_rating']
    top_2_corr =rev_score_corr.nlargest(3)
    # Evaluate
    print('Correlation with Superhost: ', round(top_2_corr['host_is_superhost'],2))
    print('Correlation with Response Rate: ', round(top_2_corr['host_response_rate'],2))
    print('\n')
    print('In summary this means that other factors apart from what the host' + 
          'can do contribute more to the review rating, but this was expected'+
          'However if, e.g. the location of your apartment isnt that great' + 
          'you can increase your chances to get customers with a good profile')
    
    return df

#%%
df = get_host_review_corr()
'''Lets try to predict the review value based on our last results''' 

# Prepare Data from Host Response Time
df_rsp = df['host_response_time']
non_responded = df_rsp.isnull().sum()
# 312 did not respond in any time of the 4 categories --> Did never respond
# This information should not be lost --> get_dummies with NaN
df_rsp = pd.get_dummies(df_rsp, prefix = 'rsptime', prefix_sep ='_',
                        dummy_na = True)
#Exchange the original column in the df with the dummy data
df_dummy = pd.concat([df_rsp, df.drop(columns = 'host_response_time')], axis = 1)

y = df_dummy['review_scores_rating']
X = df_dummy.loc[:, df_dummy.columns != 'review_scores_rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,
                                                    shuffle = True,
                                                    random_state = 42)

# Select a model and so on





   














