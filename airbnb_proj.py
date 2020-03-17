# -*- coding: utf-8 -*-
"""
Spyder Editor

Dies ist eine temporäre Skriptdatei.
"""


import pandas as pd
import numpy as np
import calendar
import re


data_calendar = r'https://raw.githubusercontent.com/LtdDan82/Airbnb_seattle/master/data_seattle/calendar.csv'
data_listings = r'https://raw.githubusercontent.com/LtdDan82/Airbnb_seattle/master/data_seattle/listings.csv'
data_reviews = r'https://raw.githubusercontent.com/LtdDan82/Airbnb_seattle/master/data_seattle/reviews.csv'
#%%
def get_csv_from_git(path_to_git):
    df = pd.read_csv(path_to_git)
    
    return df
#%%

df_cal = get_csv_from_git(data_calendar)
df_listings = get_csv_from_git(data_listings) 
df_reviews = get_csv_from_git(data_reviews) 

#%%


# 1. Business Understanding:
# Possible Question: When was the most expensive month for AirBnb in Seattle in 2016?
# This could be interesting for travelling to Seattle (vacation or business).
# 2. Data Understanding:
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
cats = ['Jan', 'Feb', 'Mar', 'Apr','May','Jun', 'Jul', 'Aug','Sep', 'Oct', 'Nov', 'Dec']
monthly_mean.index = pd.CategoricalIndex(monthly_mean.index, categories=cats, ordered=True)
monthly_mean = monthly_mean.sort_index()
ax = monthly_mean.plot.bar(title = 'Seattle Average AirBnB prices per month', color = 'gray')
ax.set_ylabel('Price in $')
# Calculate top three travelling months
top_months = monthly_mean.nsmallest(3).index.tolist()
print('The top three months to save money for your accomodation in Seattle are %s, %s and %s.'
      % (top_months[0], top_months[1], top_months[2]))



   














