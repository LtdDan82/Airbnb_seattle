# -*- coding: utf-8 -*-
"""
Spyder Editor

Dies ist eine temporäre Skriptdatei.
"""


import pandas as pd
import numpy as np
import calendar
#import re
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

#from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split, GridSearchCV
#from sklearn import linear_model
#from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
#from sklearn.neural_network import MLPRegressor, MLPClassifier
#from sklearn.metrics import r2_score, accuracy_score



data_calendar = r'https://raw.githubusercontent.com/LtdDan82/Airbnb_seattle/master/data_seattle/calendar.csv'
data_listings = r'https://raw.githubusercontent.com/LtdDan82/Airbnb_seattle/master/data_seattle/listings.csv'
data_reviews = r'https://raw.githubusercontent.com/LtdDan82/Airbnb_seattle/master/data_seattle/reviews.csv'
#%%
def get_csv_from_git(path_to_git):
    '''Input: path to *.csv file as rawstring
       Reads *.csv file with pandas class method "read_csv" 
       Output: pandas dataframe
    '''
    df = pd.read_csv(path_to_git)
    
    return df
#%%

#df_cal = get_csv_from_git(data_calendar)
#df_listings = get_csv_from_git(data_listings) 
#df_reviews = get_csv_from_git(data_reviews) 

#%%
def get_avg_price_per_month():
    '''
    1st Question: When is the most financially beneficial time to offer your apartment to a guest ?
    - Loads dataset "calendar" from github and creates a dataframe
    - Data is cleansed and dates are converted from type(str) to datetime objects
    - Mean price value per month in the year 2016 is calculated
    - A bar plot is created vor visualization
    - Returns: top_months
    
    '''
    df_cal = get_csv_from_git(data_calendar)
  
    
    # 1. Data Preparation
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
    
    # 2.  Modeling (here just some statistics)
    # Get mean prices per month and sort by month
    monthly_mean = final_df.groupby('month')['price_in_$'].mean()
    
    # 3. Evaluation
    # Graphical evaluation for communicating with stakeholders 
    cats = ['Jan', 'Feb', 'Mar',
            'Apr','May','Jun',
            'Jul', 'Aug','Sep',
            'Oct', 'Nov', 'Dec']
    
    monthly_mean.index = pd.CategoricalIndex(monthly_mean.index,
                                             categories=cats, ordered=True)
    monthly_mean = monthly_mean.sort_index()
    plt.figure(figsize=(1400/96, 787/96), dpi=96)
    ax = monthly_mean.plot.bar(title = 'Seattle Average AirBnB prices per month',
                               color = 'gray')
    ax.set_ylabel('Price in $')
    plt.savefig('cost_per_month.png')
    # Calculate top three travelling months
    top_months = monthly_mean.nlargest(3).index.tolist()
    return top_months
    
#%%
# Answer to question #1    
top_months = get_avg_price_per_month()
print('The top three months to maximize profit' + 
          ' for your accomodation in Seattle are %s, %s and %s.'
          % (top_months[0], top_months[1], top_months[2]))    

#%%
def get_host_review_corr():
    df_listings = get_csv_from_git(data_listings)     
    # 1./2. Business/Data Understanding
    '''
    2nd Question: Is there a correlation between the host profile and the review_scores_rating on AirBnB ?
    - Loads dataset "listings" from github and creates a dataframe
    - Prepares the data (removing NaN from target value: review_scores, maps boolean values, ...)
    - Visualizes the correlation with review_scores_rating
    - Visualizes the difference in mean review scores for the top 2 features
    '''
    
    # All columns which you can influence easily by yourself when making an AirBnB-host account should partly influence your review score
    df = df_listings.copy(deep = True)
    host_info_cols = ['host_response_time',
                      'host_response_rate', 
                      'host_is_superhost', 
                      'host_has_profile_pic',
                      'host_about',                     
                      'review_scores_rating']
    
    
    df = df[host_info_cols]
    # 1. Data Preparation
    # 1.1 General
    #Remove rows from data where response value = NaN
    df = df.dropna(subset = ['review_scores_rating'], axis = 0)
    # 1.2 Host Response Rate
    # Transform the host_response_rate column from precentage of type "str" into decimals from 0 ... 1
    df['host_response_rate'] = df['host_response_rate'].str.extract('(\d+)')
    # Insert "0" for NaN values in column 'host_response_rate',
    # I assume that NaN means that the host never responded.
    # This information should not be lost
    df['host_response_rate'].fillna(value = str("0"), inplace = True)
    # Convert type "str" to decimals 0 ... 1
    df['host_response_rate'] = df['host_response_rate'].astype(float)/100
    # 1.3 Superhost and has_pic
    # Map t/f values in is_superhost and has_pic column to Boolean True and False
    d = {'t': True, 'f': False}
    convert_cols = ['host_is_superhost', 'host_has_profile_pic']
    df[convert_cols] = df[convert_cols].apply(lambda x: x.map(d))
    #3.4 host_about, True when host has a description, else False
    df['host_about'] = df['host_about'].notna()
        
    # 2. Evaluate - Check the correlation
    plt.figure(figsize=(1400/96, 1400/96), dpi=96)    
    sns.heatmap(df.corr(), annot = True, fmt = '.2f')
    plt.savefig('host_para_correlation.png')

    # get the correlation matrix values for review score rating
    rev_score_corr = df.corr()['review_scores_rating']
    # Find the top 2 in terms of correlation to review score rating
    top_2_corr = rev_score_corr.nlargest(4)
    top_2_corr.sort_values(ascending = False, inplace = True)
    top_1, top_2 = top_2_corr.index[[1,2]]
    # 3. Evaluate
    
    # Compare the mean values of 'review_scores_rating' for the top1 and top2
    top1_means = df.groupby(by=top_1)['review_scores_rating'].mean()
    top2_means = df.groupby(by=top_2)['review_scores_rating'].mean()

      
    # Plotting
    plt.figure(figsize=(1400/96, 787/96), dpi=96)    
    title = 'Comparison of mean review scores for feature: ' + top_1
    ax1 = top1_means.plot.bar(title = title)
    ax1.set_ylim(bottom = 80, top = 100)
    ax1.set_ylabel('review_scores_rating')
    plt.savefig('rev_scores_superhost.png')
    
    plt.figure(figsize=(1400/96, 787/96), dpi=96)
    title = 'Comparison of mean review scores for feature: ' + top_2
    ax2 = top2_means.plot.bar(title = title)
    ax2.set_ylim(bottom = 80, top = 100)
    ax2.set_ylabel('review_scores_rating')
    plt.savefig('rev_scores_host_about')
  
    return top_2_corr, top_1, top_2


#%%
# Answer question 2
top_2_corr, top_1, top_2 = get_host_review_corr()

print('Correlation with %s: %.2f' % (top_1, round(top_2_corr[top_1], 2)))
print('Correlation with %s: %.2f' % (top_2, round(top_2_corr[top_2], 2)))

print('\n')
print('The top 2 correlating "host parameters" are %s and %s' % (top_1, top_2))
print('As a host you should become a superhost and write some short stuff about yourself, so that people get to know you')
#%%
def additional_parameters():
    '''3rd Question: What are other important parameters that influence your review score ?
    - Reads in the dataset "Listings" and creates a pandas df
    - Checks the average review score for each neighbourhood + Visualization
    - Checks if the number of amenities in your apartment have a positive influence on the review score + Visualization
    - 
    '''
    
    
    
    df_listings = get_csv_from_git(data_listings)
    influences = ['neighbourhood_group_cleansed', 'amenities',
                  'accommodates', 'square_feet', # sqaure_feet per accomodate --> more space
                  'review_scores_rating']
    
    new_df = df_listings[influences]
    
    # 1. Data Preparation
    # As an example we check if the neighborhood plays a certain role and if the
    # number of amenities that the host offers can increase your review_score rating
    
    # 1.1 General
    #Remove rows from data where response value = NaN
    new_df = new_df.dropna(subset = ['review_scores_rating'], axis = 0)
    
    # 1.2 Amenities columns - Count the number of amenities
    # Clean the amenities column for further processing: Count the number of amenities
    new_df['amenities'] = new_df['amenities'].str.replace('{', '')
    new_df['amenities'] = new_df['amenities'].str.replace('}', '')
    new_df['amenities'] = new_df['amenities'].str.replace('"', '')
    #new_df['amenities'] = new_df['amenities'].str.replace('', [])
    
    # Transform to list of strings
    new_df['amenities'] = new_df['amenities'].apply(lambda x: x.split(","))
    # Insert length of list, except if the list has only the empty string: ''
    new_df['amenities'] = new_df['amenities'].apply(lambda x: len(x) if x != [''] else 0)
    
    # 1.3 What is the best neighborhood to have an apartment ?
    col = 'neighbourhood_group_cleansed'
    best_neighborhoods = new_df.groupby(col)['review_scores_rating'].mean()
    
    plt.figure(figsize=(1400/96, 787/96), dpi=96)
    title = 'Best Neighbourhoods by Review Score Rating'
    ax_nbh = best_neighborhoods.plot.bar(title = title, color = 'blue')
    ax_nbh.set_ylabel('Review Score Rating')
    ax_nbh.set_ylim(bottom = 80, top = 100)
    plt.savefig('neighbourhoods_rev_score.png')
    
    
    
    # 1.4 Does the number of amenities also influence your review score ?
    # For that we first need to find out the how large the sample size of the 'amenties' columns is
    count_amenities = new_df.groupby('amenities')['review_scores_rating'].count()
    
    # Evaluation / Visualization
    plt.figure(figsize=(1400/96, 787/96), dpi=96)
    ax_amen_count = count_amenities.plot.bar(color = 'black')
    ax_amen_count.set_ylabel('Number of samples')
    plt.savefig('sample_size_amenities.png')
    # We can clearly see that the sample size of very few 
    # and very many amenities is rather low
    # Consequently we only take samples with a sample size larger than 50
    # into account in order to answer the question
    filter_counts = count_amenities[count_amenities > 50].index.tolist()
    filter_df = new_df[new_df['amenities'].isin(filter_counts)]
    min_amenities = filter_df.groupby('amenities')['review_scores_rating'].mean()
    
    plt.figure(figsize=(1400/96, 787/96), dpi=96)
    ax_amen = min_amenities.plot(title = 'Influence of amenities on review_score_rating')
    ax_amen.set_ylabel('average review_score_rating')
    ax_amen.set_xlabel('number of amenities')
    plt.savefig('numberAmenities_vs_rev_score.png')
    
    return best_neighborhoods, min_amenities

#%%
best_neighborhoods, min_amenities = additional_parameters()
print('From the neighborhood plot we can see that %s seems not that satisfying for people.' %(best_neighborhoods.nsmallest().index[0]))
print('From the amenities plot we can see that more amenities seem to have a positive influence on the review score')  
#%%




#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
##%% Funzt leider nicht
#def try_to_model(df):
#    '''Lets try to predict the review value based on our last results''' 
#
#    # Prepare Data from Host Response Time
#    df_rsp = df['host_response_time']
#    non_responded = df_rsp.isnull().sum()
#    # 312 did not respond in any time of the 4 categories --> Did never respond
#    # This information should not be lost --> get_dummies with NaN
#    df_rsp = pd.get_dummies(df_rsp, prefix = 'rsptime', prefix_sep ='_',
#                            dummy_na = True)
#    #Exchange the original column in the df with the dummy data
#    df_dummy = pd.concat([df_rsp, df.drop(columns = 'host_response_time')], axis = 1)
#    
#    # Introduce a new column "ratio better than average, for a binary classification"
#    mean_rev_score = np.mean(df_dummy['review_scores_rating'])
#    df_dummy['ratio_better_than_avg'] = df_dummy['review_scores_rating']>mean_rev_score
#    # Drop review_scores_rating column to prevent Data Leakage during Training of the model
#    df_dummy = df_dummy.drop(columns = 'review_scores_rating')
#    
#    sns.heatmap(df_dummy.corr(), annot = True, fmt = '.2f')
#    
#    # for regression:
#    #y = df_dummy['review_scores_rating']
#    #X = df_dummy.loc[:, df_dummy.columns != 'review_scores_rating']
#    
#    #for classification:
#    y = df_dummy['ratio_better_than_avg']
#    X = df_dummy.loc[:, df_dummy.columns != 'ratio_better_than_avg']
#    
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,
#                                                        shuffle = True,
#                                                        random_state = 42)
#    
#    
#    #Scale the data
#    scaler = StandardScaler()
#    #Use  only TrainSet to prevent Data Leakage from the TestSet data into the scaling process
#    scaler.fit(X_train)
#    X_train_scaled = scaler.transform(X_train)
#    X_test_scaled = scaler.transform(X_test)
#    
#    # Several Classifiers tested
#    #%% REGRESSION STUFF
#    #clf = linear_model.Lasso(alpha = 0.9, normalize = True, random_state = 42)
#    #clf = linear_model.LogisticRegression(random_state = 42)
#    #clf = DecisionTreeRegressor(random_state = 42)
#    #clf = MLPRegressor(random_state = 42,
#    #                   max_iter=1000,
#    #                   learning_rate='adaptive',
#    #                   activation = 'tanh',
#    #                   alpha = 0.00001)
#    
#    #%% CLASSIFIERS
#    clf = DecisionTreeClassifier(random_state = 42)
#    #clf = MLPClassifier(random_state = 42,
#    #                   max_iter=1000,
#    #                   learning_rate='adaptive',
#    #                   activation = 'tanh',
#    #                   alpha = 0.00001)
#    
#    #%% Fit and Evaluate
#    #clf.fit(X_train_scaled, y_train)
#    #y_predict = clf.predict(X_test_scaled)
#    #score = r2_score(y_test, y_predict)
#    #accuracy = accuracy_score(y_test, y_predict)
#    
#    # Tune the model with an easy GridSearch
#    
#    #%%
#    param_grid = {'min_samples_split': [2, 4, 8],
#                  'max_depth': [None, 5, 10],
#                  'max_features': [2, 3, 4]}
#    best_clf = GridSearchCV(estimator = clf, param_grid = param_grid)
#    best_clf.fit(X_train_scaled, y_train)
#    y_predict = best_clf.predict(X_test)
#    accuracy = accuracy_score(y_test, y_predict)
#    cv_results = best_clf.cv_results_
#   


#    #3.3 Accomodates and square_feet column - Get the square_feet per person
#    # Get the ratio
#    new_df['living_space'] = new_df['square_feet'] / new_df['accommodates']
#    # Drop unnecessary columns 'accommodates' and 'square_feet'
#    new_df = new_df.drop(labels = ['accommodates', 'square_feet'], axis = 1)
#    
#    #sns.heatmap(new_df.corr(), annot = True, fmt = ".2f")











