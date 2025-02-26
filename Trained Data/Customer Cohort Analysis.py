#!/usr/bin/env python
# coding: utf-8

# # Customer Cohort Analysis
# 
# 
# Cohort analysis is a useful technique to understand customer behavior over time. It involves grouping customers into cohorts based on their first purchase date and then analyzing their behavior over subsequent periods. Below is a Python script using pandas and matplotlib to perform cohort analysis on the provided dataset.

# ## Step 1: Import Libraries

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import calendar
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter


# ## Step 2: Load the Dataset

# In[10]:


import pandas as pd
import warnings

warnings.filterwarnings("ignore")

Customer_data_df = pd.read_csv(r"C:\Users\jki\Desktop\Data Scence Projects\Customer Segmentaion Cohort Analysis\Machine Learnign Project\Source Data\customer orders.csv", encoding="ISO-8859-1")
Customer_data_df.head(5)


# ## Step 3: Convert Order Datetime to Datetime Format

# In[11]:


# Convert 'Order Datetime' to datetime format
Customer_data_df['Order Datetime'] = pd.to_datetime(Customer_data_df['Order Datetime'], format='%m/%d/%Y')


# ## Step 4: Extract the Cohort (Month of First Purchase)

# In[12]:


# Extract the cohort (month of first purchase) for each customer
Customer_data_df['Cohort'] = Customer_data_df.groupby('Email Adress')['Order Datetime'].transform('min').dt.to_period('M')


# ## Step 5: Calculate the Cohort Index (Months Since First Purchase)

# # Calculate the time offset for each order within the cohort
# Customer_data_df['Cohort Index'] = (Customer_data_df['Order Datetime'].dt.to_period('M') - Customer_data_df['Cohort']).apply(lambda x: x.n)

# ## Step 6: Group by Cohort and Cohort Index

# In[14]:


# Group by Cohort and Cohort Index, then count the number of unique customers
cohort_data = Customer_data_df.groupby(['Cohort', 'Cohort Index'])['Email Adress'].nunique().reset_index()


# ## Step 7: Pivot the Data to Create a Cohort Matrix

# In[15]:


# Pivot the data to create a cohort matrix
cohort_pivot = cohort_data.pivot(index='Cohort', columns='Cohort Index', values='Email Adress')


# ## Step 8: Visualize the Cohort Analysis

# In[17]:


# Plot the cohort analysis
plt.figure(figsize=(12, 8))
sns.heatmap(cohort_pivot, annot=True, fmt='.0f', cmap='Blues', linewidths=0.5)
plt.title('Cohort Analysis - Customer Retention')
plt.xlabel('Cohort Index (Months since first purchase)')
plt.ylabel('Cohort (Month of first purchase)')
plt.show()


# ##  Step 9 : Develop a discount recommedation system based on cohort analysis

# In[19]:


Customer_data_df.info()


# In[21]:


# Calculate retention rates
cohort_size = cohort_pivot.iloc[:, 0]
retention_matrix = cohort_pivot.divide(cohort_size, axis=0)


# Define Discount Recommendation Logic
# 
# If a customer has a low retention rate (e.g., less than 50% after 3 months), offer a discount.
# 
# Otherwise, do not offer a discount.

# In[22]:


def recommend_discount(customer_name, Customer_data_df, retention_matrix):
    # Find the customer's email address
    customer_data = Customer_data_df[Customer_data_df['Full Name'] == customer_name]
    if customer_data.empty:
        return "Customer not found."
    
    email = customer_data['Email Adress'].iloc[0]
    
    # Find the cohort and cohort index for the customer
    cohort = customer_data['Cohort'].iloc[0]
    cohort_index = customer_data['Cohort Index'].iloc[0]
    
    # Check retention rate at a specific cohort index (e.g., 3 months)
    retention_rate = retention_matrix.loc[cohort, 3] if 3 in retention_matrix.columns else 0
    
    # Recommend discount based on retention rate
    if retention_rate < 0.5:
        return f"Offer discount to {customer_name}. Retention rate after 3 months: {retention_rate:.2%}."
    else:
        return f"Do not offer discount to {customer_name}. Retention rate after 3 months: {retention_rate:.2%}."


# In[27]:


def recommend_discount(customer_name, Customer_data_df, retention_matrix, download=False, filepath=None):
    """
    Generate a detailed discount recommendation in JSON format based on customer data and retention metrics.
    Optionally download the recommendation as a JSON file.
    
    Parameters:
    - customer_name (str): The full name of the customer
    - Customer_data_df (pandas.DataFrame): DataFrame containing customer information
    - retention_matrix (pandas.DataFrame): Matrix of retention rates by cohort and time period
    - download (bool): Whether to download the recommendation as a JSON file
    - filepath (str): Path where the JSON file should be saved. If None and download=True, 
                      a default path will be used based on the customer name
    
    Returns:
    - dict: JSON-serializable dictionary with recommendation details
    """
    import json
    import os
    from datetime import datetime
    
    # Initialize the result dictionary
    result = {
        "customer_name": customer_name,
        "found": False,
        "recommendation": None,
        "data": {},
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Find the customer's data
    customer_data = Customer_data_df[Customer_data_df['Full Name'] == customer_name]
    if customer_data.empty:
        result["message"] = "Customer not found."
        if download:
            _save_json(result, customer_name, filepath)
        return result
    
    # Mark customer as found and collect basic information
    result["found"] = True
    result["data"]["email"] = customer_data['Email Adress'].iloc[0]
    result["data"]["cohort"] = cohort = customer_data['Cohort'].iloc[0]
    result["data"]["cohort_index"] = cohort_index = customer_data['Cohort Index'].iloc[0]
    
    # Get retention rates for various time periods if available
    retention_periods = [1, 3, 6] if all(x in retention_matrix.columns for x in [1, 3, 6]) else retention_matrix.columns
    result["data"]["retention_rates"] = {}
    
    for period in retention_periods:
        if period in retention_matrix.columns:
            rate = retention_matrix.loc[cohort, period]
            result["data"]["retention_rates"][f"month_{period}"] = round(rate, 4)
    
    # Determine discount recommendation based on 3-month retention rate
    if 3 in retention_matrix.columns:
        retention_rate = retention_matrix.loc[cohort, 3]
        result["data"]["target_retention_rate"] = round(retention_rate, 4)
        
        if retention_rate < 0.5:
            result["recommendation"] = "offer_discount"
            result["discount_percentage"] = 15 if retention_rate < 0.3 else 10
            result["reason"] = f"Low retention rate after 3 months: {retention_rate:.2%}"
        else:
            result["recommendation"] = "no_discount"
            result["reason"] = f"Good retention rate after 3 months: {retention_rate:.2%}"
    else:
        # If 3-month data not available, use the latest available period
        latest_period = max(retention_periods)
        retention_rate = retention_matrix.loc[cohort, latest_period]
        result["data"]["target_retention_rate"] = round(retention_rate, 4)
        
        if retention_rate < 0.5:
            result["recommendation"] = "offer_discount"
            result["discount_percentage"] = 15 if retention_rate < 0.3 else 10
            result["reason"] = f"Low retention rate after {latest_period} months: {retention_rate:.2%}"
        else:
            result["recommendation"] = "no_discount"
            result["reason"] = f"Good retention rate after {latest_period} months: {retention_rate:.2%}"
    
    # Download the JSON if requested
    if download:
        _save_json(result, customer_name, filepath)
    
    return result


def _save_json(data, customer_name, filepath=None):
    """
    Helper function to save JSON data to a file.
    
    Parameters:
    - data (dict): The data to be saved
    - customer_name (str): The customer name for the default filename
    - filepath (str): Optional custom filepath. If None, a default path will be used
    
    Returns:
    - str: The path where the file was saved
    """
    import json
    import os
    from datetime import datetime
    
    # Create a sanitized filename from the customer name
    safe_name = ''.join(c if c.isalnum() else '_' for c in customer_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use provided filepath or create a default one
    if not filepath:
        # Create a 'discount_recommendations' directory if it doesn't exist
        output_dir = 'discount_recommendations'
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{safe_name}_{timestamp}.json")
    
    # Save the JSON data
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Discount recommendation saved to: {filepath}")
    return filepath


# Example usage:
# result = recommend_discount("John Doe", customer_df, retention_df, download=True)
# result = recommend_discount("Jane Smith", customer_df, retention_df, download=True, filepath="custom_location/jane_recommendation.json")


# In[ ]:




