import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import plotly.graph_objs as go
import plotly.express as px

st.title('Uber NYC January 2015')
st.image('https://d3i4yxtzktqr9n.cloudfront.net/uber-sites/f452c7aefd72a6f52b36705c8015464e.jpg', caption='Uber logo')

#Reading the data from the csv file and creating a dataframe
path = "https://raw.githubusercontent.com/uber-web/kepler.gl-data/master/nyctrips/data.csv" 
df = pd.read_csv(path, delimiter = ',')

st.header('Data explanation')
st.write('This dataset is about uber trip in NYC on the month of January 2015. It contains 12 columns and 1,000,000 rows. The data is from the website: https://www.kaggle.com/fivethirtyeight/uber-pickups-in-new-york-city')

st.subheader('Data preparation')
st.write("I dont show you how I clean my data because it's not the purpose of my streamlit web app but you can use the following commands to clean your data:")
st.code("df.dropna(), df.drop_duplicates(), df.describe(), df.info(), df.isnull().sum(), df.head()...")

# Data preocessing with pandas
#Transforming the date/time column to datetime format
df['tpep_pickup_datetime'] = df['tpep_pickup_datetime'].map(pd.to_datetime)
df['tpep_dropoff_datetime'] = df['tpep_dropoff_datetime'].map(pd.to_datetime)

# Find trip duration
df['trip_duration'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
# Only keep the trip duration in minutes
df['trip_duration'] = df['trip_duration'].dt.total_seconds()/60

# Creating a function for finding hour of date/time column example:
def get_hour(dt):
    return dt.hour

# Extracting the hour from the date/time column
df['hour'] = df['tpep_pickup_datetime'].map(get_hour)

# Correct total_amount column
df['total_amount'] = df['fare_amount'] + df['tip_amount']

# Creating a function to count the number of rows in a dataframe
def count_rows(rows):
    return len(rows)

df_by_date = df.groupby('hour').apply(count_rows)

# Data vizualisation 
st.header('Data vizualisation')
st.write('Here is a barplot of the number of trips by trip duration:')

# Create histogram trace
histogram_trace = go.Histogram(x=df.trip_duration, nbinsx=100, xbins=dict(start=0, end=100))

# Create layout
layout = go.Layout(title='Frequency by Trip Duration - Uber - January 2015', xaxis=dict(title='Trip Duration'), yaxis=dict(title='Frequency'))

# Create figure
fig = go.Figure(data=[histogram_trace], layout=layout)

# Display plot in Streamlit
st.plotly_chart(fig)

st.write('Here is a barplot of the number of trips by trip distance:')
# Plot histogram
fig, ax = plt.subplots()
ax.hist(df.trip_distance, bins = 100, range = (0, 30))
ax.set_xlabel('Trip Distance')
ax.set_ylabel('Frequency')
ax.set_title('Frequency by Trip Distance - Uber - January 2015')
st.pyplot(fig)

st.write('Plotly version :') 
fig = px.histogram(df, x='trip_distance', nbins=100, range_x=(0, 30), title='Frequency by Trip Distance - Uber - January 2015')
fig.update_layout(xaxis_title='Trip Distance', yaxis_title='Frequency')

st.plotly_chart(fig)

st.write('Here is different boxplot:')
# Box plot of the trip distance
fig, ax = plt.subplots()
ax.boxplot(df.trip_distance, vert=False)
ax.set_xlabel('Trip Distance')
ax.set_title('Boxplot of the Trip Distance - Uber - January 2015')

# Show plot using Streamlit
st.pyplot(fig)

# Box plot of passenger count
fig, ax = plt.subplots()
ax.boxplot(df.passenger_count, vert=False)
ax.set_xlabel('Passenger Count')
ax.set_title('Boxplot of the Passenger Count - Uber - January 2015')

st.pyplot(fig)

st.write('Here is a barplot of the number of trips by hour of the day:')
# Show number of trip by Hour of the day
fig, ax = plt.subplots()
df.groupby('hour').apply(count_rows).plot(ax=ax)
ax.set_xlabel('Hour of the day')
ax.set_ylabel('Trip amount')
ax.set_title('Number of trip by Hour - Uber - January 2015')

# Show plot using Streamlit
st.pyplot(fig)

# Define mean_rows function
def mean_rows(rows):
    return np.mean(rows.total_amount)

# Show average amount of trips by hour of the day
fig, ax = plt.subplots()
df.groupby('hour').apply(mean_rows).plot(ax=ax)
ax.set_xlabel('Hour of the day')
ax.set_ylabel('Trip amount')
ax.set_title('Average amount of trips by hour - Uber - January 2015')

# Show plot using Streamlit
st.pyplot(fig)


# Check the distribution of the longitude and latitude pickup
fig, ax = plt.subplots(figsize=(10, 10), dpi=70)
ax.set_title('Longitude and Latitude pickup distribution - Uber NYC- January 2015', fontsize=15)
ax.hist(df['pickup_longitude'], bins=100, range=(-74.1, -73.9), color='g', alpha=0.5, label='Longitude')
ax.legend(loc='best')
ax2 = ax.twiny()
ax2.hist(df['pickup_latitude'], bins=100, range=(40.5, 41), color='r', alpha=0.5, label='Latitude')
ax2.legend(loc='upper left')

# Show plot using Streamlit
st.pyplot(fig)

# Scatter plot pickup_latitude and pickup_longitude
fig, ax = plt.subplots(figsize=(15, 15), dpi=70)
ax.set_title('Scatter plot pickup_latitude and pickup_longitude - Uber NYC - January 2015', fontsize=20)
ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.scatter(df['pickup_latitude'], df['pickup_longitude'], s=0.8, alpha=0.4)
ax.set_ylim(-74.1, -73.8)
ax.set_xlim(40.7, 40.9)

# Show plot using Streamlit
st.pyplot(fig)

st.write('Check correlation between variables:')

# Check correlation between the length of the distance and tip amount
fig, ax = plt.subplots(figsize=(10, 10), dpi=70)
ax.set_title('Tip amount by distance of trip - Uber NYC - January 2015', fontsize=15)
ax.set_xlabel('Distance')
ax.set_ylabel('Tip Amount')
ax.scatter(df['trip_distance'], df['tip_amount'], s=0.8, alpha=0.4)

# Show plot using Streamlit
st.pyplot(fig)

# Use seaborn to plot the correlation matrix between the length of the distance and tip amount
fig, ax = plt.subplots(figsize=(10, 10), dpi=70)
ax.set_title('Correlation matrix between the length of the distance and tip amount - Uber NYC - January 2015', fontsize=15)
sns.heatmap(df[['trip_distance', 'tip_amount']].corr(), annot=True, fmt='.2f', cmap='coolwarm')

# Show plot using Streamlit
st.pyplot(fig)

st.header('MAPS')
st.write('Here is a map of the pickup locations:')
# Create a map of NYC
m = folium.Map(location=[40.7128, -74.0060], zoom_start=10)
# Show the heatmap of the pickup location
pickup_map = df[['pickup_latitude', 'pickup_longitude']]
pickup_map = pickup_map.values.tolist()
HeatMap(pickup_map).add_to(m)

# Show map using Streamlit
st.markdown(m._repr_html_(), unsafe_allow_html=True)

st.header('Compare data between vendors')
# Split the data in vendor_id 
df3_1 = df[df['VendorID'] == 1]
df3_2 = df[df['VendorID'] == 2]

# Compare number of course between vendor_id
fig, ax = plt.subplots(figsize=(10, 10), dpi=70)
ax.set_title('Number of course by vendor_id - Uber NYC - January 2015', fontsize=15)
ax.set_xlabel('Vendor ID')
ax.set_ylabel('Number of course')

ax.hist(df3_1['VendorID'], bins=20, range=(0, 2), color='g', alpha=0.5, label='Vendor ID 1')
ax.legend(loc='best')

ax2 = ax.twiny()
ax2.hist(df3_2['VendorID'], bins=20, range=(0, 2), color='r', alpha=0.5, label='Vendor ID 2')
ax2.legend(loc='upper left')

st.pyplot(fig) 

# Plot histogram of number of trips by hour for each vendor
fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
ax.hist(df3_1['hour'], bins = 100, range = (0, 24), color = 'g', alpha = 0.5, label = 'Vendor ID 1')
ax.hist(df3_2['hour'], bins = 100, range = (0, 24), color = 'r',alpha = 0.5, label = 'Vendor ID 2')
ax.legend(loc = 'best')
ax.set_xlabel('Hour of the day')
ax.set_ylabel('Number of trips')
ax.set_title('Number of trips by hour of the day by vendor_id - Uber NYC - January 2015')
st.pyplot(fig)

# Compare the distribution of the trip duration between the two vendors
fig, ax = plt.subplots(figsize=(10, 10), dpi=70)
ax.set_title('Trip duration distribution by vendor - Uber NYC - January 2015', fontsize=15)
ax.hist(df3_1.trip_duration, bins = 100, range = (0, 80), color = 'g', alpha = 0.5, label = 'Vendor 1')
ax.legend(loc = 'best')
ax2 = ax.twiny()
ax2.hist(df3_2.trip_duration, bins = 100, range = (0, 80), color = 'r', alpha = 0.5, label = 'Vendor 2')
ax2.legend(loc = 'upper left')
st.pyplot(fig)

# Compare the amount of trip between vendors by hour of the day
fig, ax = plt.subplots(figsize=(10, 10), dpi=70)
ax.set_title('Total amount of trip by hour of the day by vendor - Uber NYC - January 2015', fontsize=15)
ax.set_xlabel('Hour of the day')
ax.set_ylabel('Amount of trip')
ax.hist(df3_1['total_amount'], bins=100, range=(0, 24), color='g', alpha=0.5, label='Vendor 1')
ax.legend(loc='best')
ax2 = ax.twiny()
ax2.hist(df3_2['total_amount'], bins=100, range=(0, 24), color='r', alpha=0.5, label='Vendor 2')
ax2.legend(loc='upper left')
st.pyplot(fig)


