import streamlit as st
import pandas as pd
import numpy as np

# Use caching to efficiently load data only once
@st.cache_data
def load_data():
   # Load 10,000 rows of data
   data = pd.DataFrame(
       np.random.randn(10000, 2) / [50, 50] + [37.76, -122.4],
       columns=['lat', 'lon']
   )
   return data

st.title('Simple Data Explorer')

# Load the data
df = load_data()

# 1. Add a Widget to control the data
st.subheader('Filter Data')
num_points = st.slider('Number of points to display', 100, 10000, 1000)

# 2. Filter the data based on the widget value
filtered_df = df.head(num_points)

# 3. Display the results
st.subheader(f'Displaying the first {num_points} data points')
st.dataframe(filtered_df)

# 4. Visualize the results on a map
st.map(filtered_df)


