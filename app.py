import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud

# Load data
df = pd.read_excel("data.xlsx")

# KPIs
total = len(df)
promoters = (df['nps_category'].str.lower() == 'promoter').sum()
detractors = (df['nps_category'].str.lower() == 'detractor').sum()
nps_value = ((promoters/total) - (detractors/total)) * 100

st.title("Customer Experience Dashboard")
st.metric("Net Promoter Score (NPS)", f"{nps_value:.1f}")
st.metric("Promoters", f"{promoters/total*100:.1f}%")
st.metric("Detractors", f"{detractors/total*100:.1f}%")

# NPS trend
trend = df.groupby("wave")["Q"].mean().reset_index()
fig = px.line(trend, x="wave", y="Q", title="NPS Trend by Wave")
st.plotly_chart(fig)

# Ratings
ratings = df[['A','B','C','D','E']].mean().reset_index()
fig2 = px.bar(ratings, x='index', y=0, title='Average Ratings (A–E)')
st.plotly_chart(fig2)

# Word Cloud
text = " ".join(df['Comment'].dropna())
wc = WordCloud(width=800, height=400).generate(text)
st.image(wc.to_array(), caption="Customer Feedback Word Cloud")
