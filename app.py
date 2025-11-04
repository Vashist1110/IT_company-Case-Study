import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------
# Streamlit App Setup
# ---------------------------------------------------
st.set_page_config(page_title="Customer Experience Dashboard (NPS)", layout="wide")

@st.cache_data
def load_excel(path):
    return pd.read_excel(path)

# ---------------------------------------------------
# Sentiment Scoring
# ---------------------------------------------------
def score_sentiment(text):
    positive_words = set("""good great excellent awesome amazing love loved satisfied happy helpful fantastic outstanding superb quick fast easy reliable prompt smooth friendly supportive appreciate appreciated appreciable recommend recommended""".split())
    negative_words = set("""bad poor terrible awful hate hated dissatisfied unhappy unhelpful horrible worst slow late delay delayed rude unfriendly unprofessional complicated complex difficult hard buggy crash crashed broken issue issues problem problems error errors not working waste useless disappointing disappointed""".split())
    if not isinstance(text, str):
        return 0
    tokens = re.findall(r"[a-z']+", text.lower())
    pos = sum(1 for t in tokens if t in positive_words)
    neg = sum(1 for t in tokens if t in negative_words)
    return pos - neg

def add_sentiment(df):
    if 'Comment' not in df.columns:
        df['Comment'] = None
    df['sentiment_score'] = df['Comment'].apply(score_sentiment)
    df['sentiment_label'] = df['sentiment_score'].apply(lambda x: 'Positive' if x>0 else ('Negative' if x<0 else 'Neutral'))
    return df

# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------
def wave_key(w):
    m = re.match(r"([12])H(\d{2})", str(w).strip())
    if not m:
        return (9999, 9)
    half = int(m.group(1))
    yr = 2000 + int(m.group(2))
    return (yr, half)

def top_terms(texts, n=15):
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return pd.DataFrame({'term':[], 'score':[]})
    vec = TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1,2), min_df=2)
    X = vec.fit_transform(texts)
    s = np.asarray(X.mean(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())
    order = np.argsort(s)[::-1][:n]
    return pd.DataFrame({'term': terms[order], 'score': s[order]})

# ---------------------------------------------------
# Data Input
# ---------------------------------------------------
st.title("Customer Experience Dashboard (NPS)")

src = st.radio("Data source", ["Bundled file", "Upload"], horizontal=True)
if src == "Bundled file":
    df = load_excel("data/data.xlsx")
else:
    up = st.file_uploader("Upload Excel", type=["xlsx"])
    if up is None:
        st.stop()
    df = pd.read_excel(up)

# Basic cleaning
df.columns = [c.strip() for c in df.columns]
for col in ['A','B','C','D','E','Q']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

if 'nps_category' in df.columns:
    df['nps_category'] = df['nps_category'].astype(str).str.title()

df = add_sentiment(df)
if 'wave' in df.columns:
    df['__wavesort__'] = df['wave'].apply(wave_key)

# ---------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------
st.sidebar.header("Filters")
waves = sorted(df['wave'].dropna().unique().tolist(), key=lambda x: wave_key(x)) if 'wave' in df.columns else []
cats = sorted(df['nps_category'].dropna().unique().tolist()) if 'nps_category' in df.columns else []
sents = ['Positive','Neutral','Negative']

fwave = st.sidebar.multiselect("Wave", waves, default=waves)
fcat = st.sidebar.multiselect("NPS Category", cats, default=cats)
fsent = st.sidebar.multiselect("Sentiment", sents, default=sents)

mask = pd.Series(True, index=df.index)
if fwave and 'wave' in df.columns: mask &= df['wave'].isin(fwave)
if fcat and 'nps_category' in df.columns: mask &= df['nps_category'].isin(fcat)
if fsent: mask &= df['sentiment_label'].isin(fsent)
dff = df[mask].copy()

# ---------------------------------------------------
# KPI Row
# ---------------------------------------------------
total = len(dff)
prom = (dff['nps_category'].str.lower()=='promoter').sum() if 'nps_category' in dff else 0
det = (dff['nps_category'].str.lower()=='detractor').sum() if 'nps_category' in dff else 0
pas = (dff['nps_category'].str.lower()=='passive').sum() if 'nps_category' in dff else 0
nps = ((prom/total) - (det/total))*100 if total else 0
avg_q = dff['Q'].mean() if 'Q' in dff else float('nan')

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Total Responses", f"{total}")
c2.metric("NPS", f"{nps:.1f}")
c3.metric("Promoters %", f"{(prom/total*100):.1f}" if total else "0")
c4.metric("Detractors %", f"{(det/total*100):.1f}" if total else "0")
c5.metric("Avg NPS (Q)", f"{avg_q:.2f}" if pd.notna(avg_q) else "NA")

st.markdown("---")

# ---------------------------------------------------
# Tabs
# ---------------------------------------------------
t1, t2, t3, t4, t5 = st.tabs(["Trends", "Attributes", "Sentiment", "Themes", "Comments"])

with t1:
    if {'wave','Q'}.issubset(dff.columns):
        trend = dff.sort_values('__wavesort__').groupby('wave', as_index=False)['Q'].mean()
        st.plotly_chart(px.line(trend, x='wave', y='Q', markers=True, title="Mean NPS (Q) by Wave"), use_container_width=True)
    if {'wave','nps_category'}.issubset(dff.columns):
        mix = dff.pivot_table(index='wave', columns='nps_category', values='respondent_id', aggfunc='count', fill_value=0)
        mix = (mix.div(mix.sum(axis=1), axis=0)*100).reset_index().melt('wave', var_name='Category', value_name='Percent')
        st.plotly_chart(px.bar(mix, x='wave', y='Percent', color='Category', barmode='stack', title="Category Mix by Wave"), use_container_width=True)

with t2:
    rating_cols = [c for c in ['A','B','C','D','E'] if c in dff.columns]
    if rating_cols:
        ratings = dff[rating_cols].mean().reset_index()
        ratings.columns = ['Question','Mean']
        st.plotly_chart(px.bar(ratings, x='Question', y='Mean', title="Average Ratings (A–E)"), use_container_width=True)
        corr = dff[['Q'] + rating_cols].corr()
        st.plotly_chart(px.imshow(corr, text_auto=True, title="Correlation (Q with A–E)"), use_container_width=True)

with t3:
    sent = dff['sentiment_label'].value_counts().reindex(['Positive','Neutral','Negative']).fillna(0).reset_index()
    sent.columns = ['Sentiment','Count']
    st.plotly_chart(px.pie(sent, names='Sentiment', values='Count', hole=0.4, title="Sentiment Distribution"), use_container_width=True)
    if dff['Comment'].notna().any():
        text = " ".join(dff['Comment'].dropna().astype(str))
        wc = WordCloud(width=1100, height=300).generate(text)
        st.image(wc.to_array(), caption="Feedback Word Cloud", use_column_width=True)

with t4:
    prom_texts = dff.loc[dff['nps_category'].str.lower()=='promoter', 'Comment'] if 'nps_category' in dff else pd.Series([])
    det_texts = dff.loc[dff['nps_category'].str.lower()=='detractor', 'Comment'] if 'nps_category' in dff else pd.Series([])
    c1, c2 = st.columns(2)
    top_pro = top_terms(prom_texts.dropna().astype(str), n=15)
    top_det = top_terms(det_texts.dropna().astype(str), n=15)
    with c1:
        st.subheader("Promoters – Top Terms")
        if len(top_pro): st.dataframe(top_pro)
        else: st.write("No promoter comments.")
    with c2:
        st.subheader("Detractors – Top Terms")
        if len(top_det): st.dataframe(top_det)
        else: st.write("No detractor comments.")

with t5:
    cols = [c for c in ['respondent_id','nps_category','Q','wave','A','B','C','D','E','sentiment_label','Comment'] if c in dff.columns]
    st.dataframe(dff[cols].reset_index(drop=True))
    st.download_button("Download filtered CSV", data=dff[cols].to_csv(index=False), file_name="filtered_responses.csv", mime="text/csv")
