# fake_news_detector.py

import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Title
st.title("📰 Fake News Detection")

@st.cache_data
def load_data():
    fake_df = pd.read_csv("Fake_small.csv")
    true_df = pd.read_csv("True_small.csv")

    
    fake_df["label"] = 0  # fake
    true_df["label"] = 1  # real
    
    data = pd.concat([fake_df, true_df], axis=0)
    data = data[["text", "label"]]
    return data

# Text cleaning function
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    return text

# Load and clean data
data = load_data()
data["text"] = data["text"].apply(clean_text)

# Option to display class distribution
if st.checkbox("Show Class Distribution"):
    st.subheader("Class Distribution")
    st.write(data["label"].value_counts())

# Pie Chart for Class Distribution
st.subheader("Class Distribution (Pie Chart)")
class_counts = data["label"].value_counts().rename(index={0: "Fake News", 1: "Real News"})
fig = px.pie(values=class_counts.values, names=class_counts.index, title="Fake vs Real News Distribution")
st.plotly_chart(fig)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.2, random_state=42
)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Logistic Regression with class balancing
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train_vec, y_train)

# Accuracy
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)

fig_cm, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"], ax=ax)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
st.pyplot(fig_cm)

# User Input
st.subheader("Check News Authenticity")
user_input = st.text_area("Enter news article text to check:", height=200)

if st.button("Check"):
    if user_input.strip():
        cleaned_input = clean_text(user_input)
        input_vec = vectorizer.transform([cleaned_input])
        prob = model.predict_proba(input_vec)[0][1]  # probability of being real
        prediction = model.predict(input_vec)[0]
        label = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: {prob:.2f}")
    else:
        st.warning("Please enter some text.")
