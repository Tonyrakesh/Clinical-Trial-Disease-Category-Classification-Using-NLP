import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd
import altair as alt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK packages are downloaded locally in the app environment
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Initialize text prep objects
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(cleaned_tokens)

# App layout configuration
st.set_page_config(
    page_title="Medical NLP Disease Predictor",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Clinical Trial Disease Classifier")
st.markdown("Predict disease categories instantly from clinical trial brief summaries")

# Load artifacts
@st.cache_resource
def load_artifacts():
    # Using our top-performing LinearSVC model
    model = joblib.load('linear_svc_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return model, tfidf, label_encoder

try:
    model, tfidf, label_encoder = load_artifacts()
    st.success("LinearSVC model and TF-IDF pipeline successfully loaded!")
except Exception as e:
    st.error(f"Error loading models: {e}")

# Interactive interface
user_input = st.text_area(
    "Enter Clinical Trial Brief Summary:",
    placeholder="Type or paste a clinical trial brief summary here...",
    height=200
)

if st.button("Classify Disease Category", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter a brief summary first.")
    else:
        with st.spinner("Processing text & generating predictions..."):
            cleaned = preprocess_text(user_input)
            vectorized = tfidf.transform([cleaned])
            
            # Predict disease class
            predicted_class_idx = model.predict(vectorized)[0]
            predicted_category = label_encoder.inverse_transform([predicted_class_idx])[0]
            
            # Calculate confidence via decision function (since SVC doesn't natively have predict_proba)
            # Apply softmax to decision function scores to represent pseudo-probabilities
            decision_values = model.decision_function(vectorized)[0]
            exp_scores = np.exp(decision_values - np.max(decision_values))  # stable softmax
            probabilities = exp_scores / np.sum(exp_scores)
            confidence = probabilities[predicted_class_idx]
            
            # Main prediction showcase
            st.metric(label="Predicted Category", value=predicted_category.upper())
            st.metric(label="Confidence Score", value=f"{confidence * 100:.2f}%")
            
            # Visualize complete probability distribution chart
            st.markdown("### Probability Distribution across Categories")
            chart_data = pd.DataFrame({
                'Category': [cat.upper() for cat in label_encoder.classes_],
                'Probability (%)': probabilities * 100
            }).sort_values('Probability (%)', ascending=False)
            
            chart = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X('Probability (%)', title='Confidence (%)'),
                y=alt.Y('Category', sort='-x', title='Disease Category'),
                color=alt.Color('Probability (%)', scale=alt.Scale(scheme='viridis'))
            ).properties(height=300)
            
            st.altair_chart(chart, use_container_width=True)