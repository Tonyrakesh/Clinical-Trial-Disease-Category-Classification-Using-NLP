# Clinical Trial Disease Category Classification

An end-to-end Machine Learning and Natural Language Processing (NLP) project that classifies clinical trial brief summaries into corresponding disease categories using TF-IDF and supervised learning models, served through an interactive Streamlit web dashboard.

---

## 📌 Project Overview

Clinical trial descriptions contain unstructured medical text that can be difficult to organize manually. This project cleans and preprocesses raw trial summaries, transforms the text into numerical features using TF-IDF, trains machine learning models to classify the condition, and presents an interactive web application for real-time predictions.

### Key Highlights
- **Dataset Size:** 60,000+ clinical trial records across 8 disease categories.
- **Top Model:** Logistic Regression achieving **94.31% test accuracy** (compared against Multinomial Naive Bayes at 89.60%).
- **Interactive UI:** Built with Streamlit, showing predicted disease class, confidence score, and a probability distribution bar chart.

---

## 🦠 Target Disease Categories

The model classifies text into 8 categories:
1. Anxiety
2. Breast Cancer
3. Chronic Obstructive Pulmonary Disease (COPD)
4. COVID-19
5. Glaucoma
6. Rheumatoid Arthritis
7. Sickle Cell Anemia
8. Type 2 Diabetes

---

## 🛠️ Project Architecture & Pipeline

1. **Data Ingestion & Cleaning:**
   - Removal of duplicate entries and handling missing values.
2. **Text Preprocessing (NLP):**
   - Lowercasing and regex-based removal of special characters/numbers.
   - Word tokenization via NLTK.
   - Stop-word removal (English stopwords).
   - Lemmatization via NLTK's `WordNetLemmatizer`.
3. **Feature Extraction:**
   - `TfidfVectorizer` (top 5,000 features).
4. **Model Training & Evaluation:**
   - Multi-class classification using **Logistic Regression** and **Multinomial Naive Bayes**.
   - Performance evaluation via Accuracy, Precision, Recall, and F1-score.
5. **Model Serialization:**
   - Exporting the trained model, TF-IDF vectorizer, and label encoder using `joblib`.
6. **Deployment:**
   - Streamlit web interface for inference and confidence visualization.

---
