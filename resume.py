# train_model_naive_bayes.py
import pandas as pd
import spacy
import pickle
import gc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

dataset_path = "static/UpdatedResumeDataSet.csv"
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def preprocess_text(text):
    if not isinstance(text, str):  
        return ""
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

# Load dataset
df = pd.read_csv(dataset_path)
df["Processed_Resume"] = df["Resume"].apply(preprocess_text)

# TF-IDF + Naive Bayes
vectorizer = TfidfVectorizer(max_features=3000)
X_tfidf = vectorizer.fit_transform(df["Processed_Resume"])
y = df["Category"]

model = MultinomialNB()
model.fit(X_tfidf, y)

# Save model & vectorizer
with open("static/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("static/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

df.to_pickle("static/processed_resumes.pkl")
print("Naive Bayes model and vectorizer saved.")
