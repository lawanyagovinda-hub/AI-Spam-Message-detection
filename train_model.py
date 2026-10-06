import os
import re
import zipfile
import requests
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle


# ==============================
# 1. Create folders
# ==============================

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)


# ==============================
# 2. Download dataset
# ==============================

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

zip_path = "data/sms_spam.zip"

if not os.path.exists("data/SMSSpamCollection"):

    print("Downloading dataset...")

    response = requests.get(url)

    with open(zip_path, "wb") as f:
        f.write(response.content)

    print("Dataset downloaded.")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall("data")

else:
    print("Dataset already exists.")


# ==============================
# 3. Load dataset
# ==============================

df = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="latin-1"
)

print("\nDataset loaded!")
print("Total messages:", len(df))


# ==============================
# 4. Clean text
# ==============================

def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["clean_message"] = df["message"].apply(clean_text)


# ==============================
# 5. Convert labels
# ==============================

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})


# ==============================
# 6. Split data
# ==============================

X = df["clean_message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==============================
# 7. TF-IDF
# ==============================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# ==============================
# 8. Train AI model
# ==============================

model = MultinomialNB()

model.fit(X_train_tfidf, y_train)


# ==============================
# 9. Evaluate
# ==============================

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL RESULTS")
print("==============================")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Not Spam", "Spam"]
    )
)


# ==============================
# 10. Save model
# ==============================

with open("models/spam_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)


print("\n==============================")
print("MODEL SAVED SUCCESSFULLY!")
print("==============================")

print("models/spam_model.pkl")
print("models/tfidf_vectorizer.pkl")
