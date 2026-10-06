import re
import pickle
import streamlit as st


# Load trained model
with open("models/spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


# Text cleaning function
def clean_text(text):
    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Page settings
st.set_page_config(
    page_title="AI Spam Message Detection",
    page_icon="🤖",
    layout="centered"
)


# Title
st.title("🤖 AI-Based Spam Message Detection")

st.write(
    "Enter a message below and our Machine Learning model "
    "will determine whether it is Spam or Not Spam."
)

st.divider()


# Message input
message = st.text_area(
    "📩 Enter your message:",
    height=150,
    placeholder="Example: Congratulations! You have won a free prize!"
)


# Prediction
if st.button("🔍 Check Message", use_container_width=True):

    if not message.strip():

        st.warning("Please enter a message.")

    else:

        # Clean message
        cleaned_message = clean_text(message)

        # Convert text to TF-IDF
        message_vector = vectorizer.transform(
            [cleaned_message]
        )

        # Make prediction
        prediction = model.predict(
            message_vector
        )[0]

        # Get probability
        probabilities = model.predict_proba(
            message_vector
        )[0]

        confidence = max(probabilities) * 100

        st.divider()

        # Display result
        if prediction == 1:

            st.error("🚨 SPAM MESSAGE DETECTED")

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            st.write(
                "⚠️ This message has characteristics "
                "commonly associated with spam."
            )

        else:

            st.success("✅ NOT SPAM")

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            st.write(
                "This message appears to be legitimate."
            )


# About section
st.divider()

st.subheader("📌 About the Project")

st.write("""
This application uses Artificial Intelligence and Natural
Language Processing to detect spam messages.

### Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorization
- Multinomial Naive Bayes
- Streamlit
- Natural Language Processing
""")
