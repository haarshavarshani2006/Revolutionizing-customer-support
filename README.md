# Revolutionizing-customer-support
import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
model = joblib.load("chatbot_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("Intelligent Customer Support Chatbot")

user_input = st.text_input("You:", "How can I reset my password?")

if st.button("Get Response"):
    X_input = vectorizer.transform([user_input])
    intent = model.predict(X_input)[0]
    
    # Sample response map
    responses = {
        "reset_password": "Sure! Click on 'Forgot Password' at the login screen.",
        "order_status": "Please share your order ID to check the status.",
        "refund": "Your refund will be processed in 5-7 business days.",
    }

    response = responses.get(intent, "I'm sorry, I didn't understand that.")
    st.write("Bot:", response)
    import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Sample data (replace with actual dataset)
data = pd.DataFrame({
    "text": [
        "I want to reset my password",
        "Where is my order?",
        "I need a refund",
        "How do I cancel my order?",
    ],
    "intent": [
        "reset_password",
        "order_status",
        "refund",
        "cancel_order",
    ]
})

# Preprocessing
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["text"])
y = data["intent"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save artifacts
joblib.dump(model, "chatbot_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
