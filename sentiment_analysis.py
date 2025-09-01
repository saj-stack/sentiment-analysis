import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Dataset
data = {
    "review": [
        "Excellent quality",
        "Worst experience ever",
        "will not recommend to anyone",
        "Good quality and affordable",
        "Terrible Product",
        "Quality is stunning",
        "Not satisfied with this product",
        "Absolutely fantastic",
        "Very useful porduct",
        "Happy with my order",
        "Bad experience, waste of money",
        "The design is stylish",
        "Product gets dirty easily",
        "Works Greate",
        "Light weight and easy to handle",
        "Installation was complicated",
        "Poor quality Product",
        "Good Product"

    ],
    "sentiment": ["positive", "negative", "negative", "positive", "negative", "positive", "negative", "positive", "positive", 
                  "positive", "negative", "positive", "negative", "positive", "positive", "negative","negative", "positive"]
}

# DataFrame
reviews_dataset = pd.DataFrame(data)

# Features
X = reviews_dataset["review"]
y = reviews_dataset["sentiment"]

# Text to numerical features
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train and Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.3, random_state=10
)

# Train Model (Naive Bayes)
model = MultinomialNB()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# User Input
print("\nEnter your feedback for analysis::")
user_review = input("Review: ")

# Prediction
user_vectorized = vectorizer.transform([user_review])
prediction = model.predict(user_vectorized)

print("\nSentiment Prediction:", prediction[0].capitalize())