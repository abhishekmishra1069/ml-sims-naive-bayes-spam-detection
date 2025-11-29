# Import vectorize from numpy (though not used in this script)
from numpy import vectorize
# Import pandas for data manipulation and CSV file reading
import pandas as pd
# Import accuracy_score from sklearn metrics to evaluate model performance
from sklearn.metrics import accuracy_score
# Import train_test_split to divide data into training and testing sets
from sklearn.model_selection import train_test_split
# Import MultinomialNB classifier for text classification (Naive Bayes)
from sklearn.naive_bayes import MultinomialNB
# Import CountVectorizer to convert text into numerical feature vectors
from sklearn.feature_extraction.text import CountVectorizer
# Import joblib for saving and loading trained models and vectorizers
import joblib

# --- 1. Data Loading ---
# Read the CSV file containing SMS messages labeled as spam or ham
df = pd.read_csv('spam.csv')
# Extract the 'Message' column as features (input text messages)
X = df['Message']
# Extract the 'Category' column as labels (spam or ham classification)
y = df['Category']
# Split data into training (80%) and testing (20%) sets with random_state=42 for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Feature Extraction (CountVectorizer) ---
# CountVectorizer converts text messages into a matrix of word counts (features).
# Create an instance of CountVectorizer for converting text to numerical features
vectorizer = CountVectorizer()
# Fit the vectorizer on training data and transform it into a sparse matrix of word counts
X_train_vectorized = vectorizer.fit_transform(X_train)
# Transform test data using the same vectorizer fitted on training data
X_test_vectorized = vectorizer.transform(X_test)

# Save the fitted Vectorizer! It is crucial for transforming new input text later.
# Persist the vectorizer to disk so it can be loaded by the Flask app for predictions
joblib.dump(vectorizer, 'vectorizer.pkl')
# Optional debug print statement (commented out)
# print("✅ CountVectorizer Fitted and Saved.")

# --- 3. Model Training (Multinomial Naive Bayes) ---
# Create a Multinomial Naive Bayes classifier instance
model = MultinomialNB()
# Train the model on the vectorized training data and corresponding labels
model.fit(X_train_vectorized, y_train)

# Persist the trained model to disk so it can be loaded by the Flask app for predictions
joblib.dump(model, 'nb_spam_classifier_model.pkl')

# --- 4. Evaluation ---
# Use the trained model to predict labels on the test set
y_pred = model.predict(X_test_vectorized)
# Calculate and print the accuracy score (percentage of correct predictions on test data)
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred):.2f}")

