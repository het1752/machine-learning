import os

import streamlit as st
import pandas as pd
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the CSV file containing music labels
csv_path = "Indian_music_dataset/genrenew/music_labels.csv"
df = pd.read_csv(csv_path)

# Feature extraction from audio files
def extract_features(file_path, max_length=100):
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]  # Extract only the first dimension
    # Reshape spectral_centroid to match the shape of mfccs
    spectral_centroid_resized = np.resize(spectral_centroid, mfccs.shape[1])

    # Pad or truncate features to a fixed length
    if mfccs.shape[1] < max_length:
        mfccs_padded = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
        spectral_centroid_resized = np.pad(spectral_centroid_resized, (0, max_length - len(spectral_centroid_resized)),
                                           mode='constant')
    else:
        mfccs_padded = mfccs[:, :max_length]
        spectral_centroid_resized = spectral_centroid_resized[:max_length]

    combined_features = np.concatenate((np.mean(mfccs_padded, axis=0), spectral_centroid_resized))
    return combined_features


# Initialize the classifiers
classifiers = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Gaussian Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# Train the classifiers
X = []
y = df['Label']
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

for idx, row in df.iterrows():
    file_name = row['File_Name']
    file_path = os.path.join("Indian_music_dataset/genrenew", row['Label'], file_name)
    audio_features = extract_features(file_path)
    X.append(audio_features)

X = np.array(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

trained_models = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    trained_models[name] = clf
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy}")

# Load the NLP model for recommendation
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Label'])

# Function to recommend similar songs
def recommend_similar_songs(predicted_genre):
    predicted_genre_vector = tfidf_vectorizer.transform([predicted_genre])
    similarity_scores = cosine_similarity(predicted_genre_vector, tfidf_matrix)
    similar_song_indices = similarity_scores.argsort()[0][-6:-1][::-1]  # Exclude the input song itself
    similar_songs = df.iloc[similar_song_indices]['File_Name']
    return similar_songs.tolist()

# Streamlit web application
st.title('Indian Music Genre Classifier and Recommender')
for i in range(6):
    uploaded_file = st.file_uploader("Upload an MP3 file", type=["mp3"])

    if uploaded_file is not None:
        try:
            # Extract features from the uploaded file
            audio_features = extract_features(uploaded_file)

            # Display classification results for each model
            for name, clf in trained_models.items():
                predicted_genre = clf.predict([audio_features])
                st.write(f'{name} Predicted Genre:', label_encoder.inverse_transform(predicted_genre)[0])

            # Recommend similar songs using NLP
            predicted_genre_label = label_encoder.inverse_transform(predicted_genre)[0]
            recommended_songs = recommend_similar_songs(predicted_genre_label)
            st.write('Recommended Songs:', recommended_songs)

        except Exception as e:
            st.error(f"An error occurred: {e}")
