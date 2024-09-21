import streamlit as st
import speech_recognition as sr
import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from nltk.tokenize import word_tokenize
from collections import Counter
from moviepy.editor import VideoFileClip
import mutagen
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Download necessary NLTK resources if not already done
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

# Initialize sentiment analysis model
sentiment_model = pipeline("sentiment-analysis")

def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    
    # Tokenize and remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
    return text

def transcribe_video(file_path):
    video = VideoFileClip(file_path)
    audio_file = "temp_audio.wav"
    video.audio.write_audiofile(audio_file)
    transcription = transcribe_audio(audio_file)
    os.remove(audio_file)  # Remove temp audio file
    return transcription

def analyze_sentiment(transcription):
    results = sentiment_model(transcription)
    return results

def extract_topics(transcription):
    # Preprocess the transcription text
    clean_text = preprocess_text(transcription)
    
    # Create a document-term matrix
    vectorizer = CountVectorizer()
    dtm = vectorizer.fit_transform([clean_text])
    
    # Train LDA model
    lda = LatentDirichletAllocation(n_components=3, random_state=42)
    lda.fit(dtm)
    
    # Extract topics
    words = vectorizer.get_feature_names_out()
    topics = []
    
    for idx, topic in enumerate(lda.components_):
        topic_words = [words[i] for i in topic.argsort()[-5:]]
        topics.append(f"Topic {idx+1}: " + ', '.join(topic_words))
    
    return topics

def extract_audio_metadata(file_path):
    audio = mutagen.File(file_path)
    duration = audio.info.length  # Duration in seconds
    return {
        "duration": duration
    }

def extract_video_metadata(file_path):
    video = VideoFileClip(file_path)
    duration = video.duration  # Duration in seconds
    return {
        "duration": duration,
        "fps": video.fps,
        "n_frames": video.reader.nframes,
    }

def extract_metadata(file_path):
    if file_path.endswith('.mp4'):
        return extract_video_metadata(file_path)
    else:
        return extract_audio_metadata(file_path)

def visualize_emotions(sentiment_results):
    emotions = {}
    for result in sentiment_results:
        label = result['label']
        score = result['score']
        emotions[label] = emotions.get(label, 0) + score
    
    st.subheader("Emotion Analysis:")
    for emotion, score in emotions.items():
        color = 'green' if emotion == 'POSITIVE' else 'red'
        st.markdown(f"<span style='color:{color}; font-size:20px;'>{emotion}: {score:.2f}</span>", unsafe_allow_html=True)

def main():
    st.title("Conversational Insights Platform")

    uploaded_file = st.file_uploader("Upload Audio/Video File", type=['mp3', 'wav', 'mp4'])

    if uploaded_file is not None:
        with open("temp_file", "wb") as f:
            f.write(uploaded_file.getbuffer())

        if uploaded_file.type in ["audio/mp3", "audio/wav"]:
            st.success("Audio file uploaded successfully!")
            transcription = transcribe_audio("temp_file")
            st.subheader("Transcription:")
            st.write(transcription)

            # Sentiment Analysis
            sentiment_results = analyze_sentiment(transcription)
            st.subheader("Sentiment Analysis:")
            st.write(sentiment_results)

            # Visualize Emotions
            visualize_emotions(sentiment_results)

            # Topic Extraction
            topics = extract_topics(transcription)
            st.subheader("Extracted Topics:")
            for topic in topics:
                st.write(topic)

            # Extract metadata for audio
            metadata = extract_metadata("temp_file")
            st.subheader("Audio Metadata:")
            st.write(metadata)

        elif uploaded_file.type == "video/mp4":
            st.success("Video file uploaded successfully!")
            transcription = transcribe_video("temp_file")
            st.subheader("Transcription:")
            st.write(transcription)

            # Sentiment Analysis
            sentiment_results = analyze_sentiment(transcription)
            st.subheader("Sentiment Analysis:")
            st.write(sentiment_results)

            # Visualize Emotions
            visualize_emotions(sentiment_results)

            # Topic Extraction
            topics = extract_topics(transcription)
            st.subheader("Extracted Topics:")
            for topic in topics:
                st.write(topic)

            # Extract metadata for video
            metadata = extract_metadata("temp_file")
            st.subheader("Video Metadata:")
            st.write(metadata)

        os.remove("temp_file")

if __name__ == "__main__":
    main()
