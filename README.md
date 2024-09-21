# conversational-insights-platform

**Overview**:
The Conversational Insights Platform is a web application designed to analyze audio and video files, transcribing the content and extracting meaningful insights. It utilizes Natural Language Processing (NLP) techniques to perform sentiment analysis, topic extraction, and emotion visualization.

**Features**

Audio/Video Upload: Upload MP3, WAV, or MP4 files for analysis.

Transcription: Automatically transcribes spoken content into text.

Sentiment Analysis: Analyzes the sentiment of the transcribed text (positive, negative, neutral).

Topic Extraction: Identifies key topics discussed in the transcription using Latent Dirichlet Allocation (LDA).

Emotion Visualization: Displays the sentiment scores with visual indicators.

Metadata Extraction: Provides metadata such as duration and frame rate for audio and video files.

**Technologies Used**

Streamlit: For building the web application interface.

SpeechRecognition: For transcribing audio content.

MoviePy: For extracting audio from video files.

NLTK: For natural language processing tasks.

Transformers: For sentiment analysis using pre-trained models.

Scikit-learn: For implementing topic modeling using LDA.

Mutagen: For reading audio file metadata

**Installation**

**Clone the repository:**
git clone https://github.com/SHERYL267/conversational-insights-platform.git  
cd conversational-insights-platform

**Create and activate a virtual environment:**
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

**Install the required packages:**
pip install -r requirements.txt

**Start the Streamlit application:**
streamlit run app.py
