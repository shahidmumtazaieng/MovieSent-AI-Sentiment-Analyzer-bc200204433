import streamlit as st
import streamlit.components.v1 as components
import os
import json
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import nltk
import re
import string
from nltk.corpus import stopwords
import time
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="MovieSent - AI Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

download_nltk_data()

# Initialize stopwords
STOPWORDS = set(stopwords.words('english'))

# Cached model loading functions
@st.cache_resource
def load_logistic_regression_model():
    """Load the pre-trained Logistic Regression model"""
    try:
        model_path = 'models/lr_tfidf.joblib'
        if os.path.exists(model_path):
            return joblib.load(model_path)
        else:
            st.error(f"Logistic Regression model not found at {model_path}")
            return None
    except Exception as e:
        st.error(f"Error loading Logistic Regression model: {str(e)}")
        return None

@st.cache_resource
def load_tfidf_vectorizer():
    """Load the pre-trained TF-IDF vectorizer"""
    try:
        vectorizer_path = 'models/tfidf_vec.joblib'
        if os.path.exists(vectorizer_path):
            return joblib.load(vectorizer_path)
        else:
            st.error(f"TF-IDF vectorizer not found at {vectorizer_path}")
            return None
    except Exception as e:
        st.error(f"Error loading TF-IDF vectorizer: {str(e)}")
        return None

@st.cache_resource
def load_lstm_model():
    """Load the pre-trained LSTM model"""
    try:
        model_path = 'models/lstm_sent.h5'
        if os.path.exists(model_path):
            return load_model(model_path)
        else:
            st.error(f"LSTM model not found at {model_path}")
            return None
    except Exception as e:
        st.error(f"Error loading LSTM model: {str(e)}")
        return None

@st.cache_resource
def load_tokenizer():
    """Load the pre-trained tokenizer"""
    try:
        tokenizer_path = 'models/tokenizer.json'
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'r') as f:
                return json.load(f)
        else:
            st.error(f"Tokenizer not found at {tokenizer_path}")
            return None
    except Exception as e:
        st.error(f"Error loading tokenizer: {str(e)}")
        return None

def clean_text(text):
    """Clean and preprocess the input text"""
    text = re.sub(r'<.*?>', ' ', text)                    # Remove HTML tags
    text = re.sub(r'[^A-Za-z0-9 ]+', ' ', text)          # Keep alphanumeric chars
    text = text.lower().translate(str.maketrans('', '', string.punctuation))  # Lowercase & remove punctuation
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]  # Remove stopwords & short words
    return ' '.join(tokens)

def predict_logistic_regression(lr_model, tfidf_vectorizer, cleaned_text):
    """
    Predict sentiment using Logistic Regression model
    """
    try:
        # Transform the text using TF-IDF vectorizer
        text_vector = tfidf_vectorizer.transform([cleaned_text])
        
        # Get prediction probabilities
        prob_lr = lr_model.predict_proba(text_vector)[0]
        
        # Determine prediction and confidence
        pred_lr = 'Positive' if prob_lr[1] >= 0.5 else 'Negative'
        confidence_lr = round(max(prob_lr) * 100, 1)
        
        return pred_lr, confidence_lr, prob_lr
    except Exception as e:
        st.error(f"Error in Logistic Regression prediction: {str(e)}")
        return None, None, None

def predict_lstm(lstm_model, tokenizer, cleaned_text):
    """
    Predict sentiment using LSTM model
    """
    try:
        # Convert text to sequences using tokenizer
        tokenizer_obj = tokenizer_from_json(tokenizer)
            
        # Convert text to sequences
        seq = tokenizer_obj.texts_to_sequences([cleaned_text])
        
        # Pad sequences to fixed length (120 as per original model)
        seq = pad_sequences(seq, maxlen=120)
        
        # Get prediction probability
        prob_lstm = float(lstm_model.predict(seq)[0][0])
        
        # Determine prediction and confidence
        pred_lstm = 'Positive' if prob_lstm >= 0.5 else 'Negative'
        confidence_lstm = round(prob_lstm * 100, 1) if prob_lstm >= 0.5 else round((1 - prob_lstm) * 100, 1)
        
        return pred_lstm, confidence_lstm, prob_lstm
    except Exception as e:
        st.error(f"Error in LSTM prediction: {str(e)}")
        return None, None, None

def analyze_sentiment(review_text, lr_model, tfidf_vectorizer, lstm_model, tokenizer):
    """
    Perform sentiment analysis using both models
    """
    # Clean the input text
    cleaned_text = clean_text(review_text)
    
    # Return None values if cleaned text is empty
    if not cleaned_text.strip():
        return None, None, None, None, None, None
    
    # Get predictions from both models
    lr_pred, lr_conf, lr_prob = predict_logistic_regression(lr_model, tfidf_vectorizer, cleaned_text)
    lstm_pred, lstm_conf, lstm_prob = predict_lstm(lstm_model, tokenizer, cleaned_text)
    
    return lr_pred, lr_conf, lr_prob, lstm_pred, lstm_conf, lstm_prob

def create_sidebar():
    """Create sidebar with app information and settings"""
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3064/3064554.png", width=100)
        st.title("🎬 MovieSent AI")
        st.markdown("---")
        
        st.header("About")
        st.info(
            """
            **MovieSent** analyzes movie reviews using two different AI approaches:
            - 🟦 **Logistic Regression** with TF-IDF features
            - 🟩 **LSTM Neural Network** with sequence processing
            """
        )
        
        st.header("Settings")
        theme = st.selectbox("Choose Theme", ["Light", "Dark"])
        
        st.markdown("---")
        st.header("Model Info")
        st.success("**Logistic Regression**: Fast & interpretable")
        st.warning("**LSTM**: Deep learning with context awareness")
        
        st.markdown("---")
        st.markdown("### 🚀 Powered by Streamlit")

def create_main_interface():
    """Create the main interface for sentiment analysis"""
    # Main header
    st.markdown("<h1 style='text-align: center; color: #4361ee;'>🎬 MovieSent - AI Sentiment Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #808080;'>Dual-Model Sentiment Analysis for Movie Reviews</p>", unsafe_allow_html=True)
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📝 Text Input", "📋 Sample Reviews"])
    
    with tab1:
        user_input = st.text_area(
            "Enter your movie review:",
            height=150,
            placeholder="Type or paste your movie review here...",
            key="user_input"
        )
    
    with tab2:
        sample_reviews = [
            "This movie was absolutely fantastic! Great acting and storyline.",
            "Terrible film, waste of time. Poor acting and confusing plot.",
            "It was okay, nothing special but not bad either.",
            "Amazing cinematography and brilliant performances by the cast."
        ]
        selected_sample = st.selectbox("Choose a sample review:", sample_reviews)
        user_input = selected_sample if st.button("Use this sample") else user_input
    
    return user_input

def display_results(lr_pred, lr_conf, lstm_pred, lstm_conf):
    """Display prediction results in modern cards"""
    if lr_pred and lstm_pred:
        col1, col2 = st.columns(2)
        
        with col1:
            if lr_pred == "Positive":
                st.markdown(
                    f"""
                    <div style="
                        border-radius: 15px; 
                        padding: 20px; 
                        background: linear-gradient(135deg, #4361ee, #3a0ca3);
                        color: white;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                        border-left: 5px solid #ff9e00;
                    ">
                        <h3 style="color: white; margin-top: 0;">🟦 Logistic Regression</h3>
                        <h2 style="color: white; margin-bottom: 10px;">{lr_pred} ✨</h2>
                        <p style="color: #e9ecef; font-size: 18px;">Confidence: <strong>{lr_conf}%</strong></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style="
                        border-radius: 15px; 
                        padding: 20px; 
                        background: linear-gradient(135deg, #f72585, #b5179e);
                        color: white;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                        border-left: 5px solid #ff9e00;
                    ">
                        <h3 style="color: white; margin-top: 0;">🟦 Logistic Regression</h3>
                        <h2 style="color: white; margin-bottom: 10px;">{lr_pred} 😞</h2>
                        <p style="color: #e9ecef; font-size: 18px;">Confidence: <strong>{lr_conf}%</strong></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        with col2:
            if lstm_pred == "Positive":
                st.markdown(
                    f"""
                    <div style="
                        border-radius: 15px; 
                        padding: 20px; 
                        background: linear-gradient(135deg, #4cc9f0, #4895ef);
                        color: white;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                        border-left: 5px solid #ff9e00;
                    ">
                        <h3 style="color: white; margin-top: 0;">🟩 LSTM Neural Net</h3>
                        <h2 style="color: white; margin-bottom: 10px;">{lstm_pred} ✨</h2>
                        <p style="color: #e9ecef; font-size: 18px;">Confidence: <strong>{lstm_conf}%</strong></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style="
                        border-radius: 15px; 
                        padding: 20px; 
                        background: linear-gradient(135deg, #7209b7, #560bad);
                        color: white;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                        border-left: 5px solid #ff9e00;
                    ">
                        <h3 style="color: white; margin-top: 0;">🟩 LSTM Neural Net</h3>
                        <h2 style="color: white; margin-bottom: 10px;">{lstm_pred} 😞</h2>
                        <p style="color: #e9ecef; font-size: 18px;">Confidence: <strong>{lstm_conf}%</strong></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

def show_loading_animation():
    """Show a loading animation while processing"""
    with st.spinner('🤖 Analyzing your review with AI models...'):
        time.sleep(1)  # Simulate processing time

def create_comparison_chart(lr_conf, lstm_conf):
    """Create a comparison chart for model confidence"""
    data = {
        'Model': ['Logistic Regression', 'LSTM Neural Net'],
        'Confidence': [lr_conf, lstm_conf]
    }
    df = pd.DataFrame(data)
    
    st.subheader("📊 Model Confidence Comparison")
    st.bar_chart(data=df.set_index('Model'), height=300)

def show_model_accuracy_info():
    """Display information about model accuracy"""
    st.info(
        """
        **Model Performance Info:**
        - **Logistic Regression**: ~88% accuracy on test data
        - **LSTM Model**: Typically achieves higher accuracy on complex reviews
        - Results may vary based on review length and complexity
        """
    )

def main():
    # Create sidebar
    create_sidebar()
    
    # Load models with progress bar
    with st.spinner('Loading AI models... This may take a moment'):
        lr_model = load_logistic_regression_model()
        tfidf_vectorizer = load_tfidf_vectorizer()
        lstm_model = load_lstm_model()
        tokenizer = load_tokenizer()
    
    # Check if all models loaded successfully
    if not all([lr_model, tfidf_vectorizer, lstm_model, tokenizer]):
        st.error("Failed to load one or more models. Please check that the model files exist in the 'models' directory.")
        st.stop()
    
    # Show success message if models loaded
    st.success("✅ AI models loaded successfully!")
    
    # Create main interface
    user_input = create_main_interface()
    
    # Create analyze button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
    
    # Process the input when button is clicked
    if analyze_button:
        if user_input.strip():
            show_loading_animation()
            
            # Perform analysis
            lr_pred, lr_conf, lr_prob, lstm_pred, lstm_conf, lstm_prob = analyze_sentiment(
                user_input, lr_model, tfidf_vectorizer, lstm_model, tokenizer
            )
            
            if lr_pred and lstm_pred:
                # Display results
                display_results(lr_pred, lr_conf, lstm_pred, lstm_conf)
                
                # Show comparison chart
                st.markdown("---")
                create_comparison_chart(lr_conf, lstm_conf)
                
                # Show model info
                show_model_accuracy_info()
                
                # Show original text
                st.subheader("📝 Your Review")
                st.info(user_input)
            else:
                st.error("Could not analyze the sentiment. Please try a different review.")
        else:
            st.warning("Please enter a movie review to analyze.")

    # Add footer
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #808080;'>MovieSent AI Sentiment Analyzer | Powered by Streamlit</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
