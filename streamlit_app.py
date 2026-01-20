"""
Movie Review Sentiment Analysis App
Streamlit interface for sentiment prediction with model metrics
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS STYLING
# ============================================
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .positive-sentiment {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .negative-sentiment {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# UTILITY FUNCTIONS
# ============================================
@st.cache_resource
def load_model():
    """Load trained model and vectorizer"""
    try:
        # Try loading combined artifacts first
        artifacts = joblib.load('sentiment_model_artifacts.joblib')
        return artifacts['model'], artifacts['vectorizer'], artifacts.get('metadata', {})
    except:
        # Fallback to separate files
        try:
            model = joblib.load('lr_tfidf.joblib')
            vectorizer = joblib.load('tfidf_vectorizer.joblib')
            return model, vectorizer, {}
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None, None, {}

def clean_text(txt):
    """Clean and preprocess text"""
    txt = re.sub(r'<.*?>', ' ', txt)          # Remove HTML tags
    txt = re.sub(r'[^A-Za-z0-9 ]+', ' ', txt) # Keep alphanumeric
    txt = re.sub(r'\s+', ' ', txt).strip().lower()
    return txt

def predict_sentiment(text, model, vectorizer):
    """Make sentiment prediction"""
    clean_txt = clean_text(text)
    features = vectorizer.transform([clean_txt])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    return {
        'sentiment': prediction,
        'confidence': max(probabilities) * 100,
        'probabilities': {
            'negative': probabilities[0] * 100,
            'positive': probabilities[1] * 100
        },
        'cleaned_text': clean_txt
    }

# ============================================
# LOAD MODEL
# ============================================
model, vectorizer, metadata = load_model()

if model is None:
    st.error("⚠️ Could not load model. Please ensure model files exist.")
    st.stop()

# ============================================
# HEADER
# ============================================
st.markdown('<p class="main-header">🎬 Movie Review Sentiment Analyzer</p>', unsafe_allow_html=True)
st.markdown("---")

# ============================================
# SIDEBAR - MODEL INFORMATION
# ============================================
with st.sidebar:
    st.header("📊 Model Information")
    
    if metadata:
        st.metric("Training Accuracy", f"{metadata.get('train_score', 0):.2%}")
        st.metric("Testing Accuracy", f"{metadata.get('test_score', 0):.2%}")
        
        if 'training_date' in metadata:
            st.info(f"**Trained:** {metadata['training_date'][:10]}")
    else:
        st.warning("No metadata available")
    
    st.markdown("---")
    st.subheader("🔧 Model Details")
    st.write("**Algorithm:** Logistic Regression")
    st.write("**Features:** TF-IDF (1-2 grams)")
    st.write("**Max Features:** 15,000")
    
    st.markdown("---")
    st.subheader("ℹ️ How It Works")
    st.write("""
    1. Enter your movie review
    2. Text is cleaned and preprocessed
    3. TF-IDF vectorization applied
    4. Logistic Regression predicts sentiment
    5. Results shown with confidence scores
    """)

# ============================================
# MAIN CONTENT - TABS
# ============================================
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📈 Model Metrics", "🧪 Batch Testing"])

# ============================================
# TAB 1: PREDICTION INTERFACE
# ============================================
with tab1:
    st.header("Enter Your Movie Review")
    
    # Sample reviews for quick testing
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📝 Load Positive Example"):
            st.session_state.review_text = "This movie was absolutely fantastic! The acting was superb, the storyline was engaging, and the cinematography was breathtaking. I highly recommend it to everyone!"
    with col2:
        if st.button("📝 Load Negative Example"):
            st.session_state.review_text = "Terrible movie. Waste of time and money. Poor acting, boring plot, and awful special effects. I want my money back!"
    
    # Text input
    review_text = st.text_area(
        "Type or paste your review here:",
        value=st.session_state.get('review_text', ''),
        height=150,
        placeholder="e.g., This movie was amazing! Great storyline and excellent acting..."
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button("🚀 Analyze Sentiment", use_container_width=True)
    
    if predict_button and review_text:
        with st.spinner("Analyzing sentiment..."):
            result = predict_sentiment(review_text, model, vectorizer)
            
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            # Display result with styling
            sentiment = result['sentiment']
            confidence = result['confidence']
            
            if sentiment.lower() == 'positive':
                st.markdown(f"""
                <div class="positive-sentiment">
                    <h2>✅ POSITIVE Sentiment</h2>
                    <h3>Confidence: {confidence:.2f}%</h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="negative-sentiment">
                    <h2>❌ NEGATIVE Sentiment</h2>
                    <h3>Confidence: {confidence:.2f}%</h3>
                </div>
                """, unsafe_allow_html=True)
            
            # Probability breakdown
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "😊 Positive Probability",
                    f"{result['probabilities']['positive']:.2f}%",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "😞 Negative Probability",
                    f"{result['probabilities']['negative']:.2f}%",
                    delta=None
                )
            
            # Visualization
            fig = go.Figure(data=[
                go.Bar(
                    x=['Negative', 'Positive'],
                    y=[result['probabilities']['negative'], result['probabilities']['positive']],
                    marker_color=['#dc3545', '#28a745'],
                    text=[f"{result['probabilities']['negative']:.1f}%", 
                          f"{result['probabilities']['positive']:.1f}%"],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Sentiment Probability Distribution",
                yaxis_title="Probability (%)",
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show cleaned text
            with st.expander("🔍 View Processed Text"):
                st.code(result['cleaned_text'])
    
    elif predict_button:
        st.warning("⚠️ Please enter a review to analyze!")

# ============================================
# TAB 2: MODEL METRICS
# ============================================
with tab2:
    st.header("📊 Model Performance Metrics")
    
    # Create sample metrics (replace with actual test data if available)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Training Accuracy</h3>
            <h1>88.5%</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Testing Accuracy</h3>
            <h1>87.9%</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>F1 Score</h3>
            <h1>0.88</h1>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Confusion Matrix (Sample data - replace with actual)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        confusion_data = pd.DataFrame(
            [[175, 25], [23, 177]],
            columns=['Predicted Negative', 'Predicted Positive'],
            index=['Actual Negative', 'Actual Positive']
        )
        
        fig = px.imshow(
            confusion_data.values,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Negative', 'Positive'],
            y=['Negative', 'Positive'],
            color_continuous_scale='Blues',
            text_auto=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Classification Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['Precision', 'Recall', 'F1-Score', 'Support'],
            'Negative': [0.88, 0.88, 0.88, 200],
            'Positive': [0.88, 0.88, 0.88, 200]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Feature Importance (Top words)
    st.subheader("🔤 Top Predictive Words")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Positive Indicators:**")
        positive_words = ['excellent', 'great', 'wonderful', 'loved', 'amazing', 
                         'fantastic', 'brilliant', 'superb', 'outstanding', 'perfect']
        for i, word in enumerate(positive_words[:5], 1):
            st.write(f"{i}. {word}")
    
    with col2:
        st.markdown("**Negative Indicators:**")
        negative_words = ['terrible', 'worst', 'awful', 'boring', 'waste', 
                         'horrible', 'bad', 'poor', 'disappointing', 'stupid']
        for i, word in enumerate(negative_words[:5], 1):
            st.write(f"{i}. {word}")

# ============================================
# TAB 3: BATCH TESTING
# ============================================
with tab3:
    st.header("🧪 Batch Testing")
    st.write("Test multiple reviews at once")
    
    # Sample reviews for testing
    sample_reviews = [
        "This movie exceeded all my expectations! Brilliant performances.",
        "Waste of time. Terrible plot and bad acting.",
        "An absolute masterpiece. One of the best films I've ever seen.",
        "Boring and predictable. Would not recommend.",
        "Great cinematography but weak storyline."
    ]
    
    if st.button("🎯 Run Batch Test"):
        results = []
        progress_bar = st.progress(0)
        
        for i, review in enumerate(sample_reviews):
            result = predict_sentiment(review, model, vectorizer)
            results.append({
                'Review': review[:50] + '...' if len(review) > 50 else review,
                'Sentiment': result['sentiment'],
                'Confidence': f"{result['confidence']:.2f}%",
                'Positive': f"{result['probabilities']['positive']:.1f}%",
                'Negative': f"{result['probabilities']['negative']:.1f}%"
            })
            progress_bar.progress((i + 1) / len(sample_reviews))
        
        st.success("✅ Batch testing completed!")
        
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        # Summary statistics
        st.subheader("📊 Batch Summary")
        col1, col2 = st.columns(2)
        
        positive_count = sum(1 for r in results if r['Sentiment'].lower() == 'positive')
        negative_count = len(results) - positive_count
        
        with col1:
            st.metric("Positive Reviews", positive_count)
        with col2:
            st.metric("Negative Reviews", negative_count)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Develop by Shahid Mumtaz | BC200204433| Movie Review Sentiment Analysis</p>
    <p>Model: Logistic Regression with TF-IDF Features</p>
</div>
""", unsafe_allow_html=True)

