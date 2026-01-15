# MovieSent – Dual-Approach Sentiment Analysis (FYP)

## Quick Start
1. `pip install -r requirements.txt`
2. Place `5000_Movie_Reviews.csv` inside `data/`
3. Run `python app.py` → open http://127.0.0.1:5000

## Streamlit App
To run the modern Streamlit interface:
1. `pip install -r streamlit_requirements.txt`
2. Run `streamlit run streamlit_app.py`
3. Open the provided URL in your browser

## Repo Highlights
- Classical ML: Logistic Regression + TF-IDF (n-grams)
- Deep Learning: Bi-LSTM (+ GloVe embeddings)
- Web demos: Flask app and modern Streamlit interface
- Compare predictions between both models
- View confidence percentages for each model