import gradio as gr
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    filtered = [ps.stem(word) for word in tokens if word not in stop_words]
    return " ".join(filtered)

# Prediction function
def predict_news(news):
    cleaned = clean_text(news)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return "🔴 Fake News" if prediction == 0 else "🟢 Real News"

# Gradio UI
interface = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(lines=7, placeholder="Enter news text here..."),
    outputs=gr.Textbox(label="Prediction"),
    title="📰 Fake News Detector",
    description="Enter a news article and detect if it's real or fake using NLP and ML."
)

if __name__ == "__main__":
    interface.launch()
