import streamlit as st
from transformers import FlaubertTokenizer, FlaubertForSequenceClassification
import torch
import re
from youtube_transcript_api import YouTubeTranscriptApi
import fitz  # PyMuPDF
import io
import base64

# CSS pour l'alignement du contenu
st.markdown("""
    <style>
    .block-container > .element-container {
        display: flex;
        align-items: start;
    }
    </style>
    """, unsafe_allow_html=True)

def set_background(image_path):
    with open(image_path, "rb") as file:
        base64_image = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{base64_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Example usage
set_background('/Users/nabilnarif/Desktop/ar.jpeg')  # Replace with your local image path

def set_title_color():
    st.markdown(
        """
        <style>
        h1, h2, h3, h4, h5, h6 {
            color: black;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_title_color()

col1, col2 = st.columns([1, 1])
with col1:
    st.image("/Users/nabilnarif/Desktop/Logo_Université_de_Lausanne.svg.png", width=350)
with col2:
    st.markdown("# MICROSOFT GROUP")

# Chargement du modèle et du tokenizer
model_path = '/Users/nabilnarif/Desktop/model'
model = FlaubertForSequenceClassification.from_pretrained(model_path)
tokenizer = FlaubertTokenizer.from_pretrained('flaubert/flaubert_large_cased')
index_to_difficulty = {0: "A1", 1: "A2", 2: "B1", 3: "B2", 4: "C1", 5: "C2"}

# Fonction pour extraire l'ID de la vidéo à partir de l'URL YouTube
def extract_video_id(youtube_url):
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11})"
    matches = re.search(regex, youtube_url)
    if matches:
        return matches.group(1)
    return None

# Fonction pour prédire la difficulté du texte
def predict_difficulty(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=212)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction_idx = torch.argmax(logits, dim=-1).item()
    return index_to_difficulty[prediction_idx]

# Fonction pour extraire les sous-titres d'une vidéo YouTube
def extract_subtitles(youtube_url, max_words=100):
    video_id = extract_video_id(youtube_url)
    if not video_id:
        return "The URL of the YouTube video is invalid or the video ID could not be extracted."

    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript(['fr', 'fr-FR'])
        transcript = transcript.fetch()

        words = []
        for item in transcript:
            words.extend(item['text'].split())
            if len(words) >= max_words:
                break

        return ' '.join(words[:max_words])
    except Exception as e:
        return f"Error or subtitles not available : {e}"

# Interface Streamlit
# Interface pour la prédiction de texte
st.title("Predict your text !")
text = st.text_area("Enter your text here", "")
if st.button("Predict"):
    if text:
        predicted_difficulty = predict_difficulty(text)
        st.write(f"CEFR level of your text : {predicted_difficulty}")
    else:
        st.write("Please enter a valid text.")

st.title("Predict CEFR level of a Youtube video ")
youtube_url = st.text_input("Enter your Youtube URL", "")

if st.button("Predict subtitles CEFR level"):
    if youtube_url:
        sous_titres = extract_subtitles(youtube_url)
        if sous_titres.startswith("Your video doesn't have subtitles"):
            st.write(sous_titres)
        else:
            predicted_difficulty = predict_difficulty(sous_titres)
            st.write("Extracted subtitles :", sous_titres)
            st.write(f"Predicted CEFR level : {predicted_difficulty}")
    else:
        st.write("Please enter a valid Youtube URL.")


# Fonction pour lire le contenu d'un fichier PDF
def extract_text_from_pdf(pdf_file):
    try:
        # Convertir le fichier Streamlit UploadedFile en un objet bytes
        pdf_bytes = pdf_file.getvalue()
        pdf_stream = io.BytesIO(pdf_bytes)

        # Lecture du fichier PDF à partir du stream
        pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")
        text = ""
        for page in pdf_document:
            text += page.get_text()
        pdf_document.close()
        return text
    except Exception as e:
        return f"Error reading your PDF file : {e}"

# Interface Streamlit pour charger un fichier PDF
st.title("Predict CEFR level of your PDF")
pdf_file = st.file_uploader("Upload your PDF", type=["pdf"])

if st.button("Predict CEFR level of the PDF"):
    if pdf_file:
        text = extract_text_from_pdf(pdf_file)
        if text.startswith("Error"):
            st.write(text)
        else:
            predicted_difficulty = predict_difficulty(text)
            st.write(f"Predicted CEFR level : {predicted_difficulty}")
    else:
        st.write("Please upload a valid PDF file.")
