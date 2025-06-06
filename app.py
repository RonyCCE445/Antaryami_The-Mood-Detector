import streamlit as st
from transformers import pipeline
import pandas as pd

# Load the Hugging Face Emotion Classification Pipeline
@st.cache_resource(show_spinner=False)
def load_model():
    # Use "text-classification" task with a pre-trained emotion detection model
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )
st.markdown("""
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css" rel="stylesheet">
""", unsafe_allow_html=True)


emotion_classifier = load_model()

# Mapping moods to Font Awesome icons with colors (for display in markdown)
icon_map = {
    'joy': '<i class="fas fa-smile" style="color:#28a745;"></i>',
    'sadness': '<i class="fas fa-frown" style="color:#007bff;"></i>',
    'anger': '<i class="fas fa-angry" style="color:#dc3545;"></i>',
    'fear': '<i class="fas fa-surprise" style="color:#6f42c1;"></i>',
    'love': '<i class="fas fa-heart" style="color:#e83e8c;"></i>',
    'surprise': '<i class="fas fa-surprise" style="color:#fd7e14;"></i>',
    'neutral': '<i class="fas fa-meh" style="color:#6c757d;"></i>'
}

st.title("Antaryami_The Mood Detector")

# Text input area for user to enter text for mood analysis
user_input = st.text_area("Enter your text to analyse mood:", height=150)

# When the Detect Mood button is clicked
if st.button("Detect Mood"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            # Get emotion scores from the model (list of dicts)
            results = emotion_classifier(user_input)

            if results and isinstance(results, list) and len(results) > 0:
                scores = results[0]  # Extract scores list from results
                # Sort emotions by score descending
                scores = sorted(scores, key=lambda x: x['score'], reverse=True)

                # Top detected emotion and confidence
                top_emotion = scores[0]['label']
                confidence = scores[0]['score']
                icon_html = icon_map.get(top_emotion.lower(), '')

                # Display detected mood with icon and confidence
                st.markdown(
                    f"<h2>Detected Mood: {icon_html} <b>{top_emotion.capitalize()}</b></h2>",
                    unsafe_allow_html=True
                )
                st.write(f"Confidence: {confidence:.2%}")

                # Convert scores to a DataFrame for bar chart visualization
                df = pd.DataFrame(scores)
                df['score'] = df['score'].astype(float)
                st.bar_chart(df.set_index('label')['score'])

                # Create a dictionary for quick lookup of emotion scores
                scores_dict = {item['label'].lower(): item['score'] for item in scores}

                # Extract individual emotions for calculations
                joy = scores_dict.get("joy", 0)
                love = scores_dict.get("love", 0)
                anger = scores_dict.get("anger", 0)
                fear = scores_dict.get("fear", 0)
                sadness = scores_dict.get("sadness", 0)
                disgust = scores_dict.get("disgust", 0)
                surprise = scores_dict.get("surprise", 0)
                neutral = scores_dict.get("neutral", 0)

                # Calculate composite emotions with friendly names
                positive_affection = joy + love
                frustration_level = max(anger - fear, 0)
                sad_unpleasant = sadness + disgust
                startled_feelings = surprise + fear
                anxiety_indicator = fear + (surprise * 0.3)
                mixed_feelings = max((fear + surprise) - joy, 0)
               

                # Prepare a dict of composite emotions with friendly names
                composite_emotions = {
                    "Positive Affection": positive_affection,
                    "Frustration Level ": frustration_level,
                    "Sad / Unpleasant Feelings ": sad_unpleasant,
                    "Startled Feelings": startled_feelings,
                    "Anxiety Indicator ": anxiety_indicator,
                    "Mixed Feelings ": mixed_feelings,
                    
                }

                # Sort emotions by score descending and take top 3
                top_3_emotions = sorted(composite_emotions.items(), key=lambda x: x[1], reverse=True)[:3]

                # Display top 3 composite emotions
                st.markdown("### What more could you probably be feeling?")
                for name, score in top_3_emotions:
                    st.write(f"- {name}: **{score:.2%}**")

                # Display caution about probabilistic nature of these values
                st.markdown(
                    "<small>⚠️ These values are calculated using probabilities from the model and may not exactly match your true feelings.</small>",
                    unsafe_allow_html=True
                )

            else:
                st.error("Could not analyze the mood. Please try again with different text.")
