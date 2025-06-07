import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.graph_objects as go

MAX_CHARS = 300

@st.cache_resource(show_spinner=False)
def load_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )

emotion_bg_colors = {
    "joy": "#2ecc71",
    "love": "#e83e8c",
    "anger": "#e74c3c",
    "fear": "#9b59b6",
    "sadness": "#3498db",
    "disgust": "#95a5a6",
    "surprise": "#f39c12",
    "neutral": "#7f8c8d"
}

mood_quotes = {
    "joy": "Keep shining bright today! 😊",
    "love": "Love is the strongest emotion of all. ❤️",
    "anger": "Take a deep breath, anger is temporary. 🔥",
    "fear": "Face your fears and grow stronger. 🧅",
    "sadness": "It’s okay to feel blue sometimes. 🌧️",
    "disgust": "Remember, every feeling is valid. 🤢",
    "surprise": "Life is full of unexpected moments! ⚡",
    "neutral": "A calm mind is a powerful mind. 🧘‍♂️"
}

st.markdown("""
<style>
html, body {
  font-family: 'Courier New', monospace;
  color: #fff;
  background-color: #0a0a0a;
}
h1 {
  text-align: center;
  color: #40e0d0;
  text-shadow: 0 0 5px #40e0d0, 0 0 10px #40e0d0, 0 0 20px #40e0d0, 0 0 40px #40e0d0;
}
.emotion-title {
  text-align: center;
  font-size: 2rem;
  font-weight: bold;
}
.conf-bar {
  background: #333;
  border-radius: 8px;
  padding: 5px;
  max-width: 400px;
  margin: 0 auto 0.5rem auto;
}
.conf-fill {
  height: 20px;
  background: #0f0;
  border-radius: 8px;
  box-shadow: 0 0 10px #0f0;
  transition: width 0.5s ease-in-out;
}
div.stButton > button {
  background-color: #111;
  color: #40e0d0;
  border: 1px solid #40e0d0;
  border-radius: 8px;
  padding: 0.6em 1.2em;
  box-shadow: 0 0 10px #40e0d0;
  transition: 0.3s;
  font-weight: bold;
  font-family: monospace;
  margin: 1rem 0;
}
div.stButton > button:hover {
  box-shadow: 0 0 20px #40e0d0, 0 0 40px #40e0d0;
  cursor: pointer;
}
@keyframes pulse-glow {
  0% { text-shadow: 0 0 5px currentColor; transform: scale(1); }
  50% { text-shadow: 0 0 20px currentColor; transform: scale(1.1); }
  100% { text-shadow: 0 0 5px currentColor; transform: scale(1); }
}
.animated-icon {
  animation: pulse-glow 2.5s infinite ease-in-out;
  display: inline-block;
}
</style>
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css" rel="stylesheet">
""", unsafe_allow_html=True)

st.markdown("<h1>Antaryami: The Mood Detector 🎮</h1>", unsafe_allow_html=True)

emotion_classifier = load_model()

icon_map = {
    'joy': '<i class="fas fa-smile animated-icon" style="color:#2ecc71;"></i>',
    'sadness': '<i class="fas fa-frown animated-icon" style="color:#3498db;"></i>',
    'anger': '<i class="fas fa-angry animated-icon" style="color:#e74c3c;"></i>',
    'fear': '<i class="fas fa-exclamation-triangle animated-icon" style="color:#9b59b6;"></i>',
    'love': '<i class="fas fa-heart animated-icon" style="color:#e83e8c;"></i>',
    'surprise': '<i class="fas fa-bolt animated-icon" style="color:#f39c12;"></i>',
    'neutral': '<i class="fas fa-meh animated-icon" style="color:#7f8c8d;"></i>',
    'disgust': '<i class="fas fa-smile-slash animated-icon" style="color:#95a5a6;"></i>'
}

user_input = st.text_area(f"Enter your text to analyse mood (max {MAX_CHARS} chars):", height=150, max_chars=MAX_CHARS)
progress_val = min(len(user_input) / MAX_CHARS, 1.0)
st.progress(progress_val)
st.write(f"Characters typed: {len(user_input)} / {MAX_CHARS}")

if "show_detail" not in st.session_state:
    st.session_state.show_detail = False
if "scores" not in st.session_state:
    st.session_state.scores = None
if "top_emotion" not in st.session_state:
    st.session_state.top_emotion = None
if "confidence" not in st.session_state:
    st.session_state.confidence = 0

if st.button("Detect Mood"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            results = emotion_classifier(user_input)

        if results:
            scores = sorted(results[0], key=lambda x: x['score'], reverse=True)
            st.session_state.scores = scores
            st.session_state.top_emotion = scores[0]['label'].lower()
            st.session_state.confidence = scores[0]['score']
        else:
            st.error("Could not analyze the mood. Try different input.")

if st.session_state.scores and st.session_state.top_emotion:
    scores = st.session_state.scores
    top_emotion = st.session_state.top_emotion
    confidence = st.session_state.confidence

    bg_color = emotion_bg_colors.get(top_emotion, "#0a0a0a")
    st.markdown(f"<script>document.body.style.backgroundColor = '{bg_color}';</script>", unsafe_allow_html=True)

    st.markdown(f"<div class='emotion-title'>{icon_map.get(top_emotion, '')} Detected Mood: <b>{top_emotion.capitalize()}</b></div>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center;'>{mood_quotes.get(top_emotion, '')}</p>", unsafe_allow_html=True)

    st.markdown(f"""
        <div class="conf-bar">
            <div class="conf-fill" style="width:{confidence*100:.1f}%"></div>
        </div>
        <small style="display:block; text-align:center;">Confidence: {confidence:.2%}</small>
    """, unsafe_allow_html=True)

    scores_dict = {item['label'].lower(): item['score'] for item in scores}
    joy, love = scores_dict.get("joy", 0), scores_dict.get("love", 0)
    anger, fear = scores_dict.get("anger", 0), scores_dict.get("fear", 0)
    sadness, disgust = scores_dict.get("sadness", 0), scores_dict.get("disgust", 0)
    surprise, neutral = scores_dict.get("surprise", 0), scores_dict.get("neutral", 0)

    composite_emotions = {
        "Positive Affection": joy + love,
        "Frustration Level": max(anger - fear, 0),
        "Sad & Unpleasant": sadness + disgust,
        "Startled Feelings": surprise + fear,
        "Anxiety Indicator": fear + (surprise * 0.3),
        "Mixed Feelings": max((fear + surprise) - joy, 0)
    }

    st.subheader("What more could you probably be feeling?")
    for name, score in sorted(composite_emotions.items(), key=lambda x: x[1], reverse=True)[:3]:
        st.write(f"- **{name}**: {score:.2%}")
    st.markdown("<small>⚠️ These values are calculated from model outputs and may not reflect your full emotional state.</small>", unsafe_allow_html=True)
       
       
        # Mood Summary (Human-friendly)
    
    
    top_composites = sorted(composite_emotions.items(), key=lambda x: x[1], reverse=True)[:2]
    mood_summary = f"""
    <div style="background:#111;padding:1rem;border-radius:10px;margin-top:1.5rem;border:1px solid #40e0d0;">
        <h3 style="color:#40e0d0;text-align:center;">🧠 Mood Summary</h3>
        <p style="font-family:monospace;line-height:1.6;color:white;text-align:justify;">
        Based on your input, you are primarily feeling <b style="color:{emotion_bg_colors.get(top_emotion)}">{top_emotion.capitalize()}</b>.
        There are also strong signs of <b>{top_composites[0][0]}</b> ({top_composites[0][1]:.0%}) 
        and <b>{top_composites[1][0]}</b> ({top_composites[1][1]:.0%}).
        This mix suggests a complex emotional state, which is totally normal! 🌈
        </p>
    </div>
    """
    st.markdown(mood_summary, unsafe_allow_html=True)
if st.session_state.scores:
    scores = st.session_state.scores

    with st.expander("📊 Show Detailed Mood Analysis", expanded=False):
        df = pd.DataFrame(scores)
        df['score'] = df['score'].astype(float)
        st.subheader("Top Emotional Scores")
        st.bar_chart(df.set_index('label')['score'])

        st.subheader("Emotion Radar Chart")
        categories = [e['label'].capitalize() for e in scores] + [scores[0]['label'].capitalize()]
        values = [e['score'] for e in scores] + [scores[0]['score']]
        fig = go.Figure(
            data=[go.Scatterpolar(r=values, theta=categories, fill='toself')],
            layout=go.Layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False, height=400)
        )
        st.plotly_chart(fig, use_container_width=True)
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; font-style: italic; color: #40e0d0; margin-top: 2rem;">
        <strong>Note from the Developer:</strong><br>
        Hey Awesome folks! I'm Koustav Chakraborty, a Btech Undergrad from India, I'm the silly developer of this program.This mood detection tool is powered by AI models that analyze your text's emotional tones. 
        While it strives to understand your feelings, remember that emotions are complex and nuanced — 
        no AI can fully capture the depth of human experience. Use this app as a fun guide, not a diagnosis. Please provide your feedback via gmail: koustavchak24@gmail.com <br><br>
         If you liked it, Let's connect via: <br><br>
        <a href="https://github.com/RonyCCE445" target="_blank" style="color:#40e0d0; margin:0 10px; text-decoration:none;">
            <i class="fab fa-github fa-lg"></i> GitHub
        </a>
        <a href="https://www.linkedin.com/in/koustav-chakraborty-642b02247/" target="_blank" style="color:#40e0d0; margin:0 10px; text-decoration:none;">
            <i class="fab fa-linkedin fa-lg"></i> LinkedIn
        </a>
         <a href="https://www.instagram.com/a_fakir_in_disguise/" target="_blank" style="color:#40e0d0; margin:0 10px; text-decoration:none;">
            <i class="fab fa-instagram fa-lg"></i> Instagram
        </a>
        <br><br>
        Stay curious, stay kind! 😊

        
    </div>
    """,
    unsafe_allow_html=True,
)
