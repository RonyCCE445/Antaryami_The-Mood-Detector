
# Antaryami_The Mood Detector

A simple and interactive web app built with [Streamlit](https://streamlit.io/) and [Hugging Face Transformers](https://huggingface.co/transformers/) that detects emotions in your text input. The app classifies moods like joy, sadness, anger, fear, surprise, and neutral with confidence scores and visualizations.

---

## Demo

[Live Demo on Streamlit Cloud](https://your-streamlit-app-url)

---

## Features

- Detects top 3 emotions from input text.
- Displays confidence scores and intuitive icons.
- Visual bar chart of emotion probabilities.
- Provides additional probable emotions with a caution about probabilities.

---

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/antaryami-mood-detector.git
   cd antaryami-mood-detector
   ```

2. (Optional) Create and activate a virtual environment:

   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Run the app locally:

```bash
streamlit run app.py
```

Enter your text, click **Detect Mood**, and see the analysis.

---

## Project Structure

```
.
├── app.py             # Main Streamlit app
├── requirements.txt   # Python dependencies
├── LICENSE            # License file
└── README.md          # Project documentation
```

---

## Technologies Used

- [Streamlit](https://streamlit.io/)
- [Transformers](https://huggingface.co/transformers/)
- [Pandas](https://pandas.pydata.org/)

---

## Notes

- Uses the Hugging Face model `j-hartmann/emotion-english-distilroberta-base`.
- Emotion predictions are probabilistic; actual feelings might vary.
- Icons rendered via HTML + Font Awesome.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Author

Koustav Chakraborty — ckoustav7@gmail.com — RonyCCE445

---

## Acknowledgements

- Hugging Face for the amazing pre-trained models.
- Streamlit for making app deployment easy and interactive.

---

Feel free to open issues or submit pull requests to improve this project!
