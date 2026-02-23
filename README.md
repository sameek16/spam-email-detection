# Spam Email Detection using TensorFlow

A Deep Learning based Spam Email Classifier built using TensorFlow and LSTM.
The model classifies incoming emails into two categories:
- Spam
- Not Spam

## Tech Stack
- Python
- TensorFlow
- LSTM (Deep Learning)
- Natural Language Processing (Tokenization & Padding)
- Scikit-learn
- Pandas
- NumPy
- Streamlit (for web app)

## Model Performance
The model achieves approximately 97–99% accuracy on the test dataset.

## Dataset
SMS Spam Collection Dataset (UCI Machine Learning Repository)

## Project Structure
spam-email-detection-tensorflow/
│
├── notebook/
│   └── spam_email_detection_tensorflow.ipynb
│
├── app/
│   └── app.py
│
├── requirements.txt
├── README.md
└── .gitignore

## Features
- Text preprocessing and cleaning
- Tokenization and sequence padding
- LSTM-based deep learning model
- Real-time spam prediction via Streamlit web interface

## How to Run

1. Clone the repository
2. Install dependencies:
   pip install -r requirements.txt
3. Run the notebook or launch the web app:
   streamlit run app.py

## Author
Sameek Bhoir
