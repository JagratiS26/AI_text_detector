# AI Text Detector 🧠

An AI-powered web application that detects whether a given text is **Human-written** or **AI-generated** using Machine Learning.

---

## 🚀 Project Overview

This project uses a trained Machine Learning model to analyze textual patterns and predict the likelihood of AI-generated content.  
It includes:

- A **modern frontend UI** (HTML, CSS, JavaScript)
- A **Flask backend API**
- A **trained ML model** for prediction
- Real-time interaction between frontend and backend

---

## 🛠️ Tech Stack

### Frontend
- HTML
- CSS
- JavaScript (Fetch API)

### Backend
- Python
- Flask
- Flask-CORS

### Machine Learning
- Scikit-learn
- Feature Engineering
- Trained classification model

---

## 📁 Project Structure

AI_TEXT_DETECTOR/
│
├── app.py            # Flask backend entry point
│
├── public/           # Frontend files
│ ├── index.html
│ ├── style.css
│ └── script.js
│
├── models/            # Trained ML models
│ └── trained_model.pkl
│
├── figures/           # SHAP explanation plots
│ └── prediction_explanations/
│
├── data/              # Dataset files
│
├── reports/           # Analysis & evaluation reports
│
├── src/               # ML & analysis scripts
│ ├── feature_engine2.py
│ ├── predict2.py
│ ├── model_training2.py
│ ├── data_processing2.py
│ ├── evaluation2.py
│ └── other analysis scripts
│
├── requirements.txt   # Project dependencies
└── README.md

---

## How It Works

1. User enters text in the frontend UI
2. Frontend sends the text to Flask backend (`/predict` API)
3. Backend processes the text using ML model
4. Model predicts:
   - **Label** (AI / Human)
   - **AI Probability**
   - **Confidence level**
5. Result is displayed on the UI

---

## ▶️ How to Run the Project Locally

1. Clone the repository:
2. install required dependencies: pip install -r requirements.txt (in terminal)
3. Run the backend server: python app.py
4. Backend will start at: http://127.0.0.1:5000
5. Open the frontend: Open public/index.html using Live Server or any browser.


## Project Status
✔ Frontend integrated with backend  
✔ ML model integrated  
✔ Local deployment completed  
🔄 Further improvements planned

---

## Features
- Detects whether text is Human-written or AI-generated
- Displays AI probability and confidence score
- Dark / Light mode UI
- SHAP-based explainability support
- Interactive and responsive UI


---



