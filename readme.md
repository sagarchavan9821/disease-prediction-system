# 🏥 Disease Prediction System

An end-to-end Machine Learning pipeline that predicts **10 diseases** based on patient health data — built with Python, FastAPI, Streamlit, PostgreSQL, and Docker.

---

## 🎯 Diseases Predicted

| Disease | Disease |
|---|---|
| ❤️ Heart Disease | 🩸 Diabetes |
| 🧠 Stroke | 🫘 Kidney Disease |
| 🎗️ Cancer | 🧬 Alzheimer's Disease |
| 🫁 COPD | 🫀 Liver Disease |
| 🧠 Parkinson's Disease | 🦠 Tuberculosis |

---

## 🏗️ Project Architecture

```
Patient Health Data (PostgreSQL)
          ↓
  preprocessing.py (encoding + feature engineering)
          ↓
    train.py (model experimentation)
          ↓
    main.py (final model training)
          ↓
      model.pkl (saved SVM model)
          ↓
   FastAPI (REST API /predict)
          ↓
  Streamlit (interactive UI)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| ML Model | SVM (Support Vector Machine) |
| Tuning | GridSearchCV |
| API | FastAPI |
| Frontend | Streamlit |
| Database | PostgreSQL |
| Serialization | Pickle |
| Containerization | Docker + Docker Compose |

---

## 📁 Project Structure

```
disease-prediction-system/
├── preprocessing.py       ← data cleaning & encoding
├── train.py               ← model experimentation
├── main.py                ← final model training → model.pkl
├── database.py            ← PostgreSQL connection
├── app_ui.py              ← Streamlit frontend
├── requirements.txt       ← dependencies
├── Dockerfile.api         ← FastAPI container
├── Dockerfile.ui          ← Streamlit container
├── docker-compose.yml     ← runs everything
└── api/
    ├── app.py             ← FastAPI prediction endpoint
    └── model.pkl          ← trained model
```

---

## 🚀 Run with Docker (Recommended)

### Prerequisites
- Docker Desktop installed

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/sagarchavan9821/disease-prediction-system.git
cd disease-prediction-system

# 2. Run everything with one command
docker-compose up --build
```

### Access the app
```
Streamlit UI  →  http://localhost:8501
FastAPI docs  →  http://localhost:8000/docs
```

---

## 💻 Run Locally (Without Docker)

### Prerequisites
```bash
pip install -r requirements.txt
```

### Steps

```bash
# 1. Train and save model
python main.py

# 2. Start FastAPI (Terminal 1)
cd api
uvicorn app:app --reload

# 3. Start Streamlit (Terminal 2)
streamlit run app_ui.py
```

---

## 📊 ML Pipeline Details

### Models Experimented:
| Model | Accuracy |
|---|---|
| Logistic Regression | ~68% |
| Decision Tree | ~65% |
| Random Forest | ~71% |
| Gradient Boosting | ~70% |
| **SVM (Selected)** | **~71-87%** |

### Why SVM?
- Best balance of accuracy across all 10 diseases
- Works well with scaled numerical data
- Effective with limited features
- Tuned with GridSearchCV (C, kernel, gamma)

### Feature Engineering:
- User provides **height & weight**
- System automatically calculates **BMI category**
- No manual BMI input required!

---

## 🔌 API Usage

### Endpoint: `POST /predict`

**Request:**
```json
{
  "age": 65,
  "gender": "Male",
  "height": 170,
  "weight": 85,
  "blood_pressure": "High",
  "cholesterol": "High",
  "glucose": "High",
  "smoking": "Yes",
  "alcohol_consumption": "Yes",
  "exercise": "No",
  "family_history": "Yes"
}
```

**Response:**
```json
{
  "predictions": {
    "heart_disease": 1,
    "diabetes": 1,
    "stroke": 1,
    "kidney_disease": 0,
    "cancer": 0,
    "alzheimers_disease": 0,
    "copd": 1,
    "liver_disease": 1,
    "parkinsons_disease": 0,
    "tuberculosis": 0
  }
}
```

---

## 📚 What I Learned

- Building complete ML pipelines from scratch
- Deploying ML models as production REST APIs
- Connecting ML backend to interactive frontend
- PostgreSQL integration for scalable data storage
- Containerizing multi-service apps with Docker
- Hyperparameter tuning with GridSearchCV
- Feature engineering and data preprocessing

---

## 👨‍💻 Author

**Sagar Chavan**  
[GitHub](https://github.com/sagarchavan9821) • [LinkedIn](www.linkedin.com/in/sagar-chavan-b2build)

---

## ⚠️ Disclaimer

This project is for **educational purposes only**.  
Not intended for real medical diagnosis.  
Always consult a qualified doctor for medical advice.