# 🏦 Bank Customer Churn Prediction — AI-Powered Web App

An **end-to-end machine learning web application** that predicts whether a bank customer is likely to churn (leave the bank) based on their profile and transaction history — with **Explainable AI (XAI)** insights powered by SHAP and Groq LLM.

It combines a **Flask backend** (ML model + SHAP explainability) and a **React + TypeScript frontend** (modern UI with interactive visualizations).

---

## 🚀 Features

### Core Prediction
- Upload a `.csv` file with customer data (drag-and-drop supported, up to 10MB)
- Predict churn using a trained **ExtraTreesClassifier** (93.39% accuracy)
- View churn summary: total customers, at-risk count, churn rate
- Download predictions as a `.csv` with an added **Prediction** column
- Prediction history stored locally for quick access

### Explainable AI (XAI) Dashboard
- **Model Performance Metrics** — Accuracy, AUC-ROC, F1-Score
- **SHAP Feature Importance** — Top 10 features ranked by mean |SHAP| value (computed per upload)
- **SHAP Distribution Chart** — Scatter plot showing how feature values impact churn prediction
- **Individual Customer Explanation** — Select any customer to see their top 5 risk factors with SHAP values
- **AI-Generated Interpretations** — Natural language explanations powered by Groq (Llama 3.3 70B), with template fallback

### UI/UX
- Dark/Light mode toggle
- Responsive design with TailwindCSS
- Real-time processing status with animated progress
- Toast notifications for user feedback

---

## 📦 Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Backend** | Python, Flask, scikit-learn, imbalanced-learn, SHAP, Groq API |
| **Frontend** | React 18, TypeScript, Vite, TailwindCSS, Shadcn/ui, Recharts |
| **ML Model** | ExtraTreesClassifier (305 trees, SMOTE-balanced, OneHotEncoder + StandardScaler pipeline) |
| **AI Explanations** | Groq API (Llama 3.3 70B) — free tier |

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 93.39% |
| AUC-ROC | 0.9679 |
| F1-Score | 0.7721 |
| Estimators | 305 |

---

## 🔧 Prerequisites

- **Python** 3.8+
- **Node.js** 16+
- **npm** 8+

---

## ⚙️ Setup Instructions

### 1️⃣ Backend (Flask + ML Model)

```bash
# From project root
cd backend

# Create and activate virtual environment
python -m venv venv

# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Set Groq API key for AI explanations
# Get a free key at: https://console.groq.com/keys
# Create a .env file in backend/ with:
# GROQ_API_KEY=your_key_here

# Start backend server
python app.py
```

Backend will run on 👉 `http://localhost:5000`

---

### 2️⃣ Frontend (React + TypeScript)

```bash
cd frontendd

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend will run on 👉 `http://localhost:8080`

---

## 🖥️ Usage

1. Open `http://localhost:8080` in your browser.
2. Upload a `.csv` file with customer data (see `example_input.csv` for format).
3. View the churn prediction summary (total, at-risk, churn rate).
4. Download the predictions as a CSV file.
5. Navigate to **Model Insights** to explore:
   - Model performance metrics
   - SHAP-based feature importance with AI interpretation
   - Feature impact distribution (scatter plot)
   - Individual customer explanations with risk classification

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Upload CSV, returns predictions + session_id |
| `GET` | `/download` | Download predictions as CSV |
| `GET` | `/metrics` | Model performance metrics |
| `GET` | `/feature-importance?session_id=` | Top 10 features by SHAP importance |
| `GET` | `/shap-distribution?session_id=` | SHAP scatter plot data |
| `GET` | `/sample-customers?session_id=` | List of customers for individual explanation |
| `POST` | `/customer-explanation` | Individual SHAP + AI explanation |
| `GET` | `/global-interpretation?session_id=` | AI-generated global insights |
| `GET` | `/shap-status?session_id=` | Poll background SHAP computation status |

---

## 📂 Project Structure

```
BANK-CHURN-PREDICTION/
│
├── backend/
│   ├── app.py                 # Flask REST API server
│   ├── preprocess.py          # Data preprocessing (missing values, type conversion)
│   ├── shap_utils.py          # SHAP computation, AI explanations, feature analysis
│   ├── model.pkl              # Trained ML model (ExtraTreesClassifier)
│   ├── model_metrics.json     # Model performance metrics
│   ├── feature_names.json     # List of model features
│   ├── requirements.txt       # Python dependencies
│   ├── .env                   # Groq API key (not tracked in git)
│   ├── example_input.csv      # Sample input file
│   └── example_output.csv     # Sample predictions output
│
├── frontendd/
│   ├── src/
│   │   ├── App.tsx            # Router & app layout
│   │   ├── pages/
│   │   │   ├── Index.tsx      # Main prediction page
│   │   │   └── ModelInsights.tsx  # XAI dashboard
│   │   ├── components/
│   │   │   ├── FileUpload.tsx              # Drag-and-drop CSV upload
│   │   │   ├── HeroSection.tsx             # Landing hero section
│   │   │   ├── ModelMetricsCards.tsx        # Performance metric cards
│   │   │   ├── FeatureImportanceChart.tsx   # SHAP bar chart + AI interpretation
│   │   │   ├── ShapDistributionChart.tsx    # SHAP scatter plot
│   │   │   ├── CustomerExplanationPanel.tsx # Individual customer analysis
│   │   │   ├── PredictionHistory.tsx        # Past predictions table
│   │   │   ├── ProcessingStatus.tsx         # Upload progress indicator
│   │   │   ├── ResultsDownload.tsx          # Results summary + download
│   │   │   ├── ThemeToggle.tsx              # Dark/light mode toggle
│   │   │   ├── NavLink.tsx                  # Navigation link wrapper
│   │   │   └── ui/                          # Shadcn/ui component library
│   │   ├── lib/
│   │   │   ├── apiService.ts  # Backend API client
│   │   │   ├── csvUtils.ts    # CSV parsing & download utilities
│   │   │   └── utils.ts       # Tailwind class merge utility
│   │   └── hooks/
│   │       ├── usePredictionHistory.ts  # Prediction history (localStorage)
│   │       └── use-toast.ts             # Toast notification hook
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.ts
│   └── tsconfig.json
│
├── input/
│   └── credit-card-customers/
│       └── BankChurners.csv   # Raw dataset (10,127 bank customers)
│
├── save_model.py              # Model training script (SMOTE + ExtraTrees pipeline)
├── README.md
└── .gitignore
```

---

## 🔄 Data Flow

```
CSV Upload → Frontend (validation) → POST /predict → Backend (preprocess → model → predict)
                                                         ↓
                                              Background SHAP computation
                                                         ↓
Model Insights Page ← polling /shap-status ← SHAP values ready
       ↓
Feature Importance, Distribution Charts, Customer Explanations (SHAP + Groq AI)
```

---

## 🌐 Environment Variables

| Variable | Location | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | `backend/.env` | Groq API key for AI-generated explanations (optional — falls back to templates) |
| `VITE_API_URL` | `frontendd/.env` | Backend API URL (default: `http://localhost:5000`) |

---

## 📊 Example Files

- `example_input.csv` → Sample customer data for upload
- `example_output.csv` → Sample predictions output with Prediction column

---

## 📝 Notes

- The backend expects CSV files with columns matching `example_input.csv`
- The model is pre-trained and saved as `model.pkl` — retrain using `save_model.py`
- SHAP values are computed asynchronously in baseline threads for performance
- Risk classification thresholds: **High** (≥70%), **Medium** (40–70%), **Low** (<40%)
- Prediction history is stored in browser localStorage (first 50 rows per prediction)
- The Groq API key is optional — AI explanations fall back to template-based responses if unavailable

