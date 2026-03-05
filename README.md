# 🏦 Bank Customer Churn Prediction Web App

This is an **end-to-end web application** that predicts whether a bank customer is likely to churn (leave the bank) based on their profile and transaction history.
It combines a **Flask backend** (machine learning model) and a **React frontend** (user interface).

---

## 🚀 Features

* Upload a `.csv` file with customer data.
* Backend runs a **trained Voting/ExtraTreesClassifier model** to predict churn.
* Results are displayed in a dynamic React table.
* Download predictions as a `.csv` with an extra **Prediction** column.
* Example input/output files included for testing.


* Frontend → app.py → preprocessing → encoder/scaler → model → prediction → back to frontend
---

## 📦 Tech Stack

* **Backend:** Python, Flask, scikit-learn, imbalanced-learn
* **Frontend:** React, TailwindCSS
* **ML Model:** ExtraTreesClassifier (with SMOTE & preprocessing pipeline)

---

## 🔧 Prerequisites

* **Python** 3.8+
* **Node.js** 16+
* **npm** 8+

---

## ⚙️ Setup Instructions

### 1️⃣ Backend (Flask + ML Model)

```bash


# Create and activate virtual environment
python -m venv venv

# On Windows
.venv\Scripts\activate

cd backend

# Install dependencies
pip install -r requirements.txt

# Start backend server
python app.py
```

Backend will run on 👉 `http://localhost:5000`

---

### 2️⃣ Frontend (React UI)

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start 
```

Frontend will run on 👉 `http://localhost:3000`

---

## 🖥️ Usage

1. Open `http://localhost:3000` in your browser.
2. Upload a `.csv` file with customer data (see **example\_input.csv** for format).
3. View predictions in the table.
4. Optionally, download the results as a `.csv` (with `Prediction` column).

---

## 📂 Project Structure

```
BANK-CHURN-PREDICTION/
│
├── backend/
│   ├── app.py               # Flask API server
│   ├── preprocess.py        # Preprocessing logic
│   ├── save_model.py        # Model training script
│   ├── model.pkl            # Trained ML model
│   ├── example_input.csv    # Sample input
│   ├── example_output.csv   # Sample predictions
│   ├── output_predictions.csv
│   └── requirements.txt     # Python dependencies
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── FileUpload.jsx
│   │   │   └── ResultsTable.jsx
│   │   ├── App.jsx
│   │   ├── index.jsx
│   │   ├── output.css
│   │   └── styles.css
│   ├── public/index.html
│   ├── package.json
│   └── tailwind.config.js
│
├── input/credit-card-customers/BankChurners.csv
├── README.md
└── .gitignore
```

---

## 📊 Example Files

* `example_input.csv` → Example input format.
* `example_output.csv` → Example predictions file.
* `output_predictions.csv` → Sample generated results.

---

## 📝 Notes

* The **backend expects CSV files** with columns matching `example_input.csv`.
* The **model is pre-trained** and saved as `model.pkl`.
* Frontend supports **CSV upload & download** for easy workflow.


