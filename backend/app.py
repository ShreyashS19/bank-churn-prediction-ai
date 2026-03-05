from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import joblib
import os
import numpy as np
import uuid
import threading

from preprocess import preprocess_data
from shap_utils import (
    get_model_metrics,
    compute_feature_importance_dynamic,
    get_shap_distribution_dynamic,
    get_customer_explanation,
    get_global_ai_interpretation_dynamic,
    compute_shap_for_data
)

app = Flask(__name__)
CORS(app)  

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

model = joblib.load(MODEL_PATH)

# In-memory session store for per-upload SHAP data
# Key: session_id (str), Value: dict with shap_values, feature_names, X_data, etc.
sessions = {}


def _compute_shap_background(session_id):
    """Compute SHAP values in a background thread."""
    session = sessions.get(session_id)
    if not session:
        return
    try:
        df_model = pd.DataFrame(session['df_records'], columns=session['X_columns'])
        shap_result = compute_shap_for_data(df_model, model)
        session['shap_values'] = shap_result['shap_values']
        session['feature_names'] = shap_result['feature_names']
        # Store the subsampled X data so distribution indices match SHAP rows
        session['X_data'] = shap_result['X_sampled']
        session['X_columns'] = shap_result['X_sampled_columns']
        session['shap_computed'] = True
        print(f"[SHAP] Background computation finished for session {session_id}")
    except Exception as e:
        session['shap_error'] = str(e)
        print(f"[SHAP] Background computation FAILED for session {session_id}: {e}")
    finally:
        session['shap_ready_event'].set()  # unblock any waiting request


def _ensure_shap_ready(session, timeout=120):
    """Block until the background SHAP thread finishes (or timeout)."""
    event = session.get('shap_ready_event')
    if event:
        event.wait(timeout=timeout)
    if session.get('shap_error'):
        raise RuntimeError(session['shap_error'])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.get_json()
        df = pd.DataFrame(data)
        
        df = preprocess_data(df) 
        # Store original column order before any modifications
        original_columns = df.columns.tolist()
        
        # ✅ Column mapping: Rename CSV columns to match model's expected columns
        column_mapping = {
            'Customer_Age': 'Age',
            'Education_Level': 'Education',
            'Income_Category': 'Income',
            'Months_Inactive_12_mon': 'Months_Inactive',
            'Contacts_Count_12_mon': 'Contacts_Count'
        }
        df = df.rename(columns=column_mapping)
        
        # Required columns (same as training)
        required_columns = ['Age', 'Gender', 'Dependent_count', 'Education', 'Marital_Status', 'Income', 
                           'Card_Category', 'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive',
                           'Contacts_Count', 'Credit_Limit', 'Total_Revolving_Bal', 'Total_Amt_Chng_Q4_Q1', 
                           'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']
        
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            return jsonify({'error': f'Missing required columns: {missing}'}), 400
        
        # Extract model input data (only required columns) for SHAP computation
        df_model = df[required_columns].copy()
        
        # Directly predict (pipeline handles preprocessing)
        predictions = model.predict(df_model)
        probabilities = model.predict_proba(df_model)[:, 1]
        predictions_labels = ['Churn' if pred == 1 else 'Not Churn' for pred in predictions]
        
        # Store session & start background SHAP computation immediately
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            'shap_computed': False,
            'shap_error': None,
            'shap_ready_event': threading.Event(),
            'shap_values': None,
            'feature_names': None,
            'X_data': df_model.values,
            'X_columns': required_columns,
            'predictions': predictions_labels,
            'probabilities': probabilities.tolist(),
            'df_records': df_model.to_dict(orient='records')
        }
        # Fire-and-forget background thread
        threading.Thread(
            target=_compute_shap_background,
            args=(session_id,),
            daemon=True
        ).start()
        print(f"[SHAP] Background thread started for session {session_id}")
        
        # Add predictions column to full dataframe
        df['Prediction'] = predictions_labels
        
        # Preserve original column order + Prediction at the end
        updated_original_columns = []
        for col in original_columns:
            if col in column_mapping:
                updated_original_columns.append(column_mapping[col])
            else:
                updated_original_columns.append(col)
        
        # Reorder columns: original order + Prediction
        output_columns = updated_original_columns + ['Prediction']
        df = df[output_columns]
        
        # Save output CSV with preserved column order
        output_path = 'output_predictions.csv'
        df.to_csv(output_path, index=False)
        
        # Response with preserved column order + session_id
        response = {
            'predictions': df.to_dict(orient='records'),
            'csv_path': output_path,
            'session_id': session_id
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download', methods=['GET'])
def download():
    try:
        return send_file('output_predictions.csv', as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'Flask backend is running'}), 200


@app.route('/shap-status', methods=['GET'])
def shap_status():
    """Return whether SHAP computation is done for a session."""
    session_id = request.args.get('session_id')
    if not session_id or session_id not in sessions:
        return jsonify({'ready': False, 'error': 'Invalid session'}), 400
    session = sessions[session_id]
    return jsonify({
        'ready': session.get('shap_computed', False),
        'error': session.get('shap_error')
    })


# =====================================================
# EXPLAINABLE AI ENDPOINTS
# =====================================================

@app.route('/metrics', methods=['GET'])
def metrics():
    """Return model performance metrics."""
    try:
        data = get_model_metrics()
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({'error': 'Model metrics not found. Please retrain the model using save_model.py'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/feature-importance', methods=['GET'])
def feature_importance():
    """Return global feature importance based on mean |SHAP| for the uploaded data."""
    try:
        session_id = request.args.get('session_id')
        if not session_id or session_id not in sessions:
            return jsonify({'error': 'No active session. Please upload a CSV file first.'}), 400

        session = sessions[session_id]
        _ensure_shap_ready(session)
        data = compute_feature_importance_dynamic(
            session['shap_values'],
            session['feature_names']
        )
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/shap-distribution', methods=['GET'])
def shap_distribution():
    """Return SHAP distribution data for the uploaded data."""
    try:
        session_id = request.args.get('session_id')
        if not session_id or session_id not in sessions:
            return jsonify({'error': 'No active session. Please upload a CSV file first.'}), 400

        session = sessions[session_id]
        _ensure_shap_ready(session)
        data = get_shap_distribution_dynamic(
            session['shap_values'],
            session['feature_names'],
            session['X_data'],
            session['X_columns']
        )
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/customer-explanation', methods=['POST'])
def customer_explanation():
    """
    Compute individual customer SHAP explanation.
    Accepts JSON with a single customer record.
    """
    try:
        data = request.get_json()

        if isinstance(data, list):
            if len(data) == 0:
                return jsonify({'error': 'No customer data provided'}), 400
            customer_record = data[0]
        elif isinstance(data, dict):
            customer_record = data
        else:
            return jsonify({'error': 'Invalid data format'}), 400

        df = pd.DataFrame([customer_record])

        # Apply same preprocessing as prediction
        df = preprocess_data(df)

        # Column mapping
        column_mapping = {
            'Customer_Age': 'Age',
            'Education_Level': 'Education',
            'Income_Category': 'Income',
            'Months_Inactive_12_mon': 'Months_Inactive',
            'Contacts_Count_12_mon': 'Contacts_Count'
        }
        df = df.rename(columns=column_mapping)

        required_columns = [
            'Age', 'Gender', 'Dependent_count', 'Education', 'Marital_Status', 'Income',
            'Card_Category', 'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive',
            'Contacts_Count', 'Credit_Limit', 'Total_Revolving_Bal', 'Total_Amt_Chng_Q4_Q1',
            'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio'
        ]

        # Keep only required columns
        available = [col for col in required_columns if col in df.columns]
        if len(available) < len(required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            return jsonify({'error': f'Missing required columns: {missing}'}), 400

        df = df[required_columns]

        result = get_customer_explanation(df, model)
        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/global-interpretation', methods=['GET'])
def global_interpretation():
    """Return AI-generated interpretation based on the uploaded data."""
    try:
        session_id = request.args.get('session_id')
        if not session_id or session_id not in sessions:
            return jsonify({'error': 'No active session. Please upload a CSV file first.'}), 400

        session = sessions[session_id]
        _ensure_shap_ready(session)
        interpretation = get_global_ai_interpretation_dynamic(
            session['shap_values'],
            session['feature_names']
        )
        return jsonify({'interpretation': interpretation})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/sample-customers', methods=['GET'])
def sample_customers():
    """
    Return customers from the uploaded CSV for the
    individual customer explanation selector.
    """
    try:
        session_id = request.args.get('session_id')
        if not session_id or session_id not in sessions:
            return jsonify({'error': 'No active session. Please upload a CSV file first.'}), 400

        session = sessions[session_id]
        df_records = session['df_records']
        probabilities = session['probabilities']
        predictions = session['predictions']

        customers = []
        for i in range(min(50, len(df_records))):
            prob = float(probabilities[i])
            if prob >= 0.7:
                risk = "High Risk"
            elif prob >= 0.4:
                risk = "Medium Risk"
            else:
                risk = "Low Risk"

            customers.append({
                'id': f'CUST-{i+1:03d}',
                'index': i,
                'churn_probability': round(prob, 4),
                'risk_level': risk,
                'prediction': predictions[i],
                'data': df_records[i]
            })

        # Sort by churn probability descending
        customers.sort(key=lambda x: x['churn_probability'], reverse=True)

        return jsonify(customers)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
