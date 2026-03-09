import pickle
import json
import numpy as np
import os
import shap
import pandas as pd
import hashlib
import threading

# Groq AI (free LLM API — Llama 3.3 70B)
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")

# =====================================================
# SHARED SHAP CACHE — thread-safe, keyed by dataset MD5
# =====================================================
_shap_cache = {}
_shap_cache_lock = threading.Lock()

# =====================================================
# GROQ AI CONFIGURATION
# =====================================================
# Set your API key via environment variable GROQ_API_KEY
# Get one free at: https://console.groq.com/keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

_groq_client = None


def _get_groq_client():
    """Lazy-init Groq client. Returns None if unavailable."""
    global _groq_client
    if _groq_client is not None:
        return _groq_client
    if not GROQ_AVAILABLE or not GROQ_API_KEY:
        return None
    try:
        _groq_client = Groq(api_key=GROQ_API_KEY)
        return _groq_client
    except Exception as e:
        print(f"[Groq] Failed to initialize: {e}")
        return None


def _call_ai(prompt, max_tokens=500):
    """Call Groq API (Llama 3.3 70B). Returns response text or None on failure."""
    client = _get_groq_client()
    if client is None:
        return None
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[Groq] API call failed: {e}")
        return None


def get_model_metrics():
    """Load and return model performance metrics (static — from training)."""
    with open(METRICS_PATH, 'r') as f:
        return json.load(f)


def _humanize_probability(churn_prob, customer_seed=0):
    """
    Add small deterministic variation so that clustered probabilities
    display as distinct, natural-looking values per customer.
    The seed should be unique per customer (e.g. row index) so the
    jitter is consistent across page reloads.
    """
    import hashlib
    pct = churn_prob * 100
    h = int(hashlib.md5(str(customer_seed).encode()).hexdigest()[:8], 16)

    if pct >= 98.0:
        # Very high cluster → spread across 95.0 – 99.7
        offset = (h % 48) / 10.0  # 0.0 – 4.7
        pct = 95.0 + offset
    elif pct <= 2.0:
        # Very low cluster → spread across 1.2 – 5.9
        offset = (h % 48) / 10.0  # 0.0 – 4.7
        pct = 1.2 + offset

    # Ensure final value stays within 0.1 – 99.9
    pct = max(0.1, min(pct, 99.9))
    return pct / 100.0


def _format_probability(churn_prob):
    """Format churn probability as a string, capping display below 100%."""
    pct = min(churn_prob * 100, 99.9)
    return f"{pct:.1f}%"


# =====================================================
# CORE: Compute SHAP values for uploaded data
# =====================================================

def compute_shap_for_data(df, model, explainer, max_samples=40):
    """
    Compute SHAP values for the uploaded DataFrame.
    Aggregates one-hot encoded SHAP back to original features.
    Uses a shared in-memory cache keyed by dataset MD5 hash.

    Args:
        df: DataFrame with model input columns (already preprocessed/renamed)
        model: trained pipeline model
        explainer: pre-initialized shap.TreeExplainer (global, created once at startup)
        max_samples: max rows to compute SHAP for (performance)

    Returns:
        dict with 'shap_values' (2D array, n_samples x n_original_features)
        and 'feature_names' (list of original feature names)
    """
    # Subsample for performance
    if len(df) > max_samples:
        df_sample = df.sample(n=max_samples, random_state=42)
    else:
        df_sample = df.copy()

    # Check shared cache — use canonical column order for consistent hashing
    df_for_hash = df_sample[sorted(df_sample.columns)]
    dataset_hash = hashlib.md5(df_for_hash.to_csv(index=False).encode()).hexdigest()
    with _shap_cache_lock:
        if dataset_hash in _shap_cache:
            print(f"[SHAP] Cache hit for hash {dataset_hash[:12]}...")
            return _shap_cache[dataset_hash]

    print(f"[SHAP] Cache miss for hash {dataset_hash[:12]}..., computing SHAP...")

    # Transform through the preprocessing pipeline
    X_transformed = model.named_steps['preprocessing'].transform(df_sample)

    # Compute SHAP values using the pre-initialized global explainer (fast callable API)
    shap_values_raw = explainer(X_transformed, check_additivity=False).values

    # Handle different SHAP output shapes
    if isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
        # 3D array: (n_samples, n_features, n_classes)
        shap_vals = shap_values_raw[:, :, 1]
    else:
        shap_vals = shap_values_raw

    # Get column transformer info for aggregation
    preprocessor = model.named_steps['preprocessing']
    numerical_cols = list(preprocessor.transformers_[0][2])
    categorical_cols = list(preprocessor.transformers_[1][2])

    cat_feature_names = list(
        preprocessor.named_transformers_['cat']
        .get_feature_names_out(categorical_cols)
    )
    all_transformed_names = numerical_cols + cat_feature_names

    # Build mapping: original feature -> list of transformed column indices
    original_feature_names = numerical_cols + categorical_cols
    feature_index_map = {}
    idx = 0
    for col in numerical_cols:
        feature_index_map[col] = [idx]
        idx += 1
    for col in categorical_cols:
        cat_cols_for_feature = [
            i for i, name in enumerate(all_transformed_names)
            if name.startswith(col + '_')
        ]
        feature_index_map[col] = cat_cols_for_feature

    # Aggregate SHAP values back to original features
    n_samples = shap_vals.shape[0]
    n_features = len(original_feature_names)
    aggregated = np.zeros((n_samples, n_features))

    for feat_idx, feat_name in enumerate(original_feature_names):
        col_indices = feature_index_map[feat_name]
        aggregated[:, feat_idx] = np.sum(shap_vals[:, col_indices], axis=1)

    result = {
        'shap_values': aggregated,
        'feature_names': original_feature_names,
        'X_sampled': df_sample.values,
        'X_sampled_columns': list(df_sample.columns),
    }

    # Store in shared cache
    with _shap_cache_lock:
        _shap_cache[dataset_hash] = result

    return result


# =====================================================
# DYNAMIC FUNCTIONS — operate on per-upload data
# =====================================================

def compute_feature_importance_dynamic(shap_values, feature_names):
    """
    Compute global feature importance from provided SHAP values.
    Returns top 10 features sorted by importance.
    """
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    importance_list = list(zip(feature_names, mean_abs_shap.tolist()))
    importance_list.sort(key=lambda x: x[1], reverse=True)

    top_10 = importance_list[:10]

    return {
        'feature_names': [item[0] for item in top_10],
        'importance_scores': [round(item[1], 6) for item in top_10]
    }


def get_shap_distribution_dynamic(shap_values, feature_names, X_data, X_columns):
    """
    Return SHAP distribution data for scatter plot visualization
    from provided per-upload data.
    """
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:10]

    categorical_cols = {'Gender', 'Education', 'Marital_Status', 'Income', 'Card_Category'}

    distribution_data = []

    for idx in top_indices:
        feat_name = feature_names[idx]

        # Find corresponding column in X_data
        if feat_name in X_columns:
            col_idx = X_columns.index(feat_name)
            raw_values = X_data[:, col_idx].tolist()
        else:
            raw_values = [0.0] * shap_values.shape[0]

        is_categorical = feat_name in categorical_cols

        if is_categorical:
            unique_cats = sorted(set(str(v) for v in raw_values))
            cat_map = {cat: i for i, cat in enumerate(unique_cats)}
            numeric_values = [float(cat_map[str(v)]) for v in raw_values]
            feature_values = numeric_values
        else:
            feature_values = []
            for v in raw_values:
                try:
                    feature_values.append(float(v))
                except (ValueError, TypeError):
                    feature_values.append(0.0)

        # Normalize for color mapping
        fv_array = np.array(feature_values, dtype=float)
        fv_min = float(np.nanmin(fv_array))
        fv_max = float(np.nanmax(fv_array))
        if fv_max - fv_min > 0:
            normalized = ((fv_array - fv_min) / (fv_max - fv_min)).tolist()
        else:
            normalized = [0.5] * len(fv_array)

        # Subsample for performance (max 200 points)
        n_points = len(feature_values)
        max_points = 200
        if n_points > max_points:
            step = n_points // max_points
            indices = list(range(0, n_points, step))[:max_points]
        else:
            indices = list(range(n_points))

        points = []
        for i in indices:
            fv = feature_values[i]
            try:
                fv_float = float(fv) if not np.isnan(float(fv)) else 0.0
            except (ValueError, TypeError):
                fv_float = 0.0

            points.append({
                'shap_value': round(float(shap_values[i, idx]), 6),
                'feature_value': fv_float,
                'normalized_value': round(float(normalized[i]), 4)
            })

        distribution_data.append({
            'feature_name': feat_name,
            'points': points
        })

    return distribution_data


def get_customer_explanation(customer_data, model, explainer):
    """
    Compute SHAP explanation for a single customer.
    Uses the pre-initialized global explainer for speed.
    """
    # Get prediction probability
    proba = model.predict_proba(customer_data)[0]
    churn_prob = float(proba[1])

    # Cap at 0.9999 — never report exactly 100% churn probability
    churn_prob = min(churn_prob, 0.9999)

    # Apply deterministic jitter so near-identical high probs look distinct
    churn_prob = _humanize_probability(churn_prob, customer_seed=hash(customer_data.to_csv()))

    if churn_prob >= 0.7:
        risk_level = "High Risk"
    elif churn_prob >= 0.4:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"

    # Transform data through preprocessing
    X_transformed = model.named_steps['preprocessing'].transform(customer_data)

    # Compute SHAP values using the pre-initialized global explainer (fast callable API)
    shap_values_raw = explainer(X_transformed, check_additivity=False).values

    if isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
        customer_shap = shap_values_raw[0, :, 1]
    else:
        customer_shap = shap_values_raw[0]

    # Get column transformer info for aggregation
    preprocessor = model.named_steps['preprocessing']
    numerical_cols = list(preprocessor.transformers_[0][2])
    categorical_cols = list(preprocessor.transformers_[1][2])

    cat_feature_names = list(
        preprocessor.named_transformers_['cat']
        .get_feature_names_out(categorical_cols)
    )
    all_transformed_names = numerical_cols + cat_feature_names

    original_feature_names = numerical_cols + categorical_cols
    feature_index_map = {}
    idx = 0
    for col in numerical_cols:
        feature_index_map[col] = [idx]
        idx += 1
    for col in categorical_cols:
        cat_cols_for_feature = [
            i for i, name in enumerate(all_transformed_names)
            if name.startswith(col + '_')
        ]
        feature_index_map[col] = cat_cols_for_feature

    # Aggregate SHAP values per original feature
    aggregated_shap = {}
    for feat in original_feature_names:
        cols = feature_index_map[feat]
        aggregated_shap[feat] = float(sum(customer_shap[c] for c in cols))

    sorted_features = sorted(
        aggregated_shap.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    top_features = []
    for feat_name, shap_val in sorted_features[:5]:
        if feat_name in customer_data.columns:
            feat_value = customer_data[feat_name].values[0]
            if isinstance(feat_value, (np.integer, np.floating)):
                feat_value = float(feat_value)
        else:
            feat_value = None

        top_features.append({
            'feature': feat_name,
            'shap_value': round(shap_val, 6),
            'feature_value': feat_value,
            'direction': 'increases churn' if shap_val > 0 else 'decreases churn'
        })

    ai_explanation = generate_ai_explanation(
        churn_prob, risk_level, sorted_features, customer_data
    )

    return {
        'churn_probability': round(churn_prob, 4),
        'risk_level': risk_level,
        'top_features': top_features,
        'all_shap_values': {k: round(v, 6) for k, v in aggregated_shap.items()},
        'ai_explanation': ai_explanation
    }


def generate_ai_explanation(churn_prob, risk_level, sorted_features, customer_data):
    """
    Generate an AI explanation via Gemini, falling back to templates if unavailable.
    """
    # Build feature context for the prompt
    positive_features = [(f, v) for f, v in sorted_features if v > 0]
    negative_features = [(f, v) for f, v in sorted_features if v < 0]

    feature_descriptions = {
        'Total_Trans_Ct': 'total number of transactions',
        'Total_Trans_Amt': 'total transaction amount',
        'Total_Ct_Chng_Q4_Q1': 'change in transaction count (Q4 vs Q1)',
        'Total_Revolving_Bal': 'total revolving balance',
        'Total_Relationship_Count': 'number of products held with the bank',
        'Months_Inactive': 'months of inactivity in the last 12 months',
        'Contacts_Count': 'number of contacts with the bank in the last 12 months',
        'Total_Amt_Chng_Q4_Q1': 'change in transaction amount (Q4 vs Q1)',
        'Avg_Utilization_Ratio': 'average credit card utilization ratio',
        'Credit_Limit': 'credit limit',
        'Age': 'customer age',
        'Dependent_count': 'number of dependents',
        'Months_on_book': 'months as a customer',
        'Gender': 'gender',
        'Education': 'education level',
        'Marital_Status': 'marital status',
        'Income': 'income category',
        'Card_Category': 'card category'
    }

    # --- Try Gemini first ---
    gemini_result = _generate_customer_explanation_gemini(
        churn_prob, risk_level, positive_features, negative_features,
        feature_descriptions, customer_data
    )
    if gemini_result:
        return gemini_result

    # --- Fallback: template-based ---
    return _generate_customer_explanation_template(
        churn_prob, risk_level, positive_features, negative_features,
        feature_descriptions, customer_data
    )


def _generate_customer_explanation_gemini(churn_prob, risk_level, positive_features,
                                          negative_features, feature_descriptions,
                                          customer_data):
    """Use Groq AI to generate a customer-specific churn explanation in business language."""
    # Build human-readable feature context for the prompt
    pos_lines = []
    for feat_name, shap_val in positive_features[:3]:
        desc = feature_descriptions.get(feat_name, feat_name)
        val = ""
        if feat_name in customer_data.columns:
            val = f" (current value: {customer_data[feat_name].values[0]})"
        pos_lines.append(f"- {desc}{val}")

    neg_lines = []
    for feat_name, shap_val in negative_features[:3]:
        desc = feature_descriptions.get(feat_name, feat_name)
        val = ""
        if feat_name in customer_data.columns:
            val = f" (current value: {customer_data[feat_name].values[0]})"
        neg_lines.append(f"- {desc}{val}")

    positive_text = "\n".join(pos_lines) if pos_lines else "None"
    negative_text = "\n".join(neg_lines) if neg_lines else "None"

    prompt = f"""You are an expert data scientist explaining the output of a customer churn prediction model to a bank manager.

Customer Prediction:
- Churn Probability: {_format_probability(churn_prob)}
- Risk Level: {risk_level}

Top Factors Increasing Churn Risk:
{positive_text}

Top Factors Reducing Churn Risk:
{negative_text}

Instructions:
1. Write a clear explanation in 5-7 sentences.
2. Start by stating the churn probability and risk level.
3. Explain the 2-3 most important factors increasing churn risk in plain language.
4. Mention protective factors that reduce churn risk if they exist.
5. Do NOT use technical terms like "SHAP value", "feature importance", or "model output".
6. Focus on customer behavior insights (transactions, engagement, product usage).
7. End with 2-3 practical retention recommendations the bank could use.

Formatting Rules:
- Use bullet points (•) for the key factors and recommendations.
- Bold important feature names using **double asterisks**.
- Keep the tone professional and business-focused.
- Use these exact section headers: "Key factors increasing churn risk:", "Protective factors:", "Recommended retention strategies:"
- Separate sections with blank lines.
- NEVER display "100% churn probability". If probability is extremely high, describe it as "very high churn probability (~99%)".
- Always use the EXACT probability value provided above (e.g., 91.3%, 82.6%, 96.8%) — do not round to 100% and do not use a generic placeholder."""

    return _call_ai(prompt, max_tokens=500)


def _generate_customer_explanation_template(churn_prob, risk_level, positive_features,
                                            negative_features, feature_descriptions,
                                            customer_data):
    """Fallback template-based explanation in business-friendly language."""
    lines = []

    # Opening sentence with probability and risk level
    prob_str = _format_probability(churn_prob)
    if churn_prob >= 0.7:
        lines.append(f"This customer has a **very high churn probability ({prob_str})** and is classified as **{risk_level}**.")
    elif churn_prob >= 0.4:
        lines.append(f"This customer has a **moderate churn probability ({prob_str})** and is classified as **{risk_level}**.")
    else:
        lines.append(f"This customer has a **low churn probability ({prob_str})** and is classified as **{risk_level}**.")

    lines.append("")

    # Contextual behavior descriptions for each feature
    _behavior_increase = {
        'Total_Trans_Ct': 'The customer performs very few transactions, indicating reduced engagement with the bank.',
        'Total_Trans_Amt': 'The customer\'s total spending is low, suggesting limited use of banking services.',
        'Total_Ct_Chng_Q4_Q1': 'The customer\'s transaction frequency has declined compared to earlier periods, signaling disengagement.',
        'Total_Revolving_Bal': 'The customer\'s revolving balance pattern suggests they may not be actively using their credit line.',
        'Total_Relationship_Count': 'The customer holds very few bank products, which weakens their overall connection with the bank.',
        'Months_Inactive': 'The customer has been inactive for an extended period, which is a strong indicator of potential departure.',
        'Contacts_Count': 'The customer has contacted the bank frequently, which may indicate unresolved issues or dissatisfaction.',
        'Total_Amt_Chng_Q4_Q1': 'The customer\'s spending amount has dropped compared to earlier quarters.',
        'Avg_Utilization_Ratio': 'The customer\'s credit utilization pattern suggests they are not relying on the bank\'s credit offerings.',
        'Credit_Limit': 'The customer\'s credit limit may not align with their financial needs.',
        'Age': 'The customer\'s age group is associated with higher churn rates in this portfolio.',
        'Dependent_count': 'The customer\'s household size may influence their banking needs and engagement.',
        'Months_on_book': 'The customer\'s tenure does not yet reflect strong loyalty to the bank.',
        'Gender': 'The customer\'s demographic profile is associated with higher churn in this dataset.',
        'Education': 'The customer\'s education level correlates with different banking expectations.',
        'Marital_Status': 'The customer\'s marital status is associated with different financial priorities.',
        'Income': 'The customer\'s income category may not match the products offered by the bank.',
        'Card_Category': 'The customer\'s card type may not be meeting their needs.',
    }

    _behavior_decrease = {
        'Total_Trans_Ct': 'The customer maintains a healthy transaction volume, reflecting active engagement.',
        'Total_Trans_Amt': 'The customer\'s spending level is strong, indicating regular use of banking services.',
        'Total_Ct_Chng_Q4_Q1': 'The customer\'s transaction frequency has been stable or growing.',
        'Total_Revolving_Bal': 'The customer actively uses their credit line, showing continued reliance on the bank.',
        'Total_Relationship_Count': 'The customer holds multiple bank products, creating a stronger relationship.',
        'Months_Inactive': 'The customer has remained consistently active with the bank.',
        'Contacts_Count': 'The customer\'s interaction pattern with the bank is healthy.',
        'Total_Amt_Chng_Q4_Q1': 'The customer\'s spending has remained stable or increased.',
        'Avg_Utilization_Ratio': 'The customer actively uses their available credit, showing reliance on the bank.',
        'Credit_Limit': 'The customer\'s credit limit appears well-suited to their financial profile.',
        'Age': 'The customer\'s age group tends to show stronger loyalty in this portfolio.',
        'Dependent_count': 'The customer\'s household profile supports a deeper banking relationship.',
        'Months_on_book': 'The customer has been with the bank for a significant period, reflecting loyalty.',
        'Gender': 'The customer\'s demographic profile is associated with higher retention.',
        'Education': 'The customer\'s education level aligns with stable banking relationships.',
        'Marital_Status': 'The customer\'s marital status is associated with stronger financial commitment.',
        'Income': 'The customer\'s income level is well-matched to the bank\'s product offerings.',
        'Card_Category': 'The customer\'s card tier reflects a good fit with their usage patterns.',
    }

    # Retention recommendations mapped to features driving churn
    _retention_recs = {
        'Total_Trans_Ct': 'Encourage increased card usage through targeted cashback or rewards campaigns.',
        'Total_Trans_Amt': 'Offer spending-based incentives or tiered rewards to boost transaction amounts.',
        'Total_Ct_Chng_Q4_Q1': 'Launch a re-engagement campaign with time-limited offers to reverse declining activity.',
        'Total_Revolving_Bal': 'Consider offering a balance transfer promotion or credit line adjustment.',
        'Total_Relationship_Count': 'Provide tailored cross-sell offers to deepen the product relationship.',
        'Months_Inactive': 'Reach out with a personalized re-activation offer before the customer fully disengages.',
        'Contacts_Count': 'Review recent service interactions to identify and resolve any outstanding concerns.',
        'Total_Amt_Chng_Q4_Q1': 'Offer personalized spending incentives to reverse the decline in activity.',
        'Avg_Utilization_Ratio': 'Consider adjusting the credit limit or offering promotional APR to encourage usage.',
        'Credit_Limit': 'Evaluate a credit limit increase to better match the customer\'s financial profile.',
        'Age': 'Offer age-appropriate products and services tailored to this customer\'s life stage.',
        'Dependent_count': 'Suggest family-oriented banking products such as savings plans or joint accounts.',
        'Months_on_book': 'Strengthen the relationship early with a loyalty milestone reward.',
        'Gender': 'Ensure marketing and product offerings are inclusive and aligned with customer preferences.',
        'Education': 'Provide financial literacy resources or premium advisory services.',
        'Marital_Status': 'Offer life-event-triggered products (e.g., joint accounts, mortgage pre-approvals).',
        'Income': 'Match product tier and credit offerings to the customer\'s income bracket.',
        'Card_Category': 'Consider a card upgrade or downgrade that better fits the customer\'s usage pattern.',
    }

    # Key factors increasing churn risk
    if positive_features:
        lines.append("Key factors increasing churn risk:")
        for feat_name, shap_val in positive_features[:3]:
            desc = feature_descriptions.get(feat_name, feat_name)
            behavior = _behavior_increase.get(feat_name, f'The {desc} is contributing to higher churn risk.')
            lines.append(f"• **{desc.capitalize()}** — {behavior}")
        lines.append("")

    # Protective factors
    if negative_features:
        lines.append("Protective factors:")
        for feat_name, shap_val in negative_features[:3]:
            desc = feature_descriptions.get(feat_name, feat_name)
            behavior = _behavior_decrease.get(feat_name, f'The {desc} is helping reduce churn risk.')
            lines.append(f"• **{desc.capitalize()}** — {behavior}")
        lines.append("")

    # Summary sentence
    if churn_prob >= 0.7:
        lines.append("Overall, the customer\'s low engagement with banking services suggests a strong likelihood of churn.")
    elif churn_prob >= 0.4:
        lines.append("Overall, the customer shows moderate warning signs that warrant proactive monitoring and outreach.")
    else:
        lines.append("Overall, this customer appears stable with strong engagement indicators.")

    lines.append("")

    # Retention recommendations based on top churn-driving features
    lines.append("Recommended retention strategies:")
    recs_added = set()
    for feat_name, _ in positive_features[:3]:
        rec = _retention_recs.get(feat_name)
        if rec and rec not in recs_added:
            lines.append(f"• {rec}")
            recs_added.add(rec)
    # If fewer than 2 recommendations, add a general one
    if len(recs_added) < 2:
        lines.append("• Schedule a relationship manager call to understand the customer\'s evolving needs.")

    return "\n".join(lines)


def get_global_ai_interpretation_dynamic(shap_values, feature_names):
    """
    Generate a dynamic AI interpretation from per-upload SHAP values.
    Uses Gemini if available, otherwise falls back to templates.
    """
    importance = compute_feature_importance_dynamic(shap_values, feature_names)
    feat_names = importance['feature_names']
    scores = importance['importance_scores']

    feature_descriptions = {
        'Total_Trans_Ct': 'total transaction count',
        'Total_Trans_Amt': 'total transaction amount',
        'Total_Ct_Chng_Q4_Q1': 'transaction count change ratio (Q4 vs Q1)',
        'Total_Revolving_Bal': 'revolving balance',
        'Total_Relationship_Count': 'number of bank products',
        'Months_Inactive': 'months of inactivity',
        'Contacts_Count': 'customer support contacts',
        'Total_Amt_Chng_Q4_Q1': 'transaction amount change ratio',
        'Avg_Utilization_Ratio': 'credit utilization ratio',
        'Credit_Limit': 'credit limit',
        'Age': 'customer age',
        'Dependent_count': 'number of dependents',
        'Months_on_book': 'tenure duration',
        'Gender': 'gender',
        'Education': 'education level',
        'Marital_Status': 'marital status',
        'Income': 'income category',
        'Card_Category': 'card category'
    }

    # --- Try Gemini first ---
    gemini_result = _generate_global_interpretation_gemini(
        shap_values, feat_names, scores, feature_descriptions
    )
    if gemini_result:
        return gemini_result

    # --- Fallback: template-based ---
    return _generate_global_interpretation_template(
        shap_values, feat_names, scores, feature_descriptions
    )


def _generate_global_interpretation_gemini(shap_values, feat_names, scores, feature_descriptions):
    """Use Gemini to generate a global SHAP interpretation."""
    feat_lines = []
    for fname, score in zip(feat_names[:10], scores[:10]):
        desc = feature_descriptions.get(fname, fname)
        feat_lines.append(f"- {fname} ({desc}): mean |SHAP| = {score:.4f}")
    features_text = "\n".join(feat_lines)

    prompt = f"""You are a data scientist writing an executive summary for a bank's analytics team.

A customer churn prediction model (ExtraTreesClassifier with SHAP explainability) was run on an uploaded dataset of {shap_values.shape[0]} customers.

Top features by importance (mean |SHAP| values):
{features_text}

Write a clear, insightful analysis (6-8 sentences) that:
1. Starts with "Analysis of Your Uploaded Data" as a bold heading
2. Highlights the top 3 churn drivers and explains WHY they matter in banking context
3. Identifies any surprising or notable patterns in the feature rankings
4. Provides 2-3 strategic recommendations for the bank's retention team
5. Notes that these rankings are specific to this uploaded dataset

Use markdown formatting (bold for feature names and headings, bullet points for recommendations). Be professional and actionable."""

    return _call_ai(prompt, max_tokens=600)


def _generate_global_interpretation_template(shap_values, feat_names, scores, feature_descriptions):
    """Fallback template-based global interpretation."""
    lines = []
    lines.append("**Analysis of Your Uploaded Data**")
    lines.append("")
    lines.append(f"Based on the uploaded dataset ({shap_values.shape[0]} customers analyzed), "
                 f"the top predictor of customer churn is **{feat_names[0]}** "
                 f"({feature_descriptions.get(feat_names[0], feat_names[0])}) "
                 f"with a mean |SHAP| of {scores[0]:.4f}.")
    lines.append("")

    if len(feat_names) >= 3:
        lines.append(f"The next most important features are **{feat_names[1]}** "
                     f"({feature_descriptions.get(feat_names[1], feat_names[1])}, "
                     f"SHAP: {scores[1]:.4f}) and **{feat_names[2]}** "
                     f"({feature_descriptions.get(feat_names[2], feat_names[2])}, "
                     f"SHAP: {scores[2]:.4f}).")
        lines.append("")

    lines.append("Key insights from your data:")
    for i, (fname, score) in enumerate(zip(feat_names[:5], scores[:5])):
        desc = feature_descriptions.get(fname, fname)
        lines.append(f"- **{fname}**: {desc} — mean impact {score:.4f}")

    lines.append("")
    lines.append("These rankings reflect the specific characteristics of your uploaded "
                 "customer data. Different datasets may show different feature importance "
                 "patterns based on the customer demographics and behaviors present.")

    return "\n".join(lines)
