const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

export interface PredictionResponse {
  predictions: Array<Record<string, unknown>>;
  csv_path: string;
}

export interface PredictionError {
  error: string;
}

export interface ModelMetrics {
  accuracy: number;
  roc_auc: number;
  f1_score: number;
  model: string;
  estimators: number;
}

export interface FeatureImportanceData {
  feature_names: string[];
  importance_scores: number[];
}

export interface ShapPoint {
  shap_value: number;
  feature_value: number;
  normalized_value: number;
}

export interface ShapFeature {
  feature_name: string;
  points: ShapPoint[];
}

export interface CustomerInfo {
  id: string;
  index: number;
  churn_probability: number;
  risk_level: string;
  data: Record<string, unknown>;
}

export interface TopFeature {
  feature: string;
  shap_value: number;
  feature_value: unknown;
  direction: string;
}

export interface CustomerExplanation {
  churn_probability: number;
  risk_level: string;
  top_features: TopFeature[];
  all_shap_values: Record<string, number>;
  ai_explanation: string;
}

export const apiService = {
  /**
   * Send customer data to Flask backend for churn prediction
   */
  async predictChurn(csvData: Array<Record<string, unknown>>): Promise<PredictionResponse> {
    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(csvData),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Prediction failed');
      }

      return data;
    } catch (error) {
      if (error instanceof TypeError && error.message.includes('fetch')) {
        throw new Error(
          'Unable to connect to Flask backend. Please ensure the server is running on http://localhost:5000'
        );
      }
      throw error;
    }
  },

  /**
   * Download prediction results CSV from Flask backend
   */
  async downloadResults(): Promise<Blob> {
    try {
      const response = await fetch(`${API_BASE_URL}/download`);
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Download failed');
      }

      return response.blob();
    } catch (error) {
      if (error instanceof TypeError && error.message.includes('fetch')) {
        throw new Error('Unable to connect to Flask backend.');
      }
      throw error;
    }
  },

  /**
   * Check if Flask backend is running and healthy
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${API_BASE_URL}/health`, {
        method: 'GET',
      });
      return response.ok;
    } catch {
      return false;
    }
  },

  /**
   * Get model performance metrics
   */
  async getMetrics(): Promise<ModelMetrics> {
    const response = await fetch(`${API_BASE_URL}/metrics`);
    if (!response.ok) throw new Error('Failed to fetch metrics');
    return response.json();
  },

  /**
   * Get global feature importance (top 10)
   */
  async getFeatureImportance(): Promise<FeatureImportanceData> {
    const response = await fetch(`${API_BASE_URL}/feature-importance`);
    if (!response.ok) throw new Error('Failed to fetch feature importance');
    return response.json();
  },

  /**
   * Get SHAP distribution data for scatter plot
   */
  async getShapDistribution(): Promise<ShapFeature[]> {
    const response = await fetch(`${API_BASE_URL}/shap-distribution`);
    if (!response.ok) throw new Error('Failed to fetch SHAP distribution');
    return response.json();
  },

  /**
   * Get sample customers for individual explanation
   */
  async getSampleCustomers(): Promise<CustomerInfo[]> {
    const response = await fetch(`${API_BASE_URL}/sample-customers`);
    if (!response.ok) throw new Error('Failed to fetch sample customers');
    return response.json();
  },

  /**
   * Get individual customer SHAP explanation
   */
  async getCustomerExplanation(customerData: Record<string, unknown>): Promise<CustomerExplanation> {
    const response = await fetch(`${API_BASE_URL}/customer-explanation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(customerData),
    });
    if (!response.ok) throw new Error('Failed to fetch customer explanation');
    return response.json();
  },

  /**
   * Get global AI interpretation of feature importance
   */
  async getGlobalInterpretation(): Promise<string> {
    const response = await fetch(`${API_BASE_URL}/global-interpretation`);
    if (!response.ok) throw new Error('Failed to fetch interpretation');
    const data = await response.json();
    return data.interpretation;
  },

  /**
   * Get API base URL (useful for debugging)
   */
  getBaseUrl(): string {
    return API_BASE_URL;
  },
};

