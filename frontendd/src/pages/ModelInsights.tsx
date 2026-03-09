import { useState, useEffect, useCallback } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";
import { ThemeToggle } from "@/components/ThemeToggle";
import ModelMetricsCards from "@/components/ModelMetricsCards";
import FeatureImportanceChart from "@/components/FeatureImportanceChart";
import ShapDistributionChart from "@/components/ShapDistributionChart";
import CustomerExplanationPanel from "@/components/CustomerExplanationPanel";
import { Info, Lock, Loader2 } from "lucide-react";

const API_BASE_URL = "http://localhost:5000";

interface ModelMetrics {
  accuracy: number;
  roc_auc: number;
  f1_score: number;
  model: string;
  estimators: number;
}

interface FeatureImportanceData {
  feature_names: string[];
  importance_scores: number[];
}

interface ShapPoint {
  shap_value: number;
  feature_value: number;
  normalized_value: number;
}

interface ShapFeature {
  feature_name: string;
  points: ShapPoint[];
}

interface CustomerInfo {
  id: string;
  index: number;
  churn_probability: number;
  risk_level: string;
  data: Record<string, unknown>;
}

interface CustomerExplanation {
  churn_probability: number;
  risk_level: string;
  top_features: Array<{
    feature: string;
    shap_value: number;
    feature_value: unknown;
    direction: string;
  }>;
  all_shap_values: Record<string, number>;
  ai_explanation: string;
}

const ModelInsights = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const sessionId = searchParams.get("session");

  const [activeTab, setActiveTab] = useState<"overview" | "insights">(
    "insights"
  );

  // Data states
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [metricsLoading, setMetricsLoading] = useState(true);

  const [featureImportance, setFeatureImportance] =
    useState<FeatureImportanceData | null>(null);
  const [featureLoading, setFeatureLoading] = useState(true);

  const [shapDistribution, setShapDistribution] = useState<
    ShapFeature[] | null
  >(null);
  const [shapLoading, setShapLoading] = useState(true);

  const [interpretation, setInterpretation] = useState<string | null>(null);
  const [interpretationLoading, setInterpretationLoading] = useState(false);

  const [customers, setCustomers] = useState<CustomerInfo[]>([]);
  const [customersLoading, setCustomersLoading] = useState(true);

  const [customerExplanation, setCustomerExplanation] =
    useState<CustomerExplanation | null>(null);
  const [explanationLoading, setExplanationLoading] = useState(false);

  // SHAP readiness state (background computation)
  const [shapReady, setShapReady] = useState(false);
  const [shapError, setShapError] = useState<string | null>(null);

  // Poll /shap-status until ready
  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;
    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/shap-status?session_id=${sessionId}`);
        const data = await res.json();
        if (cancelled) return;
        if (data.error && typeof data.error === 'string' && data.error !== null) {
          setShapError(data.error);
          return; // stop polling
        }
        if (data.ready) {
          setShapReady(true);
          return; // stop polling
        }
      } catch (err) {
        console.error('Polling shap-status failed:', err);
      }
      if (!cancelled) {
        setTimeout(poll, 1500);
      }
    };
    poll();
    return () => { cancelled = true; };
  }, [sessionId]);

  // Fetch metrics (static - always available)
  useEffect(() => {
    if (!sessionId) return;
    const fetchMetrics = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/metrics`);
        const data = await res.json();
        setMetrics(data);
      } catch (err) {
        console.error("Failed to load metrics:", err);
      } finally {
        setMetricsLoading(false);
      }
    };
    fetchMetrics();
  }, [sessionId]);

  // Fetch feature importance (dynamic per upload — wait for SHAP)
  useEffect(() => {
    if (!sessionId || !shapReady) return;
    const fetchFeatureImportance = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/feature-importance?session_id=${sessionId}`);
        const data = await res.json();
        setFeatureImportance(data);
      } catch (err) {
        console.error("Failed to load feature importance:", err);
      } finally {
        setFeatureLoading(false);
      }
    };
    fetchFeatureImportance();
  }, [sessionId, shapReady]);

  // Fetch SHAP distribution (dynamic per upload — wait for SHAP)
  useEffect(() => {
    if (!sessionId || !shapReady) return;
    const fetchShapDistribution = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/shap-distribution?session_id=${sessionId}`);
        const data = await res.json();
        if (Array.isArray(data)) {
          setShapDistribution(data);
        } else {
          console.error('SHAP distribution response is not an array:', data);
        }
      } catch (err) {
        console.error("Failed to load SHAP distribution:", err);
      } finally {
        setShapLoading(false);
      }
    };
    fetchShapDistribution();
  }, [sessionId, shapReady]);

  // Fetch sample customers (dynamic per upload)
  useEffect(() => {
    if (!sessionId) return;
    const fetchCustomers = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/sample-customers?session_id=${sessionId}`);
        const data = await res.json();
        setCustomers(data);
      } catch (err) {
        console.error("Failed to load customers:", err);
      } finally {
        setCustomersLoading(false);
      }
    };
    fetchCustomers();
  }, [sessionId]);

  // Load global AI interpretation (dynamic per upload)
  const loadInterpretation = useCallback(async () => {
    if (!sessionId) return;
    setInterpretationLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/global-interpretation?session_id=${sessionId}`);
      const data = await res.json();
      setInterpretation(data.interpretation);
    } catch (err) {
      console.error("Failed to load interpretation:", err);
    } finally {
      setInterpretationLoading(false);
    }
  }, [sessionId]);

  // Load customer explanation
  const handleSelectCustomer = useCallback(async (customer: CustomerInfo) => {
    setExplanationLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/customer-explanation`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(customer.data),
      });
      const data = await res.json();
      setCustomerExplanation(data);
    } catch (err) {
      console.error("Failed to load customer explanation:", err);
    } finally {
      setExplanationLoading(false);
    }
  }, []);

  const scrollToUpload = () => {
    window.location.href = "/";
  };

  // Guard: require session_id
  if (!sessionId) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center max-w-md mx-auto p-8">
          <div className="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center mx-auto mb-6">
            <Lock className="w-8 h-8 text-primary" />
          </div>
          <h2 className="text-2xl font-bold font-display text-foreground mb-3">
            Upload Required
          </h2>
          <p className="text-muted-foreground mb-6">
            Please upload a CSV file first to generate dynamic model insights
            based on your data. Each upload produces unique SHAP-based analysis.
          </p>
          <button
            onClick={() => navigate("/")}
            className="px-6 py-3 bg-primary text-primary-foreground rounded-xl font-medium hover:opacity-90 transition-opacity"
          >
            Go to Upload
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-50 glass border-b">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <a href="/" className="flex items-center gap-2">
            <div className="w-10 h-10 rounded-xl gradient-primary flex items-center justify-center">
              <span className="text-primary-foreground font-bold text-lg">
                C
              </span>
            </div>
            <span className="font-display font-bold text-xl text-foreground">
              ChurnPredict
            </span>
          </a>

          <nav className="hidden md:flex items-center gap-8">
            <button
              onClick={scrollToUpload}
              className="flex items-center gap-2 text-sm font-medium text-muted-foreground hover:text-primary transition-colors duration-300 group cursor-pointer"
            >
              <span className="inline-block transform group-hover:translate-x-1 transition-transform">
                ⚡
              </span>
              Quick Start
            </button>
            <ThemeToggle />
          </nav>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        {/* Tab Switcher */}
        <div className="flex justify-center mb-8">
          <div className="inline-flex bg-card border border-border/50 rounded-full p-1 shadow-sm">
            <button
              onClick={() => navigate("/")}
              className={`px-6 py-2 rounded-full text-sm font-medium transition-all duration-300 ${
                activeTab === "overview"
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Overview
            </button>
            <button
              onClick={() => setActiveTab("insights")}
              className={`px-6 py-2 rounded-full text-sm font-medium transition-all duration-300 ${
                activeTab === "insights"
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Model Insights
            </button>
          </div>
        </div>

        {/* Model Performance Section */}
        <section className="mb-12">
          <h2 className="text-2xl md:text-3xl font-bold font-display text-foreground mb-2">
            Model Performance
          </h2>
          <p className="text-muted-foreground mb-6">
            Performance metrics evaluated on test dataset
          </p>
          <ModelMetricsCards metrics={metrics} loading={metricsLoading} />
        </section>

        {/* Global Feature Importance Section */}
        {!shapReady && !shapError && (
          <section className="mb-12">
            <div className="flex flex-col items-center justify-center py-16 gap-4">
              <Loader2 className="w-10 h-10 text-primary animate-spin" />
              <p className="text-muted-foreground text-lg font-medium">
                Analyzing your data with SHAP…
              </p>
              <p className="text-sm text-muted-foreground/70">
                This runs in the background and usually takes 10-30 seconds.
              </p>
            </div>
          </section>
        )}

        {shapError && (
          <section className="mb-12">
            <div className="p-6 rounded-2xl bg-destructive/10 border border-destructive/30 text-center">
              <p className="text-destructive font-medium">SHAP computation failed: {shapError}</p>
            </div>
          </section>
        )}

        {shapReady && (
          <>
        <section className="mb-12">
          <h2 className="text-2xl md:text-3xl font-bold font-display text-foreground mb-2">
            Feature Importance — Your Data
          </h2>
          <p className="text-muted-foreground mb-6">
            Top features ranked by SHAP importance based on your uploaded CSV
          </p>
          <FeatureImportanceChart
            data={featureImportance}
            loading={featureLoading}
            interpretation={interpretation}
            onLoadInterpretation={loadInterpretation}
            interpretationLoading={interpretationLoading}
          />
        </section>

        {/* Feature Impact Distribution Section */}
        <section className="mb-12">
          <h2 className="text-2xl md:text-3xl font-bold font-display text-foreground mb-2">
            Feature Impact Distribution — Your Data
          </h2>
          <p className="text-muted-foreground mb-6">
            SHAP values across customers in your uploaded dataset
          </p>
          <ShapDistributionChart
            data={shapDistribution}
            loading={shapLoading}
          />
        </section>

        {/* Individual Customer Explanation Section */}
        <section className="mb-12">
          <h2 className="text-2xl md:text-3xl font-bold font-display text-foreground mb-2">
            Individual Customer Explanation
          </h2>
          <p className="text-muted-foreground mb-6">
            Select a customer from your uploaded data for personalized churn analysis
          </p>
          <CustomerExplanationPanel
            customers={customers}
            customersLoading={customersLoading}
            onSelectCustomer={handleSelectCustomer}
            explanation={customerExplanation}
            explanationLoading={explanationLoading}
          />
        </section>
          </>
        )}

        {/* Model Transparency Section */}
        <section className="mb-12">
          <div className="p-6 rounded-2xl bg-primary/5 border border-primary/20">
            <div className="flex items-start gap-3">
              <Info className="w-6 h-6 text-primary mt-0.5 flex-shrink-0" />
              <div>
                <h3 className="font-semibold text-foreground mb-2">
                  Model Transparency
                </h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  This system uses{" "}
                  <strong className="text-foreground">
                    SHAP (SHapley Additive exPlanations)
                  </strong>{" "}
                  to ensure model interpretability and fairness. SHAP values are
                  computed from cooperative game theory, providing consistent and
                  locally accurate feature attributions. Each prediction can be
                  traced back to its contributing features, enabling transparent
                  decision-making and regulatory compliance.
                </p>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t bg-card">
        <div className="container mx-auto px-4 py-8">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg gradient-primary flex items-center justify-center">
                <span className="text-primary-foreground font-bold text-sm">
                  C
                </span>
              </div>
              <span className="font-display font-semibold text-foreground">
                ChurnPredict
              </span>
            </div>
            <p className="text-sm text-muted-foreground">
              © {new Date().getFullYear()} ChurnPredict. AI-powered customer
              retention analytics.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default ModelInsights;
