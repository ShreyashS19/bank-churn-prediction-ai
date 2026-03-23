import { Brain, Sparkles, BarChart3, MessageSquare } from "lucide-react";

const capabilities = [
  {
    icon: Brain,
    title: "Churn Risk Prediction",
    description: (
      <>
        <strong>Predict customers likely to leave</strong> using an advanced{" "}
        <strong>ExtraTreesClassifier</strong> <strong>Ensemble Model</strong>{" "}
        trained on <strong>Real Banking Data</strong>.
      </>
    ),
    iconBg: "bg-purple-600",
  },
  {
    icon: Sparkles,
    title: "Explainable AI (SHAP)",
    description: (
      <>
        Understand which <strong>features influence</strong> each churn decision
        with <strong>SHAP values</strong> — enabling full{" "}
        <strong>model transparency</strong>.
      </>
    ),
    iconBg: "bg-blue-600",
  },
  {
    icon: BarChart3,
    title: "Interactive Analytics",
    description: (
      <>
        Explore <strong>churn patterns</strong> through{" "}
        <strong>dynamic visual dashboards</strong> with{" "}
        <strong>feature importance</strong> and{" "}
        <strong>distribution analysis</strong>.
      </>
    ),
    iconBg: "bg-emerald-600",
  },
  {
    icon: MessageSquare,
    title: "AI-Generated Insights",
    description: (
      <>
        Automatically generate <strong>human-readable insights</strong> about{" "}
        <strong>churn drivers</strong> and{" "}
        <strong>actionable retention strategies</strong>.
      </>
    ),
    iconBg: "bg-orange-500",
  },
];

const ProductCapabilities = () => {
  return (
    <section className="py-20 lg:py-28 bg-background">
      <div className="container mx-auto px-4">
        {/* Section Header */}
        <div className="text-center mb-16">
          <p className="text-sm font-semibold uppercase tracking-[0.2em] bg-gradient-to-r from-orange-400 via-pink-400 to-purple-400 bg-clip-text text-transparent mb-4">
            Product Capabilities
          </p>
          <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold font-display text-foreground mb-5">
            Everything You Need for Churn Intelligence
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            A complete AI-powered suite to predict, explain, and act on customer
            churn.
          </p>
        </div>

        {/* Capability Cards — single row */}
        <div className="max-w-6xl mx-auto grid sm:grid-cols-2 lg:grid-cols-4 gap-5">
          {capabilities.map((cap, index) => (
            <div
              key={cap.title}
              className="group p-6 rounded-2xl bg-card border border-border hover:border-primary/40 shadow-elevated hover:shadow-lg transition-all duration-300 animate-fade-up"
              style={{ animationDelay: `${index * 0.12}s` }}
            >
              <div
                className={`w-12 h-12 rounded-xl ${cap.iconBg} flex items-center justify-center mb-5`}
              >
                <cap.icon className="w-6 h-6 text-white" />
              </div>

              <h3 className="text-lg font-bold text-foreground mb-3 font-display">
                {cap.title}
              </h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                {cap.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default ProductCapabilities;
