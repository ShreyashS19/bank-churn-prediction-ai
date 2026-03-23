import { Activity, BarChart3, FlaskConical, TreePine } from "lucide-react";

interface ModelMetrics {
  accuracy: number;
  roc_auc: number;
  roc_auc_display?: string;
  f1_score: number;
  f1_display?: string;
  model: string;
  estimators: number;
}

interface ModelMetricsCardsProps {
  metrics: ModelMetrics | null;
  loading: boolean;
}

const ModelMetricsCards = ({ metrics, loading }: ModelMetricsCardsProps) => {
  const cards = [
    {
      icon: Activity,
      label: "Accuracy",
      value: metrics ? `${(metrics.accuracy * 100).toFixed(1)}%` : "—",
      subtitle: "Test set",
      color: "text-blue-400",
    },
    {
      icon: BarChart3,
      label: "AUC-ROC",
      value: metrics ? (metrics.roc_auc_display ?? metrics.roc_auc.toFixed(2)) : "—",
      subtitle: "Area under curve",
      color: "text-green-400",
    },
    {
      icon: FlaskConical,
      label: "F1 Score",
      value: metrics ? (metrics.f1_display ?? metrics.f1_score.toFixed(2)) : "—",
      subtitle: "Harmonic mean",
      color: "text-purple-400",
    },
    {
      icon: TreePine,
      label: "Model",
      value: metrics ? metrics.model : "—",
      subtitle: metrics
        ? `v1.0 · ${metrics.estimators} estimators`
        : "—",
      color: "text-orange-400",
      isText: true,
    },
  ];

  if (loading) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {[1, 2, 3, 4].map((i) => (
          <div
            key={i}
            className="p-6 rounded-2xl bg-card border border-border/50 animate-pulse"
          >
            <div className="h-4 bg-muted rounded w-20 mb-3" />
            <div className="h-10 bg-muted rounded w-24 mb-2" />
            <div className="h-3 bg-muted rounded w-16" />
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {cards.map((card) => (
        <div
          key={card.label}
          className="p-6 rounded-2xl bg-card border border-border/50 shadow-elevated hover:shadow-lg transition-all duration-300"
        >
          <div className="flex items-center gap-2 mb-2">
            <card.icon className={`w-4 h-4 ${card.color}`} />
            <span className="text-sm text-muted-foreground font-medium">
              {card.label}
            </span>
          </div>
          <p
            className={`font-bold text-foreground mb-1 ${
              card.isText ? "text-2xl" : "text-3xl"
            }`}
          >
            {card.value}
          </p>
          <p className="text-xs text-muted-foreground">{card.subtitle}</p>
        </div>
      ))}
    </div>
  );
};

export default ModelMetricsCards;
