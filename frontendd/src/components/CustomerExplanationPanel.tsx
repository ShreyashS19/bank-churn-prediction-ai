import { useState, useEffect } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { RefreshCw, Sparkles, BarChart3 } from "lucide-react";

interface CustomerInfo {
  id: string;
  index: number;
  churn_probability: number;
  risk_level: string;
  data: Record<string, unknown>;
}

interface TopFeature {
  feature: string;
  shap_value: number;
  feature_value: unknown;
  direction: string;
}

interface CustomerExplanation {
  churn_probability: number;
  risk_level: string;
  top_features: TopFeature[];
  all_shap_values: Record<string, number>;
  ai_explanation: string;
}

interface CustomerExplanationPanelProps {
  customers: CustomerInfo[];
  customersLoading: boolean;
  onSelectCustomer: (customer: CustomerInfo) => void;
  explanation: CustomerExplanation | null;
  explanationLoading: boolean;
}

const CustomerExplanationPanel = ({
  customers,
  customersLoading,
  onSelectCustomer,
  explanation,
  explanationLoading,
}: CustomerExplanationPanelProps) => {
  const [selectedId, setSelectedId] = useState<string>("");

  const handleSelect = (value: string) => {
    setSelectedId(value);
    const customer = customers.find((c) => c.id === value);
    if (customer) {
      onSelectCustomer(customer);
    }
  };

  // Auto-select first customer
  useEffect(() => {
    if (customers.length > 0 && !selectedId) {
      const first = customers[0];
      setSelectedId(first.id);
      onSelectCustomer(first);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [customers]);

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case "High Risk":
        return "text-red-400 border-red-500/30 bg-red-500/10";
      case "Medium Risk":
        return "text-yellow-400 border-yellow-500/30 bg-yellow-500/10";
      default:
        return "text-green-400 border-green-500/30 bg-green-500/10";
    }
  };

  if (customersLoading) {
    return (
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {[1, 2, 3].map((i) => (
          <div
            key={i}
            className="p-6 rounded-2xl bg-card border border-border/50 animate-pulse"
          >
            <div className="h-40 bg-muted rounded" />
          </div>
        ))}
      </div>
    );
  }

  // Prepare chart data for top features
  const chartData = explanation
    ? explanation.top_features
        .slice(0, 3)
        .map((f) => ({
          name: f.feature.replace(/_/g, " "),
          value: Math.abs(f.shap_value),
          rawValue: f.shap_value,
          direction: f.direction,
        }))
        .reverse()
    : [];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Customer Selector Card */}
      <div className="p-6 rounded-2xl bg-card border border-border/50 shadow-elevated">
        <h3 className="font-semibold text-foreground mb-4">Select Customer</h3>
        <Select value={selectedId} onValueChange={handleSelect}>
          <SelectTrigger className="w-full mb-6">
            <SelectValue placeholder="Select a customer..." />
          </SelectTrigger>
          <SelectContent className="max-h-60">
            {customers.map((customer) => (
              <SelectItem key={customer.id} value={customer.id}>
                {customer.id} — {customer.risk_level}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        {explanationLoading ? (
          <div className="space-y-3 animate-pulse">
            <div className="h-4 bg-muted rounded w-32" />
            <div className="h-12 bg-muted rounded w-24" />
            <div className="h-4 bg-muted rounded w-20" />
            <div className="h-8 bg-muted rounded w-24" />
          </div>
        ) : explanation ? (
          <div>
            <p className="text-sm text-muted-foreground mb-1">
              Churn Probability
            </p>
            <p className="text-4xl font-bold text-foreground mb-4">
              {(explanation.churn_probability * 100).toFixed(1)}%
            </p>
            <p className="text-sm text-muted-foreground mb-2">Risk Level</p>
            <span
              className={`inline-block px-3 py-1 rounded-full text-sm font-medium border ${getRiskColor(
                explanation.risk_level
              )}`}
            >
              {explanation.risk_level}
            </span>

            {/* Risk progress bar */}
            <div className="mt-4 w-full bg-muted rounded-full h-2 overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                  explanation.churn_probability >= 0.7
                    ? "bg-red-500"
                    : explanation.churn_probability >= 0.4
                    ? "bg-yellow-500"
                    : "bg-green-500"
                }`}
                style={{
                  width: `${explanation.churn_probability * 100}%`,
                }}
              />
            </div>
          </div>
        ) : null}
      </div>

      {/* Top Contributing Features Chart */}
      <div className="p-6 rounded-2xl bg-card border border-border/50 shadow-elevated">
        <div className="flex items-center gap-2 mb-4">
          <BarChart3 className="w-5 h-5 text-primary" />
          <h3 className="font-semibold text-foreground">
            Top Contributing Features
          </h3>
        </div>

        {explanationLoading ? (
          <div className="h-64 bg-muted rounded animate-pulse" />
        ) : chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={250}>
            <BarChart
              data={chartData}
              layout="vertical"
              margin={{ top: 5, right: 20, left: 80, bottom: 5 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="hsl(var(--border))"
                opacity={0.3}
              />
              <XAxis
                type="number"
                tick={{
                  fill: "hsl(var(--muted-foreground))",
                  fontSize: 12,
                }}
                axisLine={{ stroke: "hsl(var(--border))" }}
              />
              <YAxis
                type="category"
                dataKey="name"
                tick={{
                  fill: "hsl(var(--muted-foreground))",
                  fontSize: 12,
                }}
                axisLine={{ stroke: "hsl(var(--border))" }}
                width={75}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "12px",
                  padding: "12px",
                  color: "hsl(var(--foreground))",
                }}
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
              formatter={(value: number, _name: string, props: Record<string, any>) => [
                  `${value.toFixed(4)} (${props.payload.direction})`,
                  "SHAP Impact",
                ]}
              />
              <Bar dataKey="value" radius={[0, 6, 6, 0]}>
                {chartData.map((entry, index) => (
                  <Cell
                    key={index}
                    fill={
                      entry.rawValue > 0 ? "#ef4444" : "#3b82f6"
                    }
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-sm text-muted-foreground italic">
            Select a customer to see feature contributions.
          </p>
        )}
      </div>

      {/* AI Explanation */}
      <div className="p-6 rounded-2xl bg-card border border-border/50 shadow-elevated">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-primary" />
            <h3 className="font-semibold text-foreground">AI Explanation</h3>
          </div>
          {selectedId && (
            <button
              onClick={() => {
                const customer = customers.find((c) => c.id === selectedId);
                if (customer) onSelectCustomer(customer);
              }}
              disabled={explanationLoading}
              className="p-2 rounded-lg hover:bg-secondary transition-colors"
              title="Regenerate explanation"
            >
              <RefreshCw
                className={`w-4 h-4 text-muted-foreground ${
                  explanationLoading ? "animate-spin" : ""
                }`}
              />
            </button>
          )}
        </div>

        {explanationLoading ? (
          <div className="space-y-2 animate-pulse">
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="h-3 bg-muted rounded w-full" />
            ))}
          </div>
        ) : explanation?.ai_explanation ? (
          <div className="text-sm text-muted-foreground space-y-2 leading-relaxed max-h-[300px] overflow-y-auto">
            {explanation.ai_explanation.split("\n").map((line, i) => {
              if (!line.trim()) return <div key={i} className="h-2" />;
              if (line.startsWith("**") && line.endsWith("**")) {
                return (
                  <p key={i} className="font-semibold text-foreground">
                    {line.replace(/\*\*/g, "")}
                  </p>
                );
              }
              if (line.startsWith("- **")) {
                const content = line
                  .replace(/^- /, "")
                  .replace(/\*\*/g, "");
                const parts = content.split(" — ");
                return (
                  <p key={i} className="pl-2">
                    <span className="font-medium text-foreground">
                      • {parts[0]}
                    </span>
                    {parts.length > 1 ? ` — ${parts.slice(1).join(" — ")}` : ""}
                  </p>
                );
              }
              return <p key={i}>{line.replace(/\*\*/g, "")}</p>;
            })}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground italic">
            Click regenerate to load AI insights.
          </p>
        )}
      </div>
    </div>
  );
};

export default CustomerExplanationPanel;
