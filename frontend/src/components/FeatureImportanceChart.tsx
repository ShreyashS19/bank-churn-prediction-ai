import { useState } from "react";
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
import { RefreshCw, Sparkles } from "lucide-react";

interface FeatureImportanceData {
  feature_names: string[];
  importance_scores: number[];
}

interface FeatureImportanceChartProps {
  data: FeatureImportanceData | null;
  loading: boolean;
  interpretation: string | null;
  onLoadInterpretation: () => void;
  interpretationLoading: boolean;
}

const FeatureImportanceChart = ({
  data,
  loading,
  interpretation,
  onLoadInterpretation,
  interpretationLoading,
}: FeatureImportanceChartProps) => {
  if (loading) {
    return (
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 p-6 rounded-2xl bg-card border border-border/50 animate-pulse">
          <div className="h-80 bg-muted rounded" />
        </div>
        <div className="p-6 rounded-2xl bg-card border border-border/50 animate-pulse">
          <div className="h-40 bg-muted rounded" />
        </div>
      </div>
    );
  }

  if (!data) return null;

  // Prepare chart data (reversed for horizontal layout - bottom to top)
  const chartData = data.feature_names
    .map((name, i) => ({
      name: name.replace(/_/g, " "),
      fullName: name,
      importance: data.importance_scores[i],
    }))
    .reverse();

  // Color coding: negative SHAP features get red-ish bars
  const negativeFeatures = ["Months_Inactive", "Contacts_Count"];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Chart */}
      <div className="lg:col-span-2 p-6 rounded-2xl bg-card border border-border/50 shadow-elevated">
        <ResponsiveContainer width="100%" height={400}>
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 80, bottom: 5 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="hsl(var(--border))"
              opacity={0.3}
            />
            <XAxis
              type="number"
              tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
              axisLine={{ stroke: "hsl(var(--border))" }}
            />
            <YAxis
              type="category"
              dataKey="name"
              tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
              axisLine={{ stroke: "hsl(var(--border))" }}
              width={90}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(var(--card))",
                border: "1px solid hsl(var(--border))",
                borderRadius: "12px",
                padding: "12px",
                color: "hsl(var(--foreground))",
              }}
              formatter={(value: number) => [
                value.toFixed(4),
                "Mean |SHAP|",
              ]}
            />
            <Bar dataKey="importance" radius={[0, 6, 6, 0]}>
              {chartData.map((entry) => (
                <Cell
                  key={entry.fullName}
                  fill={
                    negativeFeatures.includes(entry.fullName)
                      ? "#ef4444"
                      : "#3b82f6"
                  }
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* AI Interpretation */}
      <div className="p-6 rounded-2xl bg-card border border-border/50 shadow-elevated">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-primary" />
            <h3 className="font-semibold text-foreground">
              AI Interpretation
            </h3>
          </div>
          <button
            onClick={onLoadInterpretation}
            disabled={interpretationLoading}
            className="p-2 rounded-lg hover:bg-secondary transition-colors"
            title="Regenerate interpretation"
          >
            <RefreshCw
              className={`w-4 h-4 text-muted-foreground ${
                interpretationLoading ? "animate-spin" : ""
              }`}
            />
          </button>
        </div>

        {interpretation ? (
          <div className="text-sm text-muted-foreground space-y-2 leading-relaxed max-h-[340px] overflow-y-auto">
            {interpretation.split("\n").map((line, i) => {
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
                const parts = content.split(":");
                return (
                  <p key={i} className="pl-2">
                    <span className="font-medium text-foreground">
                      {parts[0]}
                    </span>
                    :{parts.slice(1).join(":")}
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

export default FeatureImportanceChart;
