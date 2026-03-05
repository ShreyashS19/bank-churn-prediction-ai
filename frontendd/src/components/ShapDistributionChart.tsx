import { useMemo } from "react";

interface ShapPoint {
  shap_value: number;
  feature_value: number;
  normalized_value: number;
}

interface ShapFeature {
  feature_name: string;
  points: ShapPoint[];
}

interface ShapDistributionChartProps {
  data: ShapFeature[] | null;
  loading: boolean;
}

const ShapDistributionChart = ({
  data,
  loading,
}: ShapDistributionChartProps) => {
  // Compute chart dimensions
  const chartConfig = useMemo(() => {
    if (!data || !Array.isArray(data)) return null;

    const allShapValues = data.flatMap((f) => f.points.map((p) => p.shap_value));
    const minShap = Math.min(...allShapValues);
    const maxShap = Math.max(...allShapValues);
    const absMax = Math.max(Math.abs(minShap), Math.abs(maxShap));
    const range = [-absMax * 1.1, absMax * 1.1];

    return { range, features: data };
  }, [data]);

  if (loading) {
    return (
      <div className="p-6 rounded-2xl bg-card border border-border/50 animate-pulse">
        <div className="h-96 bg-muted rounded" />
      </div>
    );
  }

  if (!chartConfig) return null;

  const { range, features } = chartConfig;
  const width = 1200;
  const marginLeft = 160;
  const marginRight = 40;
  const marginTop = 20;
  const marginBottom = 50;
  const plotWidth = width - marginLeft - marginRight;
  const rowHeight = 50;
  const height = features.length * rowHeight + marginTop + marginBottom;

  // Map SHAP value to x position
  const xScale = (val: number) => {
    return marginLeft + ((val - range[0]) / (range[1] - range[0])) * plotWidth;
  };

  // Y position for each feature
  const yScale = (index: number) => {
    return marginTop + index * rowHeight + rowHeight / 2;
  };

  // Color interpolation: blue (low) -> purple -> red (high)
  const getColor = (normalized: number) => {
    const r = Math.round(60 + normalized * 195);
    const g = Math.round(60 + (1 - Math.abs(normalized - 0.5) * 2) * 40);
    const b = Math.round(255 - normalized * 195);
    return `rgb(${r}, ${g}, ${b})`;
  };

  // Tick values for x-axis
  const ticks = [-0.4, -0.2, 0, 0.2, 0.4].filter(
    (t) => t >= range[0] && t <= range[1]
  );

  return (
    <div className="p-6 rounded-2xl bg-card border border-border/50 shadow-elevated">
      <div className="flex items-center gap-4 mb-4">
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full bg-red-400" />
          <span className="text-xs text-muted-foreground">
            High feature value
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full bg-blue-400" />
          <span className="text-xs text-muted-foreground">
            Low feature value
          </span>
        </div>
      </div>

      <div className="overflow-x-auto">
        <svg
          width="100%"
          height={height}
          viewBox={`0 0 ${width} ${height}`}
          className="min-w-[700px]"
        >
          {/* Grid lines */}
          {ticks.map((tick) => (
            <line
              key={tick}
              x1={xScale(tick)}
              y1={marginTop}
              x2={xScale(tick)}
              y2={height - marginBottom}
              stroke="hsl(var(--border))"
              strokeDasharray="4 4"
              opacity={0.4}
            />
          ))}

          {/* Zero line */}
          <line
            x1={xScale(0)}
            y1={marginTop}
            x2={xScale(0)}
            y2={height - marginBottom}
            stroke="hsl(var(--border))"
            opacity={0.6}
          />

          {/* Feature rows */}
          {features.map((feature, featureIndex) => (
            <g key={feature.feature_name}>
              {/* Feature label */}
              <text
                x={marginLeft - 10}
                y={yScale(featureIndex)}
                textAnchor="end"
                dominantBaseline="middle"
                fill="hsl(var(--muted-foreground))"
                fontSize={12}
              >
                {feature.feature_name.replace(/_/g, " ")}
              </text>

              {/* Dots */}
              {feature.points.map((point, pointIndex) => (
                <circle
                  key={pointIndex}
                  cx={xScale(point.shap_value)}
                  cy={
                    yScale(featureIndex) +
                    (Math.random() - 0.5) * (rowHeight * 0.6)
                  }
                  r={3.5}
                  fill={getColor(point.normalized_value)}
                  opacity={0.7}
                />
              ))}
            </g>
          ))}

          {/* X-axis */}
          <line
            x1={marginLeft}
            y1={height - marginBottom}
            x2={width - marginRight}
            y2={height - marginBottom}
            stroke="hsl(var(--border))"
          />

          {/* X-axis ticks */}
          {ticks.map((tick) => (
            <g key={`tick-${tick}`}>
              <line
                x1={xScale(tick)}
                y1={height - marginBottom}
                x2={xScale(tick)}
                y2={height - marginBottom + 5}
                stroke="hsl(var(--muted-foreground))"
              />
              <text
                x={xScale(tick)}
                y={height - marginBottom + 20}
                textAnchor="middle"
                fill="hsl(var(--muted-foreground))"
                fontSize={12}
              >
                {tick}
              </text>
            </g>
          ))}

          {/* X-axis label */}
          <text
            x={marginLeft + plotWidth / 2}
            y={height - 5}
            textAnchor="middle"
            fill="hsl(var(--muted-foreground))"
            fontSize={12}
          >
            SHAP Value (impact on prediction)
          </text>
        </svg>
      </div>
    </div>
  );
};

export default ShapDistributionChart;
