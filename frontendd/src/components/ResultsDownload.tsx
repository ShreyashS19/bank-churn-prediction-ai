import { Download, FileSpreadsheet, Users, AlertTriangle, TrendingUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

interface PredictionResult {
  totalRecords: number;
  churnCount: number;
  retainCount: number;
  churnRate: number;
  downloadUrl?: string;
  data?: Array<Record<string, unknown>>;
}

interface ResultsDownloadProps {
  results: PredictionResult;
  filename: string;
  onDownload: () => void;
  onNewPrediction: () => void;
}

const ResultsDownload = ({ 
  results, 
  filename, 
  onDownload,
  onNewPrediction 
}: ResultsDownloadProps) => {
  const stats = [
    {
      icon: Users,
      label: "Total Customers",
      value: results.totalRecords.toLocaleString(),
      color: "text-primary",
      bgColor: "bg-primary/10",
    },
    {
      icon: AlertTriangle,
      label: "At Risk",
      value: results.churnCount.toLocaleString(),
      color: "text-destructive",
      bgColor: "bg-destructive/10",
    },
    {
      icon: TrendingUp,
      label: "Likely to Stay",
      value: results.retainCount.toLocaleString(),
      color: "text-success",
      bgColor: "bg-success/10",
    },
  ];

  return (
    <div className="w-full max-w-2xl mx-auto space-y-6 animate-fade-up">
      {/* Stats Grid */}
      <div className="grid grid-cols-3 gap-4">
        {stats.map((stat) => (
          <Card 
            key={stat.label}
            className="p-4 bg-card border shadow-elevated"
          >
            <div className={`w-10 h-10 rounded-lg ${stat.bgColor} flex items-center justify-center mb-3`}>
              <stat.icon className={`w-5 h-5 ${stat.color}`} />
            </div>
            <p className="text-2xl font-bold text-foreground">{stat.value}</p>
            <p className="text-sm text-muted-foreground">{stat.label}</p>
          </Card>
        ))}
      </div>

      {/* Churn Rate Indicator */}
      <Card className="p-6 bg-card border shadow-elevated">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-foreground">Churn Risk Rate</h3>
          <span className={`text-3xl font-bold ${
            results.churnRate > 30 ? "text-destructive" : 
            results.churnRate > 15 ? "text-warning" : 
            "text-success"
          }`}>
            {results.churnRate.toFixed(1)}%
          </span>
        </div>
        <div className="h-3 bg-secondary rounded-full overflow-hidden">
          <div 
            className={`h-full transition-all duration-1000 rounded-full ${
              results.churnRate > 30 ? "bg-destructive" : 
              results.churnRate > 15 ? "bg-warning" : 
              "bg-success"
            }`}
            style={{ width: `${Math.min(results.churnRate, 100)}%` }}
          />
        </div>
        <p className="text-sm text-muted-foreground mt-2">
          {results.churnRate > 30 
            ? "High churn risk detected. Immediate action recommended."
            : results.churnRate > 15
            ? "Moderate churn risk. Consider retention strategies."
            : "Low churn risk. Your customer base appears stable."}
        </p>
      </Card>

      {/* Download Section */}
      <Card className="p-6 bg-card border shadow-elevated">
        <div className="flex items-center gap-4 mb-4">
          <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center">
            <FileSpreadsheet className="w-6 h-6 text-primary" />
          </div>
          <div>
            <p className="font-medium text-foreground">Prediction Report</p>
            <p className="text-sm text-muted-foreground">
              {filename.replace('.csv', '_predictions.csv')}
            </p>
          </div>
        </div>
        <div className="flex gap-3">
          <Button 
            onClick={onDownload}
            className="flex-1 gradient-primary text-primary-foreground"
          >
            <Download className="w-4 h-4 mr-2" />
            Download Report
          </Button>
          <Button 
            onClick={onNewPrediction}
            variant="outline"
            className="flex-1"
          >
            New Prediction
          </Button>
        </div>
      </Card>
    </div>
  );
};

export default ResultsDownload;
