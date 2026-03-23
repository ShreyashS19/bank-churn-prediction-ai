import { Loader2, CheckCircle2, XCircle } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

export type ProcessingState = "idle" | "uploading" | "processing" | "completed" | "error";

interface ProcessingStatusProps {
  state: ProcessingState;
  progress: number;
  message?: string;
  recordsProcessed?: number;
}

const ProcessingStatus = ({ 
  state, 
  progress, 
  message,
  recordsProcessed 
}: ProcessingStatusProps) => {
  if (state === "idle") return null;

  const statusConfig = {
    uploading: {
      icon: Loader2,
      iconClass: "text-primary animate-spin",
      title: "Uploading file...",
      bgClass: "bg-primary/5 border-primary/20",
    },
    processing: {
      icon: Loader2,
      iconClass: "text-primary animate-spin",
      title: "Analyzing customer data...",
      bgClass: "bg-primary/5 border-primary/20",
    },
    completed: {
      icon: CheckCircle2,
      iconClass: "text-success",
      title: "Analysis Complete!",
      bgClass: "bg-success/5 border-success/20",
    },
    error: {
      icon: XCircle,
      iconClass: "text-destructive",
      title: "Analysis Failed",
      bgClass: "bg-destructive/5 border-destructive/20",
    },
  };

  const config = statusConfig[state as keyof typeof statusConfig];
  if (!config) return null;

  const Icon = config.icon;

  const processingMessages = [
    "Reading customer data...",
    "Extracting features...",
    "Running churn prediction model...",
    "Generating risk scores...",
    "Preparing report...",
  ];

  const currentMessage = message || (
    state === "processing" 
      ? processingMessages[Math.min(Math.floor(progress / 25), processingMessages.length - 1)]
      : config.title
  );

  return (
    <div className={cn(
      "w-full max-w-2xl mx-auto p-6 rounded-xl border animate-fade-up",
      config.bgClass
    )}>
      <div className="flex items-start gap-4">
        <div className="flex-shrink-0">
          <Icon className={cn("w-8 h-8", config.iconClass)} />
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="text-lg font-semibold text-foreground mb-1">
            {config.title}
          </h3>
          <p className="text-sm text-muted-foreground mb-4">
            {currentMessage}
          </p>
          
          {(state === "uploading" || state === "processing") && (
            <div className="space-y-2">
              <Progress value={progress} className="h-2" />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>{Math.round(progress)}% complete</span>
                {recordsProcessed && (
                  <span>{recordsProcessed.toLocaleString()} records processed</span>
                )}
              </div>
            </div>
          )}

          {state === "completed" && recordsProcessed && (
            <p className="text-sm text-success font-medium">
              Successfully analyzed {recordsProcessed.toLocaleString()} customer records
            </p>
          )}

          {state === "error" && (
            <p className="text-sm text-destructive">
              {message || "An error occurred while processing your file. Please try again."}
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

export default ProcessingStatus;
