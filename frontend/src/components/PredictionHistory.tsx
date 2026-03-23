import { useState, useEffect } from "react";
import { History, Download, Trash2, ChevronDown, ChevronUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";

export interface HistoryItem {
  id: string;
  filename: string;
  date: string;
  recordsProcessed: number;
  churnRate: number;
  data: Array<Record<string, unknown>>;
}

interface PredictionHistoryProps {
  history: HistoryItem[];
  onDownload: (item: HistoryItem) => void;
  onDelete: (id: string) => void;
}

const PredictionHistory = ({ history, onDownload, onDelete }: PredictionHistoryProps) => {
  const [isExpanded, setIsExpanded] = useState(false);

  if (history.length === 0) return null;

  const formatDate = (dateString: string): string => {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    }).format(date);
  };

  return (
    <Card className="w-full max-w-4xl mx-auto bg-card border shadow-elevated overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full p-4 flex items-center justify-between hover:bg-secondary/50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-secondary flex items-center justify-center">
            <History className="w-5 h-5 text-muted-foreground" />
          </div>
          <div className="text-left">
            <h3 className="font-semibold text-foreground">Prediction History</h3>
            <p className="text-sm text-muted-foreground">
              {history.length} previous {history.length === 1 ? 'analysis' : 'analyses'}
            </p>
          </div>
        </div>
        {isExpanded ? (
          <ChevronUp className="w-5 h-5 text-muted-foreground" />
        ) : (
          <ChevronDown className="w-5 h-5 text-muted-foreground" />
        )}
      </button>

      <div className={cn(
        "overflow-hidden transition-all duration-300",
        isExpanded ? "max-h-[500px]" : "max-h-0"
      )}>
        <div className="border-t overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow className="bg-secondary/50">
                <TableHead>Date</TableHead>
                <TableHead>Filename</TableHead>
                <TableHead className="text-center">Records</TableHead>
                <TableHead className="text-center">Churn Rate</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {history.map((item) => (
                <TableRow key={item.id} className="hover:bg-secondary/30">
                  <TableCell className="text-sm text-muted-foreground">
                    {formatDate(item.date)}
                  </TableCell>
                  <TableCell className="font-medium text-foreground">
                    {item.filename}
                  </TableCell>
                  <TableCell className="text-center">
                    {item.recordsProcessed.toLocaleString()}
                  </TableCell>
                  <TableCell className="text-center">
                    <span className={cn(
                      "inline-flex items-center px-2 py-1 rounded-full text-xs font-medium",
                      item.churnRate > 30 
                        ? "bg-destructive/10 text-destructive"
                        : item.churnRate > 15
                        ? "bg-warning/10 text-warning"
                        : "bg-success/10 text-success"
                    )}>
                      {item.churnRate.toFixed(1)}%
                    </span>
                  </TableCell>
                  <TableCell className="text-right">
                    <div className="flex items-center justify-end gap-2">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => onDownload(item)}
                        className="h-8 w-8 p-0"
                      >
                        <Download className="w-4 h-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => onDelete(item.id)}
                        className="h-8 w-8 p-0 hover:text-destructive"
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </div>
    </Card>
  );
};

export default PredictionHistory;
