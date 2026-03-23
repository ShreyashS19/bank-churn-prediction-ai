import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Navigate, Route, Routes, useLocation } from "react-router-dom";
import Index from "./pages/Index";
import ModelInsights from "./pages/ModelInsights";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();
const ANALYSIS_STATUS_KEY = "churnAnalysisStatus";

const canAccessModelInsights = (sessionId: string | null) => {
  if (!sessionId) return false;

  try {
    const statusRaw = sessionStorage.getItem(ANALYSIS_STATUS_KEY);
    if (statusRaw) {
      const parsed = JSON.parse(statusRaw) as {
        status?: "idle" | "processing" | "completed";
        sessionId?: string;
      };

      // Block access unless the same session reached completed state.
      if (parsed.status !== "completed" || parsed.sessionId !== sessionId) {
        return false;
      }
      return true;
    }

    // Backward compatibility for existing persisted completed state.
    const persisted = sessionStorage.getItem("churnPredictState");
    if (!persisted) return false;

    const parsedPersisted = JSON.parse(persisted) as { sessionId?: string };
    return parsedPersisted.sessionId === sessionId;
  } catch {
    return false;
  }
};

const GuardedModelInsightsRoute = () => {
  const location = useLocation();
  const search = new URLSearchParams(location.search);
  const sessionId = search.get("session");

  if (!canAccessModelInsights(sessionId)) {
    return <Navigate to="/" replace />;
  }

  return <ModelInsights />;
};

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Index />} />
          <Route path="/model-insights" element={<GuardedModelInsightsRoute />} />
          {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
