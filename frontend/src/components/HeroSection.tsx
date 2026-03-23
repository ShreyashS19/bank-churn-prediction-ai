import { Upload } from "lucide-react";

interface HeroSectionProps {
  onUploadClick?: () => void;
}

const HeroSection = ({ onUploadClick }: HeroSectionProps) => {
  return (
    <section className="relative overflow-hidden">
      {/* Background Elements */}
      <div className="absolute inset-0 -z-10">
        <div className="absolute top-20 left-10 w-72 h-72 bg-primary/5 rounded-full blur-3xl" />
        <div className="absolute bottom-20 right-10 w-96 h-96 bg-accent/5 rounded-full blur-3xl" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-primary/3 rounded-full blur-[100px]" />
      </div>

      <div className="container mx-auto px-4 py-20 lg:py-32">
        <div className="max-w-4xl mx-auto text-center">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-primary/10 text-primary text-sm font-medium mb-8 animate-fade-up border border-primary/20">
            <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />
            Powered by ExtraTreesClassifier + SHAP
          </div>

          {/* Main Heading */}
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold font-display text-foreground mb-6 animate-fade-up leading-tight" style={{ animationDelay: '0.1s' }}>
            AI-Powered Customer{" "}
            <span className="gradient-text">Churn Prediction</span>
          </h1>

          {/* Subtext */}
          <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto mb-10 animate-fade-up leading-relaxed" style={{ animationDelay: '0.2s' }}>
            Predict customer churn using machine learning and uncover the exact reasons
            behind every prediction through Explainable AI.
          </p>

          {/* Button */}
          <div className="animate-fade-up" style={{ animationDelay: '0.3s' }}>
            <button
              onClick={onUploadClick}
              className="inline-flex items-center gap-2 px-8 py-4 rounded-xl gradient-primary text-primary-foreground font-semibold text-base shadow-elevated hover:shadow-lg hover:scale-[1.02] transition-all duration-300"
            >
              <Upload className="w-5 h-5" />
              Upload Customer Dataset
            </button>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
