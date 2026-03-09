import { Upload, Cpu, Search, LayoutDashboard } from "lucide-react";

const steps = [
  {
    icon: Upload,
    title: "Upload Dataset",
    description:
      "Upload your customer CSV with behavioral and demographic data.",
    borderColor: "border-purple-500",
    glowColor: "shadow-purple-500/20",
  },
  {
    icon: Cpu,
    title: "ML Model Prediction",
    description:
      "ExtraTreesClassifier analyzes patterns across 18 features.",
    borderColor: "border-blue-500",
    glowColor: "shadow-blue-500/20",
  },
  {
    icon: Search,
    title: "SHAP Explanation Engine",
    description:
      "TreeExplainer computes feature contributions for each prediction.",
    borderColor: "border-teal-400",
    glowColor: "shadow-teal-400/20",
  },
  {
    icon: LayoutDashboard,
    title: "Interactive AI Dashboard",
    description:
      "Explore predictions, feature importance, and AI-generated insights.",
    borderColor: "border-emerald-400",
    glowColor: "shadow-emerald-400/20",
  },
];

const HowItWorks = () => {
  return (
    <section className="py-20 lg:py-28 bg-secondary/40">
      <div className="container mx-auto px-4">
        {/* Section Header */}
        <div className="text-center mb-20">
          <p className="text-sm font-semibold uppercase tracking-[0.2em] bg-gradient-to-r from-purple-400 via-blue-400 to-teal-400 bg-clip-text text-transparent mb-4">
            How It Works
          </p>
          <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold font-display text-foreground mb-5">
            From Data to Decisions in Seconds
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            A streamlined AI pipeline that transforms raw customer data into
            actionable retention intelligence.
          </p>
        </div>

        {/* Steps — horizontal connected layout */}
        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-4 gap-0">
            {steps.map((step, index) => (
              <div key={step.title} className="flex flex-col items-center text-center">
                {/* Icon box + connector row */}
                <div className="flex items-center w-full justify-center mb-10">
                  {/* Left connector */}
                  {index > 0 ? (
                    <div className="flex-1 h-0.5 bg-border" />
                  ) : (
                    <div className="flex-1" />
                  )}

                  {/* Step icon box */}
                  <div
                    className={`relative shrink-0 w-[110px] h-[110px] rounded-2xl border-2 ${step.borderColor} bg-card flex flex-col items-center justify-center shadow-lg ${step.glowColor} transition-all duration-300 hover:scale-105`}
                  >
                    <step.icon className="w-9 h-9 text-foreground/80" />
                    <span className="text-[11px] font-semibold text-muted-foreground mt-2 uppercase tracking-wider">
                      Step {index + 1}
                    </span>
                  </div>

                  {/* Right connector */}
                  {index < steps.length - 1 ? (
                    <div className="flex-1 h-0.5 bg-border" />
                  ) : (
                    <div className="flex-1" />
                  )}
                </div>

                {/* Title & description */}
                <div className="px-3">
                  <h3 className="text-base md:text-lg font-bold text-foreground mb-2 font-display">
                    {step.title}
                  </h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {step.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default HowItWorks;
