import { TrendingUp, Shield, Zap } from "lucide-react";

const HeroSection = () => {
  const features = [
    {
      icon: TrendingUp,
      title: "AI-Powered Predictions",
      description: "Advanced machine learning models analyze customer behavior patterns",
    },
    {
      icon: Shield,
      title: "Data Security",
      description: "Your customer data is processed securely and never stored",
    },
    {
      icon: Zap,
      title: "Instant Results",
      description: "Get actionable insights in seconds, not hours",
    },
  ];

  return (
    <section className="relative overflow-hidden">
      {/* Background Elements */}
      <div className="absolute inset-0 -z-10">
        <div className="absolute top-20 left-10 w-72 h-72 bg-primary/5 rounded-full blur-3xl" />
        <div className="absolute bottom-20 right-10 w-96 h-96 bg-accent/5 rounded-full blur-3xl" />
      </div>

      <div className="container mx-auto px-4 py-16 lg:py-24">
        <div className="max-w-4xl mx-auto text-center mb-16">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 text-primary text-sm font-medium mb-6 animate-fade-up">
            <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />
            Powered by Advanced AI
          </div>
          
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold font-display text-foreground mb-6 animate-fade-up" style={{ animationDelay: '0.1s' }}>
            Predict Customer Churn{" "}
            <span className="gradient-text">Before It Happens</span>
          </h1>
          
          <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto animate-fade-up" style={{ animationDelay: '0.2s' }}>
            Identify at-risk customers with AI-powered analytics. Upload your customer data 
            and receive instant predictions to improve retention rates.
          </p>
        </div>

        {/* How it Works */}
        <div className="max-w-4xl mx-auto mb-16">
          <h2 className="text-center text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-8">
            How it Works
          </h2>
          <div className="flex flex-col md:flex-row items-center justify-center gap-4 md:gap-0">
            {[
              { step: "1", title: "Upload CSV", desc: "Upload your customer data file" },
              { step: "2", title: "AI Analysis", desc: "Our model analyzes patterns" },
              { step: "3", title: "Get Results", desc: "Download prediction report" },
            ].map((item, index) => (
              <div key={item.step} className="flex items-center">
                <div className="flex flex-col items-center text-center px-6 animate-fade-up" style={{ animationDelay: `${0.3 + index * 0.1}s` }}>
                  <div className="w-14 h-14 rounded-2xl gradient-primary flex items-center justify-center text-primary-foreground font-bold text-xl mb-3 shadow-elevated">
                    {item.step}
                  </div>
                  <h3 className="font-semibold text-foreground mb-1">{item.title}</h3>
                  <p className="text-sm text-muted-foreground">{item.desc}</p>
                </div>
                {index < 2 && (
                  <div className="hidden md:block w-16 h-0.5 bg-gradient-to-r from-primary/20 to-primary/40" />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Features */}
        <div className="grid md:grid-cols-3 gap-6 max-w-4xl mx-auto">
          {features.map((feature, index) => (
            <div 
              key={feature.title}
              className="p-6 rounded-2xl bg-card border shadow-elevated hover:shadow-lg transition-all duration-300 animate-fade-up"
              style={{ animationDelay: `${0.5 + index * 0.1}s` }}
            >
              <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center mb-4">
                <feature.icon className="w-6 h-6 text-primary" />
              </div>
              <h3 className="font-semibold text-foreground mb-2">{feature.title}</h3>
              <p className="text-sm text-muted-foreground">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
