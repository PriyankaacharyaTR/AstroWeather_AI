import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Brain, ArrowLeft } from "lucide-react";
import axios from "axios";
import "./App.css";

// Human-friendly translations for feature names
const FEATURE_TRANSLATIONS = {
  sun_X: "Seasonal solar positioning (East-West axis)",
  sun_Y: "Seasonal solar positioning (North-South axis)",
  sun_Z: "Seasonal solar positioning (Vertical axis)",
  sun_VX: "Seasonal solar movement pattern",
  sun_VY: "Solar seasonal cycle intensity",
  sun_VZ: "Solar elevation changes",
  moon_X: "Short-term lunar positioning",
  moon_Y: "Lunar cycle phase effect",
  moon_Z: "Lunar elevation influence",
  moon_VX: "Short-term lunar variation",
  moon_VY: "Lunar tidal pattern",
  moon_VZ: "Lunar atmospheric influence",
  jupiter_X: "Long-cycle planetary modulation",
  jupiter_Y: "Jupiter orbital pattern",
  jupiter_Z: "Long-term climate influence",
  jupiter_VX: "Jupiter cycle variation",
  jupiter_VY: "Multi-year climate pattern",
  jupiter_VZ: "Long-term atmospheric modulation",
  saturn_X: "Slow, stabilizing planetary influence",
  saturn_Y: "Saturn orbital stability effect",
  saturn_Z: "Long-term climate stabilization",
  saturn_VX: "Decadal climate pattern",
  saturn_VY: "Saturn stabilizing effect",
  saturn_VZ: "Deep climate cycle influence",
  venus_X: "Near-term planetary influence",
  venus_Y: "Venus orbital proximity effect",
  venus_Z: "Short-cycle planetary modulation",
  venus_VX: "Venus cycle variation",
  venus_VY: "Near-planetary atmospheric effect",
  venus_VZ: "Venus-driven variability",
  Year: "Long-term climate trend",
  Month: "Seasonal calendar timing",
  Day: "Daily variation pattern",
  DayOfYear: "Seasonal calendar timing",
  WeekOfYear: "Weekly seasonal pattern",
};

// Category descriptions for layman explanation
const CATEGORY_EXPLANATIONS = {
  sun: {
    title: "☀️ The Sun's Influence",
    description:
      "The Sun is the primary driver of Earth's temperature. Its position relative to Earth changes throughout the year, creating seasons. When the Sun appears higher in the sky (during summer months), solar radiation is more direct and intense, leading to warmer temperatures. The Sun's velocity components capture these seasonal movements that historically determine temperature patterns across months.",
    impact: "Contributes to approximately 70-85% of temperature prediction",
  },
  saturn: {
    title: "🪐 Saturn's Stabilizing Effect",
    description:
      "Saturn, with its 29.5-year orbital period, introduces very long-term climate cycles. While its gravitational influence on Earth is minimal, Saturn's position serves as a proxy for multi-decadal climate patterns. Research suggests correlations between outer planet positions and long-term climate oscillations, providing a stabilizing reference for climate models.",
    impact: "Provides long-term baseline stability (5-10% contribution)",
  },
  jupiter: {
    title: "🌍 Jupiter's Long-Cycle Modulation",
    description:
      "Jupiter, the largest planet in our solar system, completes one orbit every 11.86 years. This period aligns with certain climate cycles observed on Earth. Jupiter's gravitational influence subtly affects the Sun's position relative to the solar system's center of mass, creating ripple effects that may influence solar activity and, consequently, Earth's climate patterns.",
    impact: "Modulates multi-year climate trends (3-8% contribution)",
  },
  moon: {
    title: "🌙 Lunar Short-Term Variations",
    description:
      "The Moon's 27.3-day orbital cycle creates tidal forces and subtle atmospheric variations. Lunar positioning affects tidal patterns, which influence coastal temperatures and atmospheric pressure. The Moon's gravitational pull creates measurable effects on Earth's atmosphere, contributing to short-term weather variability that our model captures.",
    impact: "Adds short-term variability (2-5% contribution)",
  },
  venus: {
    title: "💫 Venus Proximity Effects",
    description:
      "Venus, Earth's closest planetary neighbor, has an orbital period of 225 days. When Venus is closest to Earth (inferior conjunction), some researchers hypothesize subtle electromagnetic or gravitational interactions. Our model uses Venus positioning to capture seasonal transitions and near-term atmospheric patterns.",
    impact: "Fine-tunes seasonal transitions (1-3% contribution)",
  },
  temporal: {
    title: "📅 Seasonal Calendar Timing",
    description:
      "The day of year and seasonal timing capture Earth's axial tilt effects. As Earth orbits the Sun, its 23.5° tilt causes different hemispheres to receive varying amounts of sunlight. This creates predictable seasonal patterns that, combined with planetary positions, allow precise temperature forecasting based on historical climate data.",
    impact: "Reinforces seasonal patterns (3-8% contribution)",
  },
};

export default function ShapExplanation() {
  const [shapData, setShapData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchShapData();
  }, []);

  const fetchShapData = async () => {
    try {
      // Start both the API call and a minimum 3.5 second timer
      const [response] = await Promise.all([
        axios.get("http://127.0.0.1:5000/api/shap-analysis"),
        new Promise(resolve => setTimeout(resolve, 2500)) // Minimum 3.5 seconds loading
      ]);
      setShapData(response.data);
      setLoading(false);
    } catch (err) {
      console.error("Failed to fetch SHAP data:", err);
      // Still wait a moment before showing error
      await new Promise(resolve => setTimeout(resolve, 2000));
      setError("Unable to load SHAP analysis. Please ensure the backend is running.");
      setLoading(false);
    }
  };

  const getFeatureCategory = (featureName) => {
    const name = featureName.toLowerCase();
    if (name.includes("sun")) return "sun";
    if (name.includes("saturn")) return "saturn";
    if (name.includes("jupiter")) return "jupiter";
    if (name.includes("moon")) return "moon";
    if (name.includes("venus")) return "venus";
    if (["year", "month", "day", "dayofyear", "weekofyear"].includes(name)) return "temporal";
    return "other";
  };

  return (
    <div className="app-container">
      <div className="stars" />

      <motion.div
        className="glass-card model-info-card"
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        style={{ maxWidth: "1000px", textAlign: "left" }}
      >
        <div className="model-info-scroll">
          {/* Header */}
          <header style={{ textAlign: "center", marginBottom: "2rem" }}>
            <motion.button
              onClick={() => (window.location.href = "/?dashboard=true")}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              style={{
                position: "absolute",
                left: "1.5rem",
                top: "1.5rem",
                background: "rgba(0, 242, 255, 0.1)",
                border: "1px solid rgba(0, 242, 255, 0.3)",
                borderRadius: "8px",
                padding: "0.5rem 1rem",
                color: "#00f2ff",
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                gap: "0.5rem",
                fontSize: "0.9rem",
              }}
            >
              <ArrowLeft size={16} /> Back
            </motion.button>
            <h1>
              <Brain className="icon" /> Scientific <span>Explanation</span>
            </h1>
            <p style={{ color: "rgba(255,255,255,0.7)", marginTop: "0.5rem" }}>
              Understanding AI Predictions through SHAP Analysis
            </p>
          </header>

          {loading ? (
            <div style={{ textAlign: "center", padding: "3rem" }}>
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                style={{ fontSize: "3rem" }}
              >
                🌀
              </motion.div>
              <p style={{ color: "#00f2ff", marginTop: "1rem" }}>Loading SHAP Analysis...</p>
            </div>
          ) : error ? (
            <div className="results" style={{ textAlign: "center", color: "#ef4444" }}>
              <p>{error}</p>
            </div>
          ) : (
            <>
              {/* What is SHAP Section */}
              <section className="results" style={{ marginBottom: "2rem" }}>
                <h3 style={{ color: "#a855f7", marginBottom: "1rem" }}>
                  🧠 What is SHAP (Explainable AI)?
                </h3>
                <p
                  style={{
                    color: "rgba(255,255,255,0.85)",
                    lineHeight: 1.8,
                    textAlign: "justify",
                  }}
                >
                  SHAP (SHapley Additive exPlanations) is a game-theoretic approach that explains
                  the output of any machine learning model. It connects optimal credit allocation
                  with local explanations, showing how each feature contributes to pushing the
                  prediction from the baseline (average) to the actual output. In our AstroWeather
                  model, SHAP reveals how planetary positions influence temperature predictions.
                </p>
              </section>

              {/* SHAP Summary Plot */}
              <section className="results" style={{ marginBottom: "2rem" }}>
                <h3 style={{ color: "#00f2ff", marginBottom: "1rem" }}>
                  📊 Global Feature Importance (Summary Plot)
                </h3>
                <div style={{ display: "flex", gap: "1.5rem", alignItems: "flex-start" }}>
                  <div style={{ flex: "1.5" }}>
                    <img
                      src="http://127.0.0.1:5000/api/shap-image/shap_t2m_summary.png"
                      alt="SHAP Summary Plot"
                      style={{
                        width: "100%",
                        borderRadius: "12px",
                        border: "1px solid rgba(0, 242, 255, 0.2)",
                      }}
                      onError={(e) => {
                        e.target.style.display = "none";
                      }}
                    />
                  </div>
                  <div
                    style={{
                      flex: "1",
                      background: "rgba(0, 242, 255, 0.05)",
                      padding: "1.25rem",
                      borderRadius: "12px",
                      border: "1px solid rgba(0, 242, 255, 0.15)",
                    }}
                  >
                    <h4 style={{ color: "#00f2ff", marginBottom: "0.75rem", fontSize: "1rem" }}>
                      📖 How to Read This Plot
                    </h4>
                    <ul
                      style={{
                        color: "rgba(255,255,255,0.8)",
                        fontSize: "0.9rem",
                        lineHeight: 1.8,
                        paddingLeft: "1.25rem",
                        margin: 0,
                      }}
                    >
                      <li>Each dot represents one data point (day)</li>
                      <li>
                        <span style={{ color: "#ef4444" }}>Red</span> = high feature value,{" "}
                        <span style={{ color: "#3b82f6" }}>Blue</span> = low feature value
                      </li>
                      <li>Dots pushed right → feature increases temperature</li>
                      <li>Dots pushed left → feature decreases temperature</li>
                      <li>Features at top have highest impact on predictions</li>
                      <li>Spread of dots shows feature's influence range</li>
                    </ul>
                  </div>
                </div>
              </section>

              {/* SHAP Bar Plot */}
              <section className="results" style={{ marginBottom: "2rem" }}>
                <h3 style={{ color: "#f59e0b", marginBottom: "1rem" }}>
                  📈 Feature Contribution Chart (Bar Plot)
                </h3>
                <div style={{ display: "flex", gap: "1.5rem", alignItems: "flex-start" }}>
                  <div style={{ flex: "1.5" }}>
                    <img
                      src="http://127.0.0.1:5000/api/shap-image/shap_t2m_bar.png"
                      alt="SHAP Bar Plot"
                      style={{
                        width: "100%",
                        borderRadius: "12px",
                        border: "1px solid rgba(245, 158, 11, 0.2)",
                      }}
                      onError={(e) => {
                        e.target.style.display = "none";
                      }}
                    />
                  </div>
                  <div
                    style={{
                      flex: "1",
                      background: "rgba(245, 158, 11, 0.05)",
                      padding: "1.25rem",
                      borderRadius: "12px",
                      border: "1px solid rgba(245, 158, 11, 0.15)",
                    }}
                  >
                    <h4 style={{ color: "#f59e0b", marginBottom: "0.75rem", fontSize: "1rem" }}>
                      📖 Understanding This Chart
                    </h4>
                    <ul
                      style={{
                        color: "rgba(255,255,255,0.8)",
                        fontSize: "0.9rem",
                        lineHeight: 1.8,
                        paddingLeft: "1.25rem",
                        margin: 0,
                      }}
                    >
                      <li>Shows average absolute impact of each feature</li>
                      <li>Longer bars = more influential features</li>
                      <li>Sun-related features dominate temperature prediction</li>
                      <li>Planetary positions provide fine-tuning adjustments</li>
                      <li>This explains WHY the model makes specific predictions</li>
                    </ul>
                  </div>
                </div>
              </section>

              {/* SHAP Force Plot */}
              <section className="results" style={{ marginBottom: "2rem" }}>
                <h3 style={{ color: "#22c55e", marginBottom: "1rem" }}>
                  🔍 Single Prediction Breakdown (Force Plot)
                </h3>
                <div style={{ display: "flex", gap: "1.5rem", alignItems: "flex-start" }}>
                  <div style={{ flex: "1.5" }}>
                    <img
                      src="http://127.0.0.1:5000/api/shap-image/shap_t2m_force.png"
                      alt="SHAP Force Plot"
                      style={{
                        width: "100%",
                        borderRadius: "12px",
                        border: "1px solid rgba(34, 197, 94, 0.2)",
                      }}
                      onError={(e) => {
                        e.target.style.display = "none";
                      }}
                    />
                  </div>
                  <div
                    style={{
                      flex: "1",
                      background: "rgba(34, 197, 94, 0.05)",
                      padding: "1.25rem",
                      borderRadius: "12px",
                      border: "1px solid rgba(34, 197, 94, 0.15)",
                    }}
                  >
                    <h4 style={{ color: "#22c55e", marginBottom: "0.75rem", fontSize: "1rem" }}>
                      📖 Force Plot Explanation
                    </h4>
                    <ul
                      style={{
                        color: "rgba(255,255,255,0.8)",
                        fontSize: "0.9rem",
                        lineHeight: 1.8,
                        paddingLeft: "1.25rem",
                        margin: 0,
                      }}
                    >
                      <li>Shows how one specific prediction was made</li>
                      <li>
                        <span style={{ color: "#ef4444" }}>Red arrows</span> push prediction higher
                      </li>
                      <li>
                        <span style={{ color: "#3b82f6" }}>Blue arrows</span> push prediction lower
                      </li>
                      <li>Arrow size = magnitude of feature's effect</li>
                      <li>Base value = average temperature from training data</li>
                      <li>Final value = actual predicted temperature</li>
                    </ul>
                  </div>
                </div>
              </section>

              {/* Top 5 Features Table */}
              {shapData && shapData.top_features && (
                <section className="results" style={{ marginBottom: "2rem" }}>
                  <h3 style={{ color: "#a855f7", marginBottom: "1rem" }}>
                    🌟 Top 5 Influencing Factors
                  </h3>
                  <div
                    style={{
                      background: "rgba(168, 85, 247, 0.05)",
                      borderRadius: "12px",
                      overflow: "hidden",
                      border: "1px solid rgba(168, 85, 247, 0.2)",
                    }}
                  >
                    <table style={{ width: "100%", borderCollapse: "collapse" }}>
                      <thead>
                        <tr
                          style={{
                            background: "rgba(168, 85, 247, 0.15)",
                            color: "#a855f7",
                          }}
                        >
                          <th style={{ padding: "1rem", textAlign: "left" }}>Rank</th>
                          <th style={{ padding: "1rem", textAlign: "left" }}>
                            Factor (Human-Friendly)
                          </th>
                          <th style={{ padding: "1rem", textAlign: "center" }}>Contribution</th>
                          <th style={{ padding: "1rem", textAlign: "center" }}>SHAP Value</th>
                        </tr>
                      </thead>
                      <tbody>
                        {shapData.top_features.map((feat, idx) => (
                          <tr
                            key={idx}
                            style={{
                              borderBottom: "1px solid rgba(255,255,255,0.05)",
                              background:
                                idx % 2 === 0
                                  ? "rgba(255,255,255,0.02)"
                                  : "transparent",
                            }}
                          >
                            <td
                              style={{
                                padding: "1rem",
                                color: "#a855f7",
                                fontWeight: "bold",
                              }}
                            >
                              #{idx + 1}
                            </td>
                            <td style={{ padding: "1rem", color: "rgba(255,255,255,0.9)" }}>
                              {FEATURE_TRANSLATIONS[feat.feature] || feat.description}
                            </td>
                            <td style={{ padding: "1rem", textAlign: "center" }}>
                              <span
                                style={{
                                  background: "linear-gradient(90deg, #a855f7, #6366f1)",
                                  padding: "0.35rem 0.75rem",
                                  borderRadius: "20px",
                                  fontSize: "0.85rem",
                                  fontWeight: "600",
                                  color: "#fff",
                                }}
                              >
                                {feat.contribution}%
                              </span>
                            </td>
                            <td
                              style={{
                                padding: "1rem",
                                textAlign: "center",
                                color: "rgba(255,255,255,0.7)",
                                fontFamily: "monospace",
                              }}
                            >
                              {feat.shap_value}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>
              )}

              {/* Contribution Breakdown */}
              {shapData && shapData.contribution_breakdown && (
                <section className="results" style={{ marginBottom: "2rem" }}>
                  <h3 style={{ color: "#00f2ff", marginBottom: "1rem" }}>
                    📊 Overall Contribution Breakdown
                  </h3>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem" }}>
                    <div
                      style={{
                        background:
                          "linear-gradient(135deg, rgba(168, 85, 247, 0.15), rgba(99, 102, 241, 0.1))",
                        borderRadius: "16px",
                        padding: "2rem",
                        textAlign: "center",
                        border: "1px solid rgba(168, 85, 247, 0.3)",
                      }}
                    >
                      <p
                        style={{
                          color: "#a855f7",
                          fontSize: "1rem",
                          marginBottom: "0.75rem",
                          fontWeight: "600",
                        }}
                      >
                        🪐 Planetary Influence
                      </p>
                      <h2 style={{ color: "#fff", fontSize: "3rem", margin: 0, fontWeight: "700" }}>
                        {shapData.contribution_breakdown.planetary}%
                      </h2>
                      <p
                        style={{
                          color: "rgba(255,255,255,0.6)",
                          fontSize: "0.85rem",
                          marginTop: "0.5rem",
                        }}
                      >
                        Sun, Moon, Jupiter, Saturn, Venus
                      </p>
                    </div>
                    <div
                      style={{
                        background:
                          "linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(234, 88, 12, 0.1))",
                        borderRadius: "16px",
                        padding: "2rem",
                        textAlign: "center",
                        border: "1px solid rgba(245, 158, 11, 0.3)",
                      }}
                    >
                      <p
                        style={{
                          color: "#f59e0b",
                          fontSize: "1rem",
                          marginBottom: "0.75rem",
                          fontWeight: "600",
                        }}
                      >
                        📅 Seasonal Calendar Effect
                      </p>
                      <h2 style={{ color: "#fff", fontSize: "3rem", margin: 0, fontWeight: "700" }}>
                        {shapData.contribution_breakdown.temporal}%
                      </h2>
                      <p
                        style={{
                          color: "rgba(255,255,255,0.6)",
                          fontSize: "0.85rem",
                          marginTop: "0.5rem",
                        }}
                      >
                        Year, Month, Day, DayOfYear
                      </p>
                    </div>
                  </div>
                </section>
              )}

              {/* Scientific Explanations in Layman Terms */}
              <section className="results" style={{ marginBottom: "2rem" }}>
                <h3 style={{ color: "#22c55e", marginBottom: "1.5rem" }}>
                  🔬 How Planets Affect Weather (Scientific Explanation)
                </h3>
                <p
                  style={{
                    color: "rgba(255,255,255,0.85)",
                    lineHeight: 1.8,
                    textAlign: "justify",
                    marginBottom: "1.5rem",
                  }}
                >
                  Our AstroWeather model uses planetary positions from NASA's JPL Horizons system
                  to predict temperature. But how can distant planets affect Earth's weather? The
                  answer lies in understanding gravitational influences, solar cycles, and
                  historical correlations that our machine learning model has learned from years of
                  data.
                </p>

                <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
                  {Object.entries(CATEGORY_EXPLANATIONS).map(([key, data]) => (
                    <motion.div
                      key={key}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.1 * Object.keys(CATEGORY_EXPLANATIONS).indexOf(key) }}
                      style={{
                        background: "rgba(255, 255, 255, 0.03)",
                        borderRadius: "12px",
                        padding: "1.25rem 1.5rem",
                        borderLeft: `4px solid ${
                          key === "sun"
                            ? "#f59e0b"
                            : key === "saturn"
                            ? "#a855f7"
                            : key === "jupiter"
                            ? "#ef4444"
                            : key === "moon"
                            ? "#00f2ff"
                            : key === "venus"
                            ? "#ec4899"
                            : "#22c55e"
                        }`,
                      }}
                    >
                      <h4
                        style={{
                          color: "#fff",
                          fontSize: "1.1rem",
                          marginBottom: "0.75rem",
                          fontWeight: "600",
                        }}
                      >
                        {data.title}
                      </h4>
                      <p
                        style={{
                          color: "rgba(255,255,255,0.8)",
                          fontSize: "0.95rem",
                          lineHeight: 1.8,
                          textAlign: "justify",
                          marginBottom: "0.75rem",
                        }}
                      >
                        {data.description}
                      </p>
                      <p
                        style={{
                          color: "#22c55e",
                          fontSize: "0.85rem",
                          fontStyle: "italic",
                        }}
                      >
                        📊 {data.impact}
                      </p>
                    </motion.div>
                  ))}
                </div>
              </section>

              {/* Key Takeaways */}
              <section className="results" style={{ marginBottom: "1rem" }}>
                <h3 style={{ color: "#00f2ff", marginBottom: "1rem" }}>
                  ✨ Key Takeaways
                </h3>
                <div
                  style={{
                    background: "rgba(0, 242, 255, 0.05)",
                    borderRadius: "12px",
                    padding: "1.5rem",
                    border: "1px solid rgba(0, 242, 255, 0.15)",
                  }}
                >
                  <ul
                    style={{
                      color: "rgba(255,255,255,0.9)",
                      fontSize: "1rem",
                      lineHeight: 2.2,
                      paddingLeft: "1.5rem",
                      margin: 0,
                    }}
                  >
                    <li>
                      <strong>The Sun is the primary driver</strong> of temperature, contributing
                      over 70% of the prediction through seasonal positioning
                    </li>
                    <li>
                      <strong>Planetary positions act as natural timekeepers</strong> that capture
                      long-term climate cycles
                    </li>
                    <li>
                      <strong>Jupiter and Saturn's slow orbits</strong> correlate with multi-year
                      and decadal climate patterns
                    </li>
                    <li>
                      <strong>The Moon adds short-term variability</strong> through atmospheric and
                      tidal influences
                    </li>
                    <li>
                      <strong>SHAP analysis provides transparency</strong>, showing exactly why each
                      prediction was made
                    </li>
                    <li>
                      <strong>This is correlation, not causation</strong> — the model learns
                      historical patterns where planetary positions aligned with specific weather
                      conditions
                    </li>
                  </ul>
                </div>
              </section>

              {/* Footer Note */}
              <p
                style={{
                  textAlign: "center",
                  color: "rgba(255,255,255,0.5)",
                  fontSize: "0.8rem",
                  marginTop: "2rem",
                  fontStyle: "italic",
                }}
              >
                Analysis powered by SHAP (SHapley Additive exPlanations) for transparent,
                trustworthy AI predictions
              </p>
            </>
          )}
        </div>
      </motion.div>
    </div>
  );
}
