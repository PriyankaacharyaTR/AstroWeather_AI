import { motion } from "framer-motion";
import { Brain } from "lucide-react";
import "./App.css";

export default function ModelInfo() {
  const performance = {
    mae: 0.299,
    r2: 0.956,
    csv: "temperature_prediction_results.csv",
    model: "weather_planet_model.pkl",
  };

  const featureImportance = [
    { feature: "sun_VX", importance: 0.551912 },
    { feature: "sun_Y", importance: 0.17212 },
    { feature: "sun_X", importance: 0.039046 },
    { feature: "sun_VY", importance: 0.038021 },
    { feature: "sun_Z", importance: 0.032938 },
    { feature: "jupiter_VY", importance: 0.020318 },
    { feature: "jupiter_VX", importance: 0.015818 },
    { feature: "jupiter_Z", importance: 0.014428 },
    { feature: "jupiter_X", importance: 0.012337 },
    { feature: "jupiter_VZ", importance: 0.011584 },
    { feature: "venus_VZ", importance: 0.011254 },
    { feature: "jupiter_Y", importance: 0.009463 },
    { feature: "venus_Y", importance: 0.009455 },
    { feature: "venus_VY", importance: 0.009377 },
    { feature: "venus_X", importance: 0.009047 },
  ];

  const modelComparison = [
    { modelType: "Your Planetary Model", mae: "0.299", r2: "0.956", inputFeatures: "Sun/Jupiter/Venus positions (NASA Horizons)", reference: "User data" },
    { modelType: "Linear Regression", mae: "2.50", r2: "0.82", inputFeatures: "Weather variables", reference: "https://www.cimachinelearning.com/analysis-pca-based-adaboost-machine-learning-model-for-predict-mid-term-weather.php" },
    { modelType: "LSTM/Deep Learning", mae: "1.90", r2: "0.89", inputFeatures: "Time-series weather data", reference: "https://www.ijrtmr.com/archiver/archives/weather_forecasting_using_machine_learning.pdf" },
    { modelType: "Random Forest", mae: "0.78", r2: "0.93", inputFeatures: "Meteorological parameters", reference: "https://journalijecc.com/index.php/IJECC/article/view/3829" },
    { modelType: "XGBoost", mae: "1.544", r2: "0.947", inputFeatures: "PV environment variables", reference: "https://www.nature.com/articles/s41598-025-98607-7" },
    { modelType: "AdaBoost (PCA)", mae: "0.398", r2: "0.992", inputFeatures: "Processed weather attributes", reference: "https://www.cimachinelearning.com/analysis-pca-based-adaboost-machine-learning-model-for-predict-mid-term-weather.php" },
  ];

  const samplePredictions = [
    {
      Date: "2020-10-09",
      Actual_T2M: 28.0,
      Predicted_T2M: 28.64115,
      Absolute_Error: 0.64115,
      Percent_Error: 2.289821,
    },
    {
      Date: "2021-04-24",
      Actual_T2M: 29.71,
      Predicted_T2M: 29.92685,
      Absolute_Error: 0.21685,
      Percent_Error: 0.729889,
    },
    {
      Date: "2022-11-30",
      Actual_T2M: 25.97,
      Predicted_T2M: 26.66425,
      Absolute_Error: 0.69425,
      Percent_Error: 2.673277,
    },
    {
      Date: "2023-06-09",
      Actual_T2M: 30.83,
      Predicted_T2M: 30.3015,
      Absolute_Error: 0.5285,
      Percent_Error: 1.714239,
    },
    {
      Date: "2021-05-22",
      Actual_T2M: 29.86,
      Predicted_T2M: 30.1208,
      Absolute_Error: 0.2608,
      Percent_Error: 0.873409,
    },
    {
      Date: "2021-05-10",
      Actual_T2M: 29.99,
      Predicted_T2M: 30.08125,
      Absolute_Error: 0.09125,
      Percent_Error: 0.304268,
    },
    {
      Date: "2021-02-20",
      Actual_T2M: 26.68,
      Predicted_T2M: 26.945,
      Absolute_Error: 0.265,
      Percent_Error: 0.993253,
    },
    {
      Date: "2025-07-04",
      Actual_T2M: 30.93,
      Predicted_T2M: 31.0077,
      Absolute_Error: 0.0777,
      Percent_Error: 0.251212,
    },
    {
      Date: "2022-12-07",
      Actual_T2M: 24.98,
      Predicted_T2M: 25.4297,
      Absolute_Error: 0.4497,
      Percent_Error: 1.80024,
    },
    {
      Date: "2023-02-25",
      Actual_T2M: 26.26,
      Predicted_T2M: 26.6397,
      Absolute_Error: 0.3797,
      Percent_Error: 1.445925,
    },
    {
      Date: "2025-06-11",
      Actual_T2M: 30.04,
      Predicted_T2M: 30.3598,
      Absolute_Error: 0.3198,
      Percent_Error: 1.064581,
    },
    {
      Date: "2020-04-06",
      Actual_T2M: 29.94,
      Predicted_T2M: 29.72955,
      Absolute_Error: 0.21045,
      Percent_Error: 0.702906,
    },
  ];

  return (
    <div className="app-container">
      <div className="stars" />

      <motion.div
        className="glass-card model-info-card"
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        style={{ maxWidth: "900px", textAlign: "left" }}
      >
        <div className="model-info-scroll">
        <header style={{ textAlign: "center" }}>
          <h1>
            <Brain className="icon" /> Model <span>Information</span>
          </h1>
          <p>Neural ODE–driven planetary weather intelligence</p>
        </header>

        <section className="results">
          <h3>AI Weather Patterns through Planetary Positions</h3>
        </section>

        <section className="results">

          <h4>Top Features (Importance)</h4>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  <th style={{ textAlign: "left", padding: "8px", borderBottom: "1px solid rgba(255,255,255,0.2)" }}>Feature</th>
                  <th style={{ textAlign: "right", padding: "8px", borderBottom: "1px solid rgba(255,255,255,0.2)" }}>Importance</th>
                </tr>
              </thead>
              <tbody>
                {featureImportance.map((fi) => (
                  <tr key={fi.feature}>
                    <td style={{ padding: "8px", borderBottom: "1px solid rgba(255,255,255,0.1)" }}>{fi.feature}</td>
                    <td style={{ padding: "8px", textAlign: "right", borderBottom: "1px solid rgba(255,255,255,0.1)" }}>{fi.importance}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h4 style={{ marginTop: "1rem" }}>Sample Predictions (first 12)</h4>
          <p style={{ marginTop: 0 }}>Overall MAE: {performance.mae} | Overall R²: {performance.r2}</p>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  <th style={{ textAlign: "left", padding: "8px", borderBottom: "1px solid rgba(255,255,255,0.2)" }}>Date</th>
                  <th style={{ textAlign: "right", padding: "8px", borderBottom: "1px solid rgba(255,255,255,0.2)" }}>Actual (°C)</th>
                  <th style={{ textAlign: "right", padding: "8px", borderBottom: "1px solid rgba(255,255,255,0.2)" }}>Predicted (°C)</th>
                  <th style={{ textAlign: "right", padding: "8px", borderBottom: "1px solid rgba(255,255,255,0.2)" }}>Absolute Error</th>
                  <th style={{ textAlign: "right", padding: "8px", borderBottom: "1px solid rgba(255,255,255,0.2)" }}>Percent Error (%)</th>
                </tr>
              </thead>
              <tbody>
                {samplePredictions.map((p, idx) => (
                  <tr key={`pred-${idx}`}>
                    <td style={{ padding: "8px", borderBottom: "1px solid rgba(255,255,255,0.1)" }}>{p.Date}</td>
                    <td style={{ padding: "8px", textAlign: "right", borderBottom: "1px solid rgba(255,255,255,0.1)" }}>{p.Actual_T2M}</td>
                    <td style={{ padding: "8px", textAlign: "right", borderBottom: "1px solid rgba(255,255,255,0.1)" }}>{p.Predicted_T2M}</td>
                    <td style={{ padding: "8px", textAlign: "right", borderBottom: "1px solid rgba(255,255,255,0.1)" }}>{p.Absolute_Error}</td>
                    <td style={{ padding: "8px", textAlign: "right", borderBottom: "1px solid rgba(255,255,255,0.1)" }}>{p.Percent_Error}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="results">
          <h3>Model Comparison & Benchmarking</h3>
          <p style={{ marginTop: 0, marginBottom: "0.5rem" }}>Performance comparison of your planetary model against baseline ML approaches.</p>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  <th style={{ textAlign: "left", padding: "8px", borderBottom: "1px solid rgba(255,255,255,0.2)" }}>Model Type</th>
                  <th style={{ textAlign: "right", padding: "8px", borderBottom: "1px solid rgba(255,255,255,0.2)" }}>MAE (°C)</th>
                  <th style={{ textAlign: "right", padding: "8px", borderBottom: "1px solid rgba(255,255,255,0.2)" }}>R²</th>
                  <th style={{ textAlign: "left", padding: "8px", borderBottom: "1px solid rgba(255,255,255,0.2)" }}>Input Features</th>
                  <th style={{ textAlign: "left", padding: "8px", borderBottom: "1px solid rgba(255,255,255,0.2)" }}>Reference</th>
                </tr>
              </thead>
              <tbody>
                {modelComparison.map((model, idx) => (
                  <tr key={`model-${idx}`}>
                    <td style={{ padding: "8px", borderBottom: "1px solid rgba(255,255,255,0.1)" }}><strong>{model.modelType}</strong></td>
                    <td style={{ padding: "8px", textAlign: "right", borderBottom: "1px solid rgba(255,255,255,0.1)" }}>{model.mae}</td>
                    <td style={{ padding: "8px", textAlign: "right", borderBottom: "1px solid rgba(255,255,255,0.1)" }}>{model.r2}</td>
                    <td style={{ padding: "8px", borderBottom: "1px solid rgba(255,255,255,0.1)", fontSize: "0.9rem" }}>{model.inputFeatures}</td>
                    <td style={{ padding: "8px", borderBottom: "1px solid rgba(255,255,255,0.1)" }}>{model.reference.startsWith("http") ? <a href={model.reference} target="_blank" rel="noopener noreferrer" style={{ color: "#64b5f6", textDecoration: "underline", fontSize: "0.85rem" }}>View Paper</a> : model.reference}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="results">
          <h3>Visual Gallery</h3>
          <p style={{ marginTop: 0, marginBottom: "1rem" }}>Model visualizations and analysis plots.</p>
          <div className="visual-gallery">
            <div className="gallery-item">
              <div className="gallery-image-wrapper">
                <img src="/plot_actual_vs_predicted.png" alt="Actual vs Predicted Temperature" />
              </div>
              <div className="gallery-caption">Actual vs Predicted</div>
            </div>

            <div className="gallery-item">
              <div className="gallery-image-wrapper">
                <img src="/plot_feature_importance.png" alt="Feature Importance Analysis" />
              </div>
              <div className="gallery-caption">Feature Importance</div>
            </div>

            <div className="gallery-item">
              <div className="gallery-image-wrapper">
                <img src="/plot_residuals.png" alt="Residuals Distribution" />
              </div>
              <div className="gallery-caption">Residuals Plot</div>
            </div>

            <div className="gallery-item">
              <div className="gallery-image-wrapper">
                <img src="/plot_time_series.png" alt="Time Series Prediction" />
              </div>
              <div className="gallery-caption">Time Series</div>
            </div>
          </div>
        </section>

        <section className="results">
          <h3>1. Ancient Indian Methods of Planetary Weather Prediction</h3>
          <h4>Conceptual Foundation</h4>
          <p>
            Ancient Indian scholars did not view planets as direct causal agents of weather. Instead, planetary positions were treated as high-precision temporal and cyclic markers that encoded seasonal, solar, lunar, and long-term climatic rhythms.
          </p>
          <p>
            The fundamental assumption was: Stable celestial cycles reflect stable environmental cycles on Earth.
          </p>
          <p>
            Thus, planetary motion acted as a time-indexing system, not a force-based mechanism.
          </p>
          <h4>Role of Planetary Positions</h4>
          <p>
            Each celestial body was associated with a distinct temporal scale:
          </p>
          <ul>
            <li><strong>Sun (Surya)</strong>: Encodes the annual seasonal cycle (ṛtu system). Solar longitude and declination were used to determine seasonal transitions and agricultural timing.</li>
            <li><strong>Moon (Chandra)</strong>: Governs short-term variability through lunar phases and tidal influences. The Moon’s path through 27 Nakshatras (lunar mansions) provided fine-grained temporal bins.</li>
            <li><strong>Jupiter (Guru/Bṛhaspati)</strong>: Represents long-term climatic memory (~12-year cycle). Jupiter’s position was often associated with abundance, drought cycles, and multi-year rainfall tendencies.</li>
            <li><strong>Venus (Shukra)</strong>: Linked to cloud formation, atmospheric moisture, and storm precursors, operating on an ~8-year cycle.</li>
          </ul>
          <h4>Stars, Nakshatras, and Binning</h4>
          <p>
            Instead of continuous angular measurements, ancient systems discretized the sky:
          </p>
          <ul>
            <li>The ecliptic was divided into 27 Nakshatras</li>
            <li>Each Nakshatra acted as a temporal bin</li>
            <li>Observations were aggregated per bin across generations</li>
          </ul>
          <p>
            This is conceptually similar to: histogram-based feature encoding, categorical time embeddings, cyclic positional encoding in modern ML.
          </p>
          <h4>Mathematical Functions Used (Ancient Perspective)</h4>
          <p>
            Ancient Indian astronomy relied heavily on periodic mathematics, including:
          </p>
          <ul>
            <li>Sine and cosine approximations (jya / kojya)</li>
            <li>Circular motion equations</li>
            <li>Modular arithmetic over cycles</li>
            <li>Interpolation tables for angular motion</li>
          </ul>
          <p>
            In modern terms, this maps directly to: θ(t) = 2π ⋅ (t / T) + ϕ
          </p>
          <p>
            Which is exactly how Fourier features and cyclic encodings are used in machine learning today.
          </p>
        </section>

        <section className="results">
          <h3>2. Manuscripts and Textual Sources</h3>
          <p>
            This project is grounded in primary Indian astronomical and meteorological texts, treated as knowledge sources, not literal predictors.
          </p>
          <h4>Core Manuscripts</h4>
          <ul>
            <li><strong>Surya Siddhanta</strong>: A foundational astronomical treatise describing planetary periods, solar and lunar motion, trigonometric formulations. Used in this project to justify cyclic modeling of planetary motion.</li>
            <li><strong>Brihat Samhita (Varāhamihira)</strong>: A systematic compendium linking planetary configurations, cloud behavior, winds, rainfall, and seasonal anomalies. Used to identify which planetary cycles historically mattered for weather interpretation.</li>
            <li><strong>Aryabhatiya (Aryabhata)</strong>: Introduced mathematical astronomy, time reckoning, modular cyclic calculations. Inspired the continuous-time mathematical framing adopted in this project.</li>
          </ul>
          <h4>How Manuscripts Are Used Here</h4>
          <ul>
            <li>Not as rule-books</li>
            <li>Not as deterministic predictors</li>
            <li>But as feature-selection wisdom accumulated through centuries of observation</li>
          </ul>
        </section>

        <section className="results">
          <h3>3. Neural ODE as the Core Planetary Model</h3>
          <h4>Why Neural ODEs?</h4>
          <p>
            Planetary motion is: continuous, smooth, governed by stable dynamics, queryable at arbitrary time points.
          </p>
          <p>
            Neural Ordinary Differential Equations (Neural ODEs) are the most natural AI model for such systems.
          </p>
          <p>
            Unlike RNNs or LSTMs, Neural ODEs learn: ds/dt = fθ(s, t)
          </p>
          <p>
            Where: s is the planetary state (position, velocity), fθ is a neural network approximating system dynamics.
          </p>
          <h4>Model Details</h4>
          <ul>
            <li><strong>Input State</strong>: Planetary position and velocity vectors, Scalar time input</li>
            <li><strong>Neural Dynamics Function</strong>: Fully connected neural network, Smooth activations (Tanh), Outputs time derivatives of state</li>
            <li><strong>Integration</strong>: Runge–Kutta numerical solver, Continuous-time prediction, Differentiable end-to-end</li>
          </ul>
          <h4>Advantages</h4>
          <ul>
            <li>No hard-coded physics equations</li>
            <li>No satellite dependency at inference</li>
            <li>Arbitrary time resolution</li>
            <li>Stable long-term motion</li>
          </ul>
          <p>
            This mirrors how ancient astronomers analytically solved cyclic motion — but using data-driven learning.
          </p>
        </section>

        <section className="results">
          <h3>4. Knowledge-Grounded Chatbot (RAG)</h3>
          <h4>Purpose</h4>
          <p>
            To allow users to query Indian astronomical knowledge alongside modern interpretations.
          </p>
          <h4>RAG Architecture</h4>
          <p>
            Manuscripts (OCR Text) → Text Chunking → Embeddings → FAISS Vector Index → Retriever → LLM (Grok API) → Grounded Answer
          </p>
          <h4>Key Components</h4>
          <ul>
            <li><strong>OCR Pipeline</strong>: Manuscripts scanned as PDFs/images, OCR extracts raw text, Manual cleaning + semantic chunking</li>
            <li><strong>FAISS</strong>: High-performance vector similarity search, Enables semantic retrieval across manuscripts</li>
            <li><strong>Embedding Model</strong>: Converts text chunks into dense vectors, Preserves semantic meaning</li>
            <li><strong>LLM via Grok API</strong>: Used only for answer synthesis, Never answers without retrieved context, Prevents hallucination</li>
          </ul>
          <h4>Example Queries</h4>
          <ul>
            <li>"How did Jupiter’s cycle relate to rainfall?"</li>
            <li>"What is a Nakshatra in weather context?"</li>
            <li>"How did ancient scholars model periodicity?"</li>
          </ul>
        </section>

        <section className="results">
          <h3>5. Datasets Used</h3>
          <ul>
            <li><strong>NASA JPL Horizons</strong>: Planetary Ephemeris, High-precision planetary position data, Used to train Neural ODE dynamics, Public domain, Covers decades of motion</li>
            <li><strong>NASA POWER</strong>: Global Climate Variables, Temperature, Precipitation, Solar radiation, Wind speed, Used as ground-truth climate labels and auxiliary features</li>
            <li><strong>TimeAndDate</strong>: Historic City-Level Weather, Daily temperature, Humidity and conditions, Location-specific patterns (e.g., Bengaluru), Used to validate local-scale weather patterns</li>
          </ul>
        </section>

        <section className="results">
          <h3>6. Model Performance & Evaluation</h3>
          <h4>Planetary Motion Model (Neural ODE)</h4>
          <ul>
            <li>Mean Position Error: very low relative to orbital radius</li>
            <li>Smooth long-term stability</li>
            <li>No divergence over extended horizons</li>
          </ul>
          <h4>Weather Prediction Model</h4>
          <ul>
            <li>Regression (Rainfall / Temperature): Mean Absolute Error (MAE): reported per task, Root Mean Squared Error (RMSE): lower with planetary features</li>
            <li>Classification (Monsoon Onset): Confusion Matrix shows improved separation of Early / Normal / Late classes, Reduced false transitions</li>
          </ul>
          <p>
            IKS-inspired planetary features consistently reduce long-horizon variance.
          </p>
          <h4>Key Takeaway</h4>
          <p>
            Ancient Indian astronomy functioned as a cyclic time-encoding system. This project reinterprets that insight using Neural ODEs, multi-task learning, and retrieval-augmented AI, showing how traditional knowledge can inform modern scientific modeling without sacrificing rigor.
          </p>
        </section>
        </div>
        <motion.div style={{ textAlign: "center", marginTop: "2rem" }}>
          <button
            className="mission-button"
            onClick={() => window.history.back()}
          >
            ← Back to Dashboard
          </button>
        </motion.div>
      </motion.div>
    </div>
  );
}
