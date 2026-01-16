import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Thermometer, Orbit, Gauge, Droplets, Wind, Cloud } from 'lucide-react';
import axios from 'axios';
import './App.css';
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import AskMore from "./AskMore";
import ModelInfo from "./ModelInfo";

// Parameter mapping with friendly names, units, and icons
const PARAMETER_INFO = {
  T2M: { name: 'Air Temperature', unit: '°C', icon: Thermometer },
  PS: { name: 'Atm Pressure', unit: 'kPa', icon: Gauge },
  QV2M: { name: 'Specific Humidity', unit: 'g/kg', icon: Droplets },
  GWETTOP: { name: 'Top Soil Wetness', unit: '', icon: Cloud },
  WS2M: { name: 'Wind Speed', unit: 'm/s', icon: Wind },
};


// --- SUB-COMPONENT: Sequential Letter Animation ---
const WelcomeText = ({ text }) => {
  const letters = Array.from(text);
  const container = {
    hidden: { opacity: 0 },
    visible: (i = 1) => ({
      opacity: 1,
      transition: { staggerChildren: 0.08, delayChildren: 0.5 },
    }),
  };
  const child = {
    visible: { opacity: 1, y: 0, transition: { type: "spring", damping: 12, stiffness: 100 } },
    hidden: { opacity: 0, y: 20 },
  };

  return (
    <motion.h1 className="welcome-title" variants={container} initial="hidden" animate="visible">
      {letters.map((letter, index) => (
        <motion.span variants={child} key={index}>
          {letter === " " ? "\u00A0" : letter}
        </motion.span>
      ))}
    </motion.h1>
  );
};

const App = () => {
  const [showLanding, setShowLanding] = useState(true); 
  const [date, setDate] = useState('');
  const [city, setCity] = useState('bengaluru');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [summary, setSummary] = useState('');
  const [summarizing, setSummarizing] = useState(false);

  const getPrediction = async () => {
    if (!date) return;
    setLoading(true);
    setError(null);
    try {
      const endpoint =
        city === 'bengaluru'
          ? 'http://127.0.0.1:5000/predict-bangaluru'
          : 'http://127.0.0.1:5000/predict-delhi';

      const response = await axios.post(endpoint, { date });
      setPrediction(response.data);
    } catch (err) {
      console.error("Backend unreachable", err);
      setError('Unable to fetch prediction. Check server status.');
    } finally {
      setLoading(false);
    }
  };

  const handleSummarize = async () => {
    if (!prediction || !prediction.predictions) {
      setError('Please fetch weather predictions first.');
      return;
    }

    setSummarizing(true);
    try {
      const response = await axios.post('http://127.0.0.1:5000/summarize', {
        weather: prediction.predictions,
        planets: {},
        city: city,
        date: date
      });
      setSummary(response.data.summary);
    } catch (err) {
      console.error("Summarize failed", err);
      setError('Unable to generate summary. Please try again.');
    } finally {
      setSummarizing(false);
    }
  };

  return (
      <Router>
    <Routes>
      <Route path="/model-info" element={<ModelInfo />} />
      <Route
        path="/"
        element={
    <div className="app-container">
      <div className="stars" />

      <AnimatePresence mode="wait">
        {showLanding ? (
          <motion.div 
            key="landing"
            className="landing-overlay"
            exit={{ opacity: 0, scale: 1.1 }}
            transition={{ duration: 0.8 }}
          >
            {/* THE BIG CINEMATIC COMET */}
            <motion.div
              className="big-comet-container"
              initial={{ x: '-20vw', y: '-10vh', opacity: 0, rotate: 25 }}
              animate={{ x: '110vw', y: '50vh', opacity: [0, 1, 1, 0] }}
              transition={{ duration: 4.5, repeat: Infinity, repeatDelay: 2, ease: "linear" }}
            >
              <div className="comet-glow" />
              <div className="comet-head-green" />
              <div className="comet-tail-long" />
            </motion.div>

            <div className="landing-content">
              <WelcomeText text="WELCOME TO ASTROWEATHER AI" />
              <motion.p 
                initial={{ opacity: 0 }} 
                animate={{ opacity: 1 }} 
                transition={{ delay: 3 }}
                className="tagline"
              >
                Analyzing planetary vectors for terrestrial forecasts.
              </motion.p>
              <motion.button
                className="mission-button"
                whileHover={{ scale: 1.05, boxShadow: "0px 0px 20px #00f2ff" }}
                whileTap={{ scale: 0.95 }}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 3.5 }}
                onClick={() => setShowLanding(false)}
              >
                INITIALIZE MISSION
              </motion.button>
            </div>
          </motion.div>
        ) : (
          <motion.div 
            key="dashboard"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="dashboard-wrapper"
          >
            <motion.div
              className="comet"
              initial={{ x: '-100vw', y: '0vh', opacity: 0 }}
              animate={{ x: '100vw', y: '100vh', opacity: [0, 1, 0] }}
              transition={{ duration: 3, repeat: Infinity, repeatDelay: 7, ease: "linear" }}
            />

            <div className="solar-system">
              <div className="sun" />
              <motion.div
                className="orbit earth-orbit"
                animate={{ rotate: 360 }}
                transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
              >
                <div className="planet earth" />
              </motion.div>
            </div>
            <div className="panel-wrapper">
              {/* LEFT ATTACHED BUTTONS */}
              <div className="side-buttons">
                <motion.button
                  className="side-btn"
                  whileHover={{ x: -6, scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => window.open('http://127.0.0.1:5500/Frontend/index.html', '_blank')}
                >
                  Simulation
                </motion.button>

                <motion.button
                  className="side-btn"
                  whileHover={{ x: -6, scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => window.location.href = "/ask-more"}
                >
                  Ask More
                </motion.button>
                <motion.button
                  className="side-btn secondary"
                  whileHover={{ x: -6, scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => window.location.href = "/model-info"}
                >
                  Model Info
                </motion.button>

              </div>
            <main className="glass-card">
              <header>
                <h1><Orbit className="icon" /> AstroWeather <span>AI</span></h1>
                <p>Unlocking Earth's climate through the stars</p>
              </header>

              <section className="input-section">
                <div className="input-row">
                  <input type="date" value={date} onChange={(e) => setDate(e.target.value)} />
                  <select value={city} onChange={(e) => setCity(e.target.value)}>
                    <option value="bengaluru">Bengaluru</option>
                    <option value="delhi">Delhi</option>
                  </select>
                </div>
                <button onClick={getPrediction} disabled={loading}>
                  {loading ? "Analyzing NASA Vectors..." : "Initiate Prediction"}
                </button>
              </section>

              <AnimatePresence>
                {error && (
                  <motion.div
                    className="results error-card"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                  >
                    {error}
                  </motion.div>
                )}
              </AnimatePresence>

              <AnimatePresence>
                {prediction && prediction.predictions && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="results"
                  >
                    <div className="results-header">
                      <div className="temp-display">
                        <Thermometer />
                        <h2>{prediction.predictions.T2M ?? '--'}°C</h2>
                      </div>
                      <div className="status-block">
                        <p className="status">Target Date: {prediction.date}</p>
                        <p className="status sub">City: {city === 'bengaluru' ? 'Bengaluru' : 'Delhi'}</p>
                      </div>
                    </div>

                    <div className="metrics-grid">
                      {Object.entries(prediction.predictions).map(([key, value]) => {
                        const info = PARAMETER_INFO[key] || { name: key, unit: '', icon: Cloud };
                        const IconComponent = info.icon;
                        return (
                          <div className="metric-card" key={key}>
                            <div className="metric-label">
                              <IconComponent size={18} style={{ display: 'inline', marginRight: '0.5rem', verticalAlign: 'middle' }} />
                              {info.name}
                            </div>
                            <div className="metric-value">
                              {value}{info.unit && <span style={{ fontSize: '0.8em', marginLeft: '0.3em' }}>{info.unit}</span>}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                    <motion.button
                      className="summarize-button"
                      onClick={handleSummarize}
                      disabled={summarizing}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      style={{
                        marginTop: '2rem',
                        padding: '0.9rem 2rem',
                        background: 'linear-gradient(45deg, #6366f1, #00f2ff)',
                        border: 'none',
                        borderRadius: '12px',
                        color: '#000',
                        cursor: summarizing ? 'not-allowed' : 'pointer',
                        fontWeight: 'bold',
                        fontSize: '1rem',
                        fontFamily: 'Orbitron, sans-serif',
                        opacity: summarizing ? 0.6 : 1,
                        boxShadow: '0 4px 15px rgba(0, 242, 255, 0.3)'
                      }}
                    >
                      {summarizing ? '🌌 Generating Summary...' : '✨ Summarize Weather Report'}
                    </motion.button>

                    {summary && (
                      <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="results"
                        style={{ marginTop: '1.5rem', textAlign: 'left' }}
                      >
                        <h3 style={{
                          margin: '0 0 1.5rem 0',
                          fontSize: '1.2rem',
                          color: '#00f2ff',
                          letterSpacing: '1px',
                          fontWeight: '700',
                          borderBottom: '2px solid rgba(0, 242, 255, 0.2)',
                          paddingBottom: '0.75rem'
                        }}>
                          WEATHER SUMMARY REPORT
                        </h3>
                        <div style={{
                          whiteSpace: 'pre-wrap',
                          wordWrap: 'break-word',
                          lineHeight: 1.8,
                          color: 'rgba(255, 255, 255, 0.9)',
                          fontSize: '0.95rem',
                          fontFamily: '"Segoe UI", Tahoma, Geneva, Verdana, sans-serif',
                          textAlign: 'justify'
                        }}>
                          {summary}
                        </div>
                      </motion.div>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </main>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
    }
      />

      <Route path="/ask-more" element={<AskMore />} />
      
    </Routes>
  </Router>
  );
};

export default App;