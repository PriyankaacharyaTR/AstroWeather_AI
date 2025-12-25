import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Thermometer, Orbit } from 'lucide-react';
import axios from 'axios';
import './App.css';

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
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const getPrediction = async () => {
    if (!date) return;
    setLoading(true);
    try {
      const response = await axios.post('http://127.0.0.1:5000/predict', { date });
      setPrediction(response.data);
    } catch (err) {
      console.error("Backend unreachable", err);
    } finally {
      setLoading(false);
    }
  };

  return (
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

            <main className="glass-card">
              <header>
                <h1><Orbit className="icon" /> AstroWeather <span>AI</span></h1>
                <p>Unlocking Earth's climate through the stars</p>
              </header>

              <section className="input-section">
                <input type="date" value={date} onChange={(e) => setDate(e.target.value)} />
                <button onClick={getPrediction} disabled={loading}>
                  {loading ? "Analyzing NASA Vectors..." : "Initiate Prediction"}
                </button>
              </section>

              <AnimatePresence>
                {prediction && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="results"
                  >
                    <div className="temp-display">
                      <Thermometer />
                      <h2>{prediction.temperature}°C</h2>
                    </div>
                    <p className="status">Target Date: {date}</p>
                  </motion.div>
                )}
              </AnimatePresence>
            </main>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default App;