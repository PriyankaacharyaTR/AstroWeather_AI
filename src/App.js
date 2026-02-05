import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Thermometer, Orbit, Gauge, Droplets, Wind, Cloud, ArrowUpDown, AlertTriangle, TrendingUp } from 'lucide-react';
import axios from 'axios';
import './App.css';
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import AskMore from "./AskMore";
import ModelInfo from "./ModelInfo";
import ShapExplanation from "./ShapExplanation";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart } from 'recharts';

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
  // Check URL parameter to skip landing page (for back navigation only)
  const urlParams = new URLSearchParams(window.location.search);
  const skipLanding = urlParams.get('dashboard') === 'true';
  const [showLanding, setShowLanding] = useState(!skipLanding); 
  const [date, setDate] = useState('');
  const [city, setCity] = useState('bengaluru');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [summary, setSummary] = useState('');
  const [summarizing, setSummarizing] = useState(false);
  const [comparison, setComparison] = useState(null);
  const [comparingTemp, setComparingTemp] = useState(false);
  const [arimaForecast, setArimaForecast] = useState(null);
  const [loadingForecast, setLoadingForecast] = useState(false);

  const getArimaForecast = async (selectedCity) => {
    setLoadingForecast(true);
    try {
      const response = await axios.post('http://127.0.0.1:5000/api/arima/forecast', {
        steps: 10,
        parameters: ['T2M'],
        city: selectedCity || city  // Use passed city or current city state
      });
      setArimaForecast(response.data);
    } catch (err) {
      console.error("ARIMA forecast failed", err);
      // Don't show error - just silently fail if ARIMA not available
    } finally {
      setLoadingForecast(false);
    }
  };

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
      
      // Also fetch ARIMA 10-day temperature forecast for selected city
      getArimaForecast(city);
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

  const handleCompareTemperature = async () => {
    if (!date) {
      setError('Please select a date first.');
      return;
    }

    setComparingTemp(true);
    setComparison(null);
    try {
      const response = await axios.post('http://127.0.0.1:5000/api/temperature-comparison', {
        date: date,
        city: city
      });
      setComparison(response.data);
    } catch (err) {
      console.error("Comparison failed", err);
      const errMsg = err.response?.data?.error || 'Unable to fetch comparison. Check if WeatherAPI key is configured.';
      setError(errMsg);
    } finally {
      setComparingTemp(false);
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
                <motion.button
                  className="side-btn shap"
                  whileHover={{ x: -6, scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => window.location.href = "/shap-explanation"}
                >
                  Scientific Explanation (SHAP)
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

                    {/* ARIMA 10-Day Temperature Forecast Chart */}
                    {arimaForecast && arimaForecast.forecasts && arimaForecast.forecasts.T2M && (
                      <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.3 }}
                        style={{
                          marginTop: '2rem',
                          background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(0, 242, 255, 0.05))',
                          borderRadius: '16px',
                          padding: '1.5rem',
                          border: '1px solid rgba(99, 102, 241, 0.2)'
                        }}
                      >
                        <h3 style={{
                          margin: '0 0 1rem 0',
                          fontSize: '1.1rem',
                          color: '#00f2ff',
                          letterSpacing: '1px',
                          fontWeight: '700',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.5rem'
                        }}>
                          <TrendingUp size={20} />
                          10-DAY TEMPERATURE FORECAST (ARIMA)
                        </h3>
                        <p style={{ 
                          color: 'rgba(255,255,255,0.6)', 
                          fontSize: '0.8rem', 
                          marginBottom: '1rem' 
                        }}>
                          Time-series forecast based on historical weather patterns
                        </p>
                        
                        <ResponsiveContainer width="100%" height={300}>
                          <AreaChart
                            data={arimaForecast.forecasts.T2M.dates.map((date, idx) => ({
                              date: new Date(date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
                              temperature: arimaForecast.forecasts.T2M.values[idx],
                              lowerCI: arimaForecast.forecasts.T2M.lower_ci[idx],
                              upperCI: arimaForecast.forecasts.T2M.upper_ci[idx]
                            }))}
                            margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
                          >
                            <defs>
                              <linearGradient id="colorTemp" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#00f2ff" stopOpacity={0.8}/>
                                <stop offset="95%" stopColor="#00f2ff" stopOpacity={0.1}/>
                              </linearGradient>
                              <linearGradient id="colorCI" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3}/>
                                <stop offset="95%" stopColor="#6366f1" stopOpacity={0.05}/>
                              </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                            <XAxis 
                              dataKey="date" 
                              stroke="rgba(255,255,255,0.5)"
                              tick={{ fill: 'rgba(255,255,255,0.7)', fontSize: 12 }}
                            />
                            <YAxis 
                              stroke="rgba(255,255,255,0.5)"
                              tick={{ fill: 'rgba(255,255,255,0.7)', fontSize: 12 }}
                              domain={['dataMin - 2', 'dataMax + 2']}
                              tickFormatter={(value) => `${value}°C`}
                            />
                            <Tooltip 
                              contentStyle={{
                                background: 'rgba(0, 0, 0, 0.9)',
                                border: '1px solid rgba(0, 242, 255, 0.3)',
                                borderRadius: '8px',
                                color: '#fff'
                              }}
                              formatter={(value, name) => {
                                const labels = {
                                  temperature: 'Predicted',
                                  lowerCI: 'Lower Bound',
                                  upperCI: 'Upper Bound'
                                };
                                return [`${value.toFixed(1)}°C`, labels[name] || name];
                              }}
                            />
                            <Legend 
                              wrapperStyle={{ color: 'rgba(255,255,255,0.7)' }}
                              formatter={(value) => {
                                const labels = {
                                  temperature: 'Temperature',
                                  lowerCI: 'Lower CI (95%)',
                                  upperCI: 'Upper CI (95%)'
                                };
                                return labels[value] || value;
                              }}
                            />
                            <Area 
                              type="monotone" 
                              dataKey="upperCI" 
                              stroke="transparent"
                              fill="url(#colorCI)" 
                              name="upperCI"
                            />
                            <Area 
                              type="monotone" 
                              dataKey="lowerCI" 
                              stroke="transparent"
                              fill="rgba(99, 102, 241, 0.1)" 
                              name="lowerCI"
                            />
                            <Line 
                              type="monotone" 
                              dataKey="temperature" 
                              stroke="#00f2ff" 
                              strokeWidth={3}
                              dot={{ fill: '#00f2ff', strokeWidth: 2, r: 5 }}
                              activeDot={{ r: 8, fill: '#00f2ff', stroke: '#fff', strokeWidth: 2 }}
                              name="temperature"
                            />
                          </AreaChart>
                        </ResponsiveContainer>

                        <div style={{
                          marginTop: '1rem',
                          display: 'grid',
                          gridTemplateColumns: 'repeat(5, 1fr)',
                          gap: '0.5rem'
                        }}>
                          {arimaForecast.forecasts.T2M.values.slice(0, 5).map((temp, idx) => (
                            <div key={idx} style={{
                              background: 'rgba(0, 242, 255, 0.1)',
                              borderRadius: '8px',
                              padding: '0.5rem',
                              textAlign: 'center',
                              border: '1px solid rgba(0, 242, 255, 0.2)'
                            }}>
                              <p style={{ color: 'rgba(255,255,255,0.6)', fontSize: '0.7rem', margin: 0 }}>
                                Day {idx + 1}
                              </p>
                              <p style={{ color: '#00f2ff', fontSize: '1rem', fontWeight: 'bold', margin: '0.25rem 0 0 0' }}>
                                {temp.toFixed(1)}°C
                              </p>
                            </div>
                          ))}
                        </div>

                        <p style={{
                          marginTop: '1rem',
                          color: 'rgba(255,255,255,0.5)',
                          fontSize: '0.75rem',
                          textAlign: 'center'
                        }}>
                          Model: {arimaForecast.forecasts.T2M.model_params} | Shaded area represents 95% confidence interval
                        </p>
                      </motion.div>
                    )}

                    {loadingForecast && (
                      <div style={{
                        marginTop: '2rem',
                        textAlign: 'center',
                        color: 'rgba(255,255,255,0.6)'
                      }}>
                        <p>📈 Loading 10-day forecast...</p>
                      </div>
                    )}

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

                    <motion.button
                      className="compare-button"
                      onClick={handleCompareTemperature}
                      disabled={comparingTemp}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      style={{
                        marginTop: '1rem',
                        padding: '0.9rem 2rem',
                        background: 'linear-gradient(45deg, #f59e0b, #ef4444)',
                        border: 'none',
                        borderRadius: '12px',
                        color: '#000',
                        cursor: comparingTemp ? 'not-allowed' : 'pointer',
                        fontWeight: 'bold',
                        fontSize: '1rem',
                        fontFamily: 'Orbitron, sans-serif',
                        opacity: comparingTemp ? 0.6 : 1,
                        boxShadow: '0 4px 15px rgba(245, 158, 11, 0.3)'
                      }}
                    >
                      {comparingTemp ? '🔄 Comparing...' : '🌡️ Compare with Real Weather'}
                    </motion.button>

                    {comparison && (
                      <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="results comparison-section"
                        style={{ marginTop: '1.5rem', textAlign: 'left' }}
                      >
                        <h3 style={{
                          margin: '0 0 1.5rem 0',
                          fontSize: '1.2rem',
                          color: '#f59e0b',
                          letterSpacing: '1px',
                          fontWeight: '700',
                          borderBottom: '2px solid rgba(245, 158, 11, 0.2)',
                          paddingBottom: '0.75rem',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.5rem'
                        }}>
                          <ArrowUpDown size={20} />
                          TEMPERATURE COMPARISON
                        </h3>
                        
                        <div style={{
                          display: 'grid',
                          gridTemplateColumns: '1fr 1fr',
                          gap: '1.5rem',
                          marginBottom: '1.5rem'
                        }}>
                          {/* Astro Prediction Box */}
                          <div style={{
                            background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(0, 242, 255, 0.1))',
                            borderRadius: '12px',
                            padding: '1.5rem',
                            border: '1px solid rgba(99, 102, 241, 0.3)',
                            textAlign: 'center'
                          }}>
                            <p style={{ color: '#00f2ff', fontSize: '0.85rem', marginBottom: '0.5rem', fontWeight: '600' }}>
                              🌌 ASTRO PREDICTION
                            </p>
                            <h2 style={{ fontSize: '2.5rem', margin: '0.5rem 0', color: '#fff' }}>
                              {comparison.astro_prediction.temperature}°C
                            </h2>
                            <p style={{ color: 'rgba(255,255,255,0.6)', fontSize: '0.75rem' }}>
                              {comparison.astro_prediction.source}
                            </p>
                          </div>

                          {/* Actual Weather Box */}
                          <div style={{
                            background: 'linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(239, 68, 68, 0.1))',
                            borderRadius: '12px',
                            padding: '1.5rem',
                            border: '1px solid rgba(245, 158, 11, 0.3)',
                            textAlign: 'center'
                          }}>
                            <p style={{ color: '#f59e0b', fontSize: '0.85rem', marginBottom: '0.5rem', fontWeight: '600' }}>
                              🌍 ACTUAL WEATHER
                            </p>
                            <h2 style={{ fontSize: '2.5rem', margin: '0.5rem 0', color: '#fff' }}>
                              {comparison.actual_weather.temperature}°C
                            </h2>
                            <p style={{ color: 'rgba(255,255,255,0.5)', fontSize: '0.7rem', marginBottom: '0.3rem' }}>
                              (Avg Temperature)
                            </p>
                            <div style={{ 
                              display: 'flex', 
                              justifyContent: 'center', 
                              gap: '1rem', 
                              marginTop: '0.5rem',
                              fontSize: '0.8rem'
                            }}>
                              <span style={{ color: '#60a5fa' }}>
                                ❄️ Min: {comparison.actual_weather.min_temp}°C
                              </span>
                              <span style={{ color: '#f87171' }}>
                                🔥 Max: {comparison.actual_weather.max_temp}°C
                              </span>
                            </div>
                            <p style={{ color: 'rgba(255,255,255,0.6)', fontSize: '0.75rem', marginTop: '0.5rem' }}>
                              {comparison.actual_weather.source}
                            </p>
                            <p style={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.8rem', marginTop: '0.5rem' }}>
                              {comparison.actual_weather.condition}
                            </p>
                          </div>
                        </div>

                        {/* Difference Display */}
                        <div style={{
                          background: comparison.comparison.abs_difference <= 2 
                            ? 'rgba(34, 197, 94, 0.15)' 
                            : comparison.comparison.abs_difference <= 5 
                              ? 'rgba(245, 158, 11, 0.15)'
                              : 'rgba(239, 68, 68, 0.15)',
                          borderRadius: '12px',
                          padding: '1rem 1.5rem',
                          textAlign: 'center',
                          border: `1px solid ${comparison.comparison.abs_difference <= 2 
                            ? 'rgba(34, 197, 94, 0.3)' 
                            : comparison.comparison.abs_difference <= 5 
                              ? 'rgba(245, 158, 11, 0.3)'
                              : 'rgba(239, 68, 68, 0.3)'}`
                        }}>
                          <p style={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.85rem', marginBottom: '0.5rem' }}>
                            Difference
                          </p>
                          <h3 style={{ 
                            fontSize: '1.5rem', 
                            margin: 0, 
                            color: comparison.comparison.abs_difference <= 2 
                              ? '#22c55e' 
                              : comparison.comparison.abs_difference <= 5 
                                ? '#f59e0b'
                                : '#ef4444'
                          }}>
                            {comparison.comparison.astro_higher ? '+' : ''}{comparison.comparison.difference}°C
                          </h3>
                          <p style={{ color: 'rgba(255,255,255,0.5)', fontSize: '0.75rem', marginTop: '0.5rem' }}>
                            {comparison.comparison.abs_difference <= 2 
                              ? '✅ Excellent accuracy!' 
                              : comparison.comparison.abs_difference <= 5 
                                ? '⚡ Good prediction'
                                : '⚠️ Notable variance'}
                          </p>
                        </div>

                        {/* Disclaimer */}
                        <div style={{
                          marginTop: '1rem',
                          padding: '0.75rem 1rem',
                          background: 'rgba(255, 255, 255, 0.05)',
                          borderRadius: '8px',
                          display: 'flex',
                          alignItems: 'flex-start',
                          gap: '0.5rem'
                        }}>
                          <AlertTriangle size={16} style={{ color: '#f59e0b', flexShrink: 0, marginTop: '2px' }} />
                          <p style={{
                            color: 'rgba(255, 255, 255, 0.6)',
                            fontSize: '0.75rem',
                            margin: 0,
                            lineHeight: 1.5,
                            textAlign: 'left'
                          }}>
                            {comparison.disclaimer}
                          </p>
                        </div>
                      </motion.div>
                    )}

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
      <Route path="/shap-explanation" element={<ShapExplanation />} />
      
    </Routes>
  </Router>
  );
};

export default App;