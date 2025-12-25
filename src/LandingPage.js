import React from 'react';
import { motion } from 'framer-motion';
import WelcomeText from './WelcomeText';
import './LandingPage.css';

const LandingPage = ({ onEnter }) => {
  return (
    <div className="landing-container">
      {/* Dynamic Comet Animation */}
      <motion.div
        className="main-comet"
        initial={{ x: '-10vw', y: '-10vh', opacity: 0, scale: 0.5 }}
        animate={{ 
          x: '110vw', 
          y: '50vh', 
          opacity: [0, 1, 1, 0],
          scale: [0.5, 1, 1, 0.2] 
        }}
        transition={{ 
          duration: 4, 
          ease: "easeInOut",
          repeat: Infinity,
          repeatDelay: 2
        }}
      >
        <div className="comet-head" />
        <div className="comet-tail" />
      </motion.div>

      <div className="content-wrapper">
        <WelcomeText text="WELCOME TO ASTROWEATHER AI" />
        
        <motion.p 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 3, duration: 1 }}
          className="sub-tagline"
        >
          Decoding the thermal signature of our solar system.
        </motion.p>

        <motion.button
          whileHover={{ scale: 1.1, boxShadow: "0px 0px 20px #00f2ff" }}
          whileTap={{ scale: 0.95 }}
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 3.5 }}
          className="enter-button"
          onClick={onEnter}
        >
          INITIALIZE MISSION
        </motion.button>
      </div>

      {/* Background Starfield */}
      <div className="star-field" />
    </div>
  );
};

export default LandingPage;