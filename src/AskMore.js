import React, { useState } from "react";
import { motion } from "framer-motion";
import { Send, ArrowLeft } from "lucide-react";
import "./AskMore.css";

const AskMore = () => {
  const [messages, setMessages] = useState([
    { role: "bot", text: "Greetings, explorer. Ask me about planetary influence, ancient calculations, or climate patterns." }
  ]);
  const [input, setInput] = useState("");

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = input;
    setMessages(prev => [
      ...prev,
      { role: "user", text: userMessage }
    ]);
    setInput("");

    try {
      const response = await fetch('http://localhost:5000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: userMessage }),
      });
      const data = await response.json();
      if (response.ok) {
        setMessages(prev => [
          ...prev,
          { role: "bot", text: data.answer }
        ]);
      } else {
        setMessages(prev => [
          ...prev,
          { role: "bot", text: `Error: ${data.error}` }
        ]);
      }
    } catch (error) {
      setMessages(prev => [
        ...prev,
        { role: "bot", text: "Sorry, I couldn't connect to the server." }
      ]);
    }
  };

  return (
    <div className="chat-container">
      <div className="stars" />

      {/* Back Button */}
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
          zIndex: 100,
        }}
      >
        <ArrowLeft size={16} /> Back
      </motion.button>

      <motion.div
        className="chat-glass"
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <header className="chat-header">
          <h2>AstroWeather AI · Oracle</h2>
        </header>

        <div className="chat-body">
          {messages.map((msg, i) => (
            <div key={i} className={`chat-bubble ${msg.role}`}>
              {msg.text}
            </div>
          ))}
        </div>

        <div className="chat-input">
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Ask about planets, seasons, heat, rainfall…"
            onKeyDown={e => e.key === "Enter" && sendMessage()}
          />
          <button onClick={sendMessage}>
            <Send size={18} />
          </button>
        </div>
      </motion.div>
    </div>
  );
};

export default AskMore;
