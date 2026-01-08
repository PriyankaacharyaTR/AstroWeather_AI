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

      <motion.div
        className="chat-glass"
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <header className="chat-header">
          <ArrowLeft onClick={() => window.location.href = "/"} />
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
