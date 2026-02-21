import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await axios.post('http://localhost:8000/api/chat', {
        query: input
      });

      const aiMessage = { 
        role: 'assistant', 
        content: response.data.response 
      };
      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      const errorMessage = { 
        role: 'assistant', 
        content: 'Sorry, something went wrong. Please try again.' 
      };
      setMessages(prev => [...prev, errorMessage]);
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <div className="chat-container">
        <div className="chat-header">
          <h1>🤖 Jarvis AI Assistant</h1>
          <p>Your Personal Enterprise AI</p>
        </div>

        <div className="messages-container">
          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              <div className="message-content">
                {msg.content}
              </div>
            </div>
          ))}
          {loading && (
            <div className="message assistant">
              <div className="message-content typing">
                Thinking...
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <form onSubmit={sendMessage} className="input-form">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask Jarvis anything..."
            disabled={loading}
          />
          <button type="submit" disabled={loading}>
            Send
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;