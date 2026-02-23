import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const CATEGORY_META = {
  violent_content:                       { icon: '[V]',  label: 'Violent Content' },
  crime_content:                         { icon: '[C]',  label: 'Crime Content' },
  abusive_content:                       { icon: '[A]',  label: 'Abusive Content' },
  hateful_content:                       { icon: '[H]',  label: 'Hateful Content' },
  'cybersecurity_threats_(beyond_malware)': { icon: '[CS]', label: 'Cybersecurity Threat' },
  cybersecurity_threats_beyond_malware:  { icon: '[CS]',  label: 'Cybersecurity Threat' },
  'self-harm_content':                   { icon: '[SH]',  label: 'Self-Harm Content' },
  self_harm_content:                     { icon: '[SH]',  label: 'Self-Harm Content' },
  malware_code:                          { icon: '[M]',  label: 'Malware / Code' },
  'illegal_weapons_(non-cbrn)':          { icon: '[W]',  label: 'Illegal Weapons' },
  illegal_weapons_non_cbrn:              { icon: '[W]',  label: 'Illegal Weapons' },
  misinformation:                        { icon: '[MI]',  label: 'Misinformation' },
  economic_harm:                         { icon: '[E]',  label: 'Economic Harm' },
  child_safety:                          { icon: '[CS]',  label: 'Child Safety' },
  extremism_and_radicalization:          { icon: '[EX]',  label: 'Extremism' },
  sexual_content:                        { icon: '[SC]',  label: 'Sexual Content' },
  data_privacy:                          { icon: '[DP]',  label: 'Data Privacy' },
  environmental_harm:                    { icon: '[EH]',  label: 'Environmental Harm' },
  cbrn_information_or_capabilities:      { icon: '[CBRN]',  label: 'CBRN Threat' },
  bias_content:                          { icon: '[B]',  label: 'Bias Content' },
  election_interference:                 { icon: '[EI]',  label: 'Election Interference' },
  intellectual_property:                 { icon: '[IP]',   label: 'Intellectual Property' },
};

function getCategoryMeta(raw) {
  if (!raw) return { icon: '[U]', label: 'Unsafe Content' };
  const key = raw.toLowerCase().replace(/[\s\-]/g, '_');
  return (
    CATEGORY_META[key] ||
    CATEGORY_META[raw] || {
      icon: '🛡️',
      label: raw.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
    }
  );
}

function BlockedCard({ category, triggerWord, confidence }) {
  const meta = getCategoryMeta(category);

  return (
    <div className="blocked-card">
      <div className="blocked-header">
        <span className="blocked-icon">[!]</span>
        <span className="blocked-title">Prompt didn't go through</span>
      </div>

      <div className="blocked-body">
        <div className="blocked-row">
          <span className="blocked-label">Category</span>
          <span className="category-pill">{meta.icon} {meta.label}</span>
        </div>

        {triggerWord && (
          <div className="blocked-row">
            <span className="blocked-label">Trigger word</span>
            <span className="trigger-pill">"{triggerWord}"</span>
          </div>
        )}
      </div>

      <div className="blocked-footer">
        flagged by jarvis ai safety filter
      </div>
    </div>
  );
}

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
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
        query: input,
      });

      let aiMessage;
      if (response.data.status === 'blocked') {
        aiMessage = {
          role:        'assistant',
          type:        'blocked',
          category:    response.data.category_raw || response.data.category,
          triggerWord: response.data.trigger_word,
          confidence:  response.data.confidence,
        };
      } else {
        aiMessage = {
          role:    'assistant',
          type:    'text',
          content: response.data.response,
        };
      }

      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      setMessages(prev => [...prev, {
        role:    'assistant',
        type:    'text',
        content: 'Something went wrong. Please try again.',
      }]);
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <div className="chat-container">

        <div className="chat-header">
          <h1><em>Jarvis</em> AI Assistant</h1>
        </div>

        <div className="messages-container">
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`message ${msg.role}${msg.type === 'blocked' ? ' blocked-msg' : ''}`}
            >
              {msg.type === 'blocked' ? (
                <BlockedCard
                  category={msg.category}
                  triggerWord={msg.triggerWord}
                  confidence={msg.confidence}
                />
              ) : (
                <div className="message-content">{msg.content}</div>
              )}
            </div>
          ))}

          {loading && (
            <div className="message assistant">
              <div className="message-content">
                <div className="typing">
                  <span /><span /><span />
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        <form onSubmit={sendMessage} className="input-form">
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Ask Jarvis anything..."
            disabled={loading}
          />
          <button type="submit" disabled={loading}>Send</button>
        </form>

      </div>
    </div>
  );
}

export default App;