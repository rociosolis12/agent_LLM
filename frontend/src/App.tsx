// src/App.jsx

import React, { useState, useEffect } from 'react';
import './App.css';

// Importar el nuevo componente del predictor
import { HybridPredictorDashboard } from './components/predictor/HybridPredictorDashboard';

function App() {
  const [systemStatus, setSystemStatus] = useState(null);
  const [selectedFunction, setSelectedFunction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [responses, setResponses] = useState([]);
  const [showChat, setShowChat] = useState(false);
  const [chatQuestion, setChatQuestion] = useState('');
  const [modalResponse, setModalResponse] = useState(null);
  
  // NUEVO: Estado para navegación entre secciones
  const [currentSection, setCurrentSection] = useState('dashboard'); // 'dashboard' o 'predictor'

  // Verificar estado del backend al cargar
  useEffect(() => {
    checkBackendStatus();
    const interval = setInterval(checkBackendStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const checkBackendStatus = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/system-status');
      const data = await response.json();
      setSystemStatus(data);
    } catch (error) {
      console.log('Backend no disponible:', error);
      setSystemStatus(null);
    }
  };

  const handleFunctionClick = async (functionName) => {
    if (loading) return;
    
    setSelectedFunction(functionName);
    setLoading(true);
    setShowChat(false);
    
    try {
      const response = await fetch('http://127.0.0.1:8000/execute-function', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          function_name: functionName,
          question: functionName === 'text_to_sql' ? chatQuestion : null
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      
      const data = await response.json();
      
      setResponses(prev => [{
        function: functionName,
        data: data,
        timestamp: new Date().toISOString()
      }, ...prev]);
      
      if (functionName === 'text_to_sql') {
        setShowChat(false);
        setChatQuestion('');
      }
      
    } catch (error) {
      console.error('Error:', error);
      alert(`Error ejecutando ${functionName}: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleChatSubmit = (e) => {
    e.preventDefault();
    if (chatQuestion.trim()) {
      handleFunctionClick('text_to_sql');
    }
  };

  const openModal = (response) => {
    setModalResponse(response);
  };

  const closeModal = () => {
    setModalResponse(null);
  };

  const formatResponse = (data) => {
    if (typeof data === 'object') {
      return JSON.stringify(data, null, 2);
    }
    return String(data);
  };

  const getStatusColor = () => {
    if (!systemStatus) return '#ff4444';
    return systemStatus.status === 'operational' ? '#4CAF50' : '#ff9800';
  };

  const functions = [
    { 
      name: 'ingest_retrieve', 
      title: 'Ingest & Retrieve',
      description: 'Procesar documentos y realizar búsquedas',
      icon: '📄'
    },
    { 
      name: 'text_to_sql', 
      title: 'Text to SQL',
      description: 'Convertir preguntas en consultas SQL',
      icon: '💬',
      requiresInput: true
    },
    { 
      name: 'embeddings_memory', 
      title: 'Embeddings con Memory',
      description: 'Vector store con memoria persistente',
      icon: '🧠'
    },
    { 
      name: 'text_to_cypher', 
      title: 'Text to Cypher',
      description: 'Consultas en bases de datos de grafos',
      icon: '🔗'
    }
  ];

  return (
    <div className="App">
      {/* NUEVO: Navegación entre secciones */}
      <div className="section-nav">
        <button 
          className={`section-tab ${currentSection === 'dashboard' ? 'active' : ''}`}
          onClick={() => setCurrentSection('dashboard')}
        >
          📊 Dashboard Principal
        </button>
        <button 
          className={`section-tab ${currentSection === 'predictor' ? 'active' : ''}`}
          onClick={() => setCurrentSection('predictor')}
        >
          🚀 Predictor Híbrido <span className="beta-badge">Beta</span>
        </button>
      </div>

      {/* Mostrar Dashboard Original */}
      {currentSection === 'dashboard' && (
        <div className="App-header">
          <div className="status-indicator" style={{ backgroundColor: getStatusColor() }}>
            {systemStatus ? (
              <>
                <span className="status-dot"></span>
                Backend: {systemStatus.status}
              </>
            ) : (
              <>
                <span className="status-dot offline"></span>
                Backend: Offline
              </>
            )}
          </div>

          <h1>🤖 Sistema Multi-Agente LLM</h1>
          <h2>Selecciona una función para ejecutar</h2>

          <div className="functions-grid">
            {functions.map((func) => (
              <div
                key={func.name}
                className={`function-card ${selectedFunction === func.name ? 'active' : ''} ${loading ? 'disabled' : ''}`}
                onClick={() => {
                  if (func.requiresInput) {
                    setShowChat(true);
                    setSelectedFunction(func.name);
                  } else {
                    handleFunctionClick(func.name);
                  }
                }}
              >
                <div className="function-icon">{func.icon}</div>
                <h3>{func.title}</h3>
                <p>{func.description}</p>
                {loading && selectedFunction === func.name && (
                  <div className="loading-spinner">
                    <div className="spinner"></div>
                  </div>
                )}
              </div>
            ))}
          </div>

          {showChat && selectedFunction === 'text_to_sql' && (
            <div className="chat-interface">
              <h3>💬 Ingresa tu pregunta</h3>
              <form onSubmit={handleChatSubmit}>
                <input
                  type="text"
                  value={chatQuestion}
                  onChange={(e) => setChatQuestion(e.target.value)}
                  placeholder="Ej: ¿Cuáles son las ventas del último mes?"
                  className="chat-input"
                  autoFocus
                />
                <button 
                  type="submit" 
                  className="chat-submit"
                  disabled={loading || !chatQuestion.trim()}
                >
                  {loading ? 'Procesando...' : 'Enviar'}
                </button>
              </form>
            </div>
          )}

          {responses.length > 0 && (
            <div className="responses-section">
              <h2>📋 Historial de Respuestas</h2>
              <div className="responses-grid">
                {responses.map((response, index) => (
                  <div key={index} className="response-card">
                    <div className="response-header">
                      <h3>{functions.find(f => f.name === response.function)?.title || response.function}</h3>
                      <small>{new Date(response.timestamp).toLocaleString()}</small>
                    </div>
                    <div className="response-preview">
                      <pre>{formatResponse(response.data).substring(0, 200)}...</pre>
                    </div>
                    <button 
                      className="view-details-btn"
                      onClick={() => openModal(response)}
                    >
                      Ver detalles completos
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {modalResponse && (
            <div className="modal-overlay" onClick={closeModal}>
              <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                <div className="modal-header">
                  <h2>{functions.find(f => f.name === modalResponse.function)?.title || modalResponse.function}</h2>
                  <button className="close-btn" onClick={closeModal}>✕</button>
                </div>
                <div className="modal-body">
                  <pre>{formatResponse(modalResponse.data)}</pre>
                </div>
                <div className="modal-footer">
                  <small>Timestamp: {new Date(modalResponse.timestamp).toLocaleString()}</small>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* NUEVO: Mostrar Predictor Híbrido */}
      {currentSection === 'predictor' && (
        <div className="predictor-section">
          <HybridPredictorDashboard bankSymbol="BBVA.MC" />
        </div>
      )}
    </div>
  );
}

export default App;
