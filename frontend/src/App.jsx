import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [systemStatus, setSystemStatus] = useState(null);
  const [selectedFunction, setSelectedFunction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [responses, setResponses] = useState([]);
  const [showChat, setShowChat] = useState(false);
  const [chatQuestion, setChatQuestion] = useState('');
  const [modalResponse, setModalResponse] = useState(null);

  // Verificar estado del backend al cargar
  useEffect(() => {
    checkBackendStatus();
    // Verificar estado cada 30 segundos
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
    setLoading(true);
    setSelectedFunction(functionName);
    
    // Mapeo de preguntas más específicas y detalladas
    const questionMap = {
      'balance': '¿Puedes analizar el balance general del documento? Incluye activos totales, pasivos, patrimonio y ratios financieros importantes. Proporciona cifras específicas y análisis detallado.',
      'income': '¿Cuáles son los principales ingresos y gastos del estado de resultados? Analiza la rentabilidad, márgenes y evolución financiera con datos concretos.',
      'cashflows': '¿Cómo están los flujos de efectivo? Analiza flujos operativos, de inversión y financiación. Incluye variaciones y tendencias importantes.',
      'equity': '¿Cuál es la situación del patrimonio y capital? Examina cambios en capital social, reservas y resultados acumulados con análisis detallado.',
      'predictions': '¿Qué predicciones y recomendaciones puedes hacer sobre estas finanzas? Incluye tendencias futuras, riesgos y oportunidades basadas en los datos.'
    };

    try {
      const question = questionMap[functionName] || 'Analiza el documento financiero de manera integral';
      
      console.log('🔍 Enviando solicitud a:', 'http://127.0.0.1:8000/ask-question');
      console.log('📝 Pregunta:', question);
      
      // ✅ URL CORREGIDA - ask-question en lugar de query
      const response = await fetch('http://127.0.0.1:8000/ask-question', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: question })
      });

      console.log('📡 Status de respuesta:', response.status, response.statusText);

      if (!response.ok) {
        throw new Error(`Error del servidor: ${response.status} - ${response.statusText}`);
      }

      const result = await response.json();
      console.log('✅ Datos recibidos:', result);
      
      // Extraer respuesta con múltiples campos posibles
      const responseText = result.answer || result.response || result.message || result.data || 'No se pudo obtener una respuesta específica del agente.';
      
      // Agregar respuesta al historial
      const newResponse = {
        id: Date.now(),
        function: functionName,
        question: question,
        answer: responseText,
        timestamp: new Date().toLocaleTimeString(),
        fullTimestamp: new Date().toLocaleString()
      };
      
      setResponses(prev => [newResponse, ...prev]);
      
      // Mostrar en modal en lugar de alert simple
      setModalResponse({
        title: `📊 Análisis: ${functions.find(f => f.id === functionName)?.name}`,
        content: responseText,
        timestamp: newResponse.fullTimestamp,
        functionName: functionName
      });
      
    } catch (error) {
      console.error('❌ Error completo:', error);
      
      // Manejo de errores mejorado
      const errorMessage = `Error al consultar ${functionName}: ${error.message}`;
      
      const errorResponse = {
        id: Date.now(),
        function: functionName,
        question: 'Error de conexión',
        answer: error.message,
        timestamp: new Date().toLocaleTimeString(),
        fullTimestamp: new Date().toLocaleString(),
        isError: true
      };
      
      setResponses(prev => [errorResponse, ...prev]);
      
      // Mostrar error en modal
      setModalResponse({
        title: `❌ Error en ${functionName}`,
        content: errorMessage,
        timestamp: new Date().toLocaleString(),
        isError: true
      });
      
    } finally {
      // Siempre limpiar el estado de loading
      setLoading(false);
      setSelectedFunction(null);
    }
  };

  const handleCustomQuestion = async () => {
    if (!chatQuestion.trim()) return;
    
    setLoading(true);
    
    try {
      console.log('🔍 Enviando consulta personalizada a:', 'http://127.0.0.1:8000/ask-question');
      console.log('📝 Pregunta personalizada:', chatQuestion);
      
      // ✅ URL CORREGIDA - ask-question en lugar de query
      const response = await fetch('http://127.0.0.1:8000/ask-question', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: chatQuestion })
      });

      if (!response.ok) {
        throw new Error(`Error del servidor: ${response.status}`);
      }

      const result = await response.json();
      const responseText = result.answer || result.response || result.message || 'No se pudo obtener respuesta.';
      
      const newResponse = {
        id: Date.now(),
        function: 'custom',
        question: chatQuestion,
        answer: responseText,
        timestamp: new Date().toLocaleTimeString(),
        fullTimestamp: new Date().toLocaleString()
      };
      
      setResponses(prev => [newResponse, ...prev]);
      
      // Mostrar respuesta personalizada en modal
      setModalResponse({
        title: '💬 Consulta Personalizada',
        content: responseText,
        timestamp: newResponse.fullTimestamp,
        question: chatQuestion
      });
      
      setChatQuestion('');
      
    } catch (error) {
      console.error('❌ Error en consulta personalizada:', error);
      
      const errorResponse = {
        id: Date.now(),
        function: 'custom',
        question: chatQuestion,
        answer: error.message,
        timestamp: new Date().toLocaleTimeString(),
        fullTimestamp: new Date().toLocaleString(),
        isError: true
      };
      
      setResponses(prev => [errorResponse, ...prev]);
      
      setModalResponse({
        title: '❌ Error en Consulta',
        content: `Error: ${error.message}`,
        timestamp: new Date().toLocaleString(),
        isError: true
      });
    } finally {
      setLoading(false);
    }
  };

  const clearHistory = () => {
    setResponses([]);
  };

  const functions = [
    { 
      id: 'balance', 
      icon: '📊', 
      name: 'Balance General',
      description: 'Activos, pasivos y patrimonio',
      color: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
    },
    { 
      id: 'income', 
      icon: '💰', 
      name: 'Estado de Resultados',
      description: 'Ingresos, gastos y rentabilidad',
      color: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)'
    },
    { 
      id: 'cashflows', 
      icon: '💧', 
      name: 'Flujos de Efectivo',
      description: 'Operativos, inversión y financiación',
      color: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)'
    },
    { 
      id: 'equity', 
      icon: '📈', 
      name: 'Estado de Patrimonio',
      description: 'Capital y reservas',
      color: 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)'
    },
    { 
      id: 'predictions', 
      icon: '🔮', 
      name: 'Predicciones AI',
      description: 'Proyecciones inteligentes',
      color: 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)'
    }
  ];

  return (
    <div className="App">
      <header className="App-header">
        <div className="header-content">
          <h1>🤖 Sistema Multi-Agente Financiero</h1>
          
          <div className="status-indicator">
            {systemStatus ? (
              <div className="status-active">
                ✅ Sistema operativo - {systemStatus.available_agents?.length || 5} agentes activos
                <div className="health-score">
                  Salud del sistema: {Math.round((systemStatus.system_health?.health_score || 1) * 100)}%
                </div>
              </div>
            ) : (
              <div className="status-checking">
                ⚠️ Verificando conexión con backend...
              </div>
            )}
          </div>
        </div>

        <div className="main-content">
          <div className="functions-section">
            <h2>🎯 Análisis Financiero Especializado</h2>
            
            <div className="functions-grid">
              {functions.map((func) => (
                <div 
                  key={func.id}
                  className={`function-card ${selectedFunction === func.id ? 'active' : ''}`}
                  onClick={() => handleFunctionClick(func.id)}
                  style={{
                    background: func.color,
                    cursor: loading ? 'wait' : 'pointer',
                    opacity: loading && selectedFunction !== func.id ? 0.7 : 1
                  }}
                >
                  <div className="function-icon">{func.icon}</div>
                  <div className="function-content">
                    <h3>{func.name}</h3>
                    <p>{func.description}</p>
                    {selectedFunction === func.id && loading && (
                      <div className="loading-spinner">
                        <div className="spinner"></div>
                        Procesando análisis...
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="chat-section">
            <div className="chat-header">
              <h3>💬 Consulta Personalizada</h3>
              <button 
                className="toggle-chat"
                onClick={() => setShowChat(!showChat)}
              >
                {showChat ? '🔼 Ocultar' : '🔽 Mostrar'}
              </button>
            </div>
            
            {showChat && (
              <div className="chat-interface">
                <div className="chat-input">
                  <input
                    type="text"
                    value={chatQuestion}
                    onChange={(e) => setChatQuestion(e.target.value)}
                    placeholder="Ejemplo: ¿Cuál es la liquidez de la empresa? ¿Cómo han evolucionado los ingresos?"
                    onKeyPress={(e) => e.key === 'Enter' && handleCustomQuestion()}
                    disabled={loading}
                  />
                  <button 
                    onClick={handleCustomQuestion}
                    disabled={loading || !chatQuestion.trim()}
                    className="send-button"
                  >
                    {loading ? '⏳' : '📤'}
                  </button>
                </div>
                <div className="chat-examples">
                  <p><strong>💡 Ejemplos de preguntas:</strong></p>
                  <button onClick={() => setChatQuestion('¿Cuáles son los principales riesgos financieros?')}>
                    💸 Riesgos financieros
                  </button>
                  <button onClick={() => setChatQuestion('¿Cómo ha evolucionado la rentabilidad?')}>
                    📈 Evolución rentabilidad
                  </button>
                  <button onClick={() => setChatQuestion('¿Cuál es la situación de liquidez?')}>
                    💰 Análisis liquidez
                  </button>
                </div>
              </div>
            )}
          </div>

          <div className="responses-section">
            <div className="responses-header">
              <h3>📋 Historial de Análisis ({responses.length})</h3>
              {responses.length > 0 && (
                <button onClick={clearHistory} className="clear-history">
                  🗑️ Limpiar Historial
                </button>
              )}
            </div>
            
            <div className="responses-container">
              {responses.length === 0 ? (
                <div className="no-responses">
                  <div className="no-responses-icon">📊</div>
                  <p>Haz clic en una funcionalidad o usa el chat para comenzar el análisis financiero</p>
                  <p className="no-responses-tip">💡 Tip: Cada análisis se guardará aquí para tu referencia</p>
                </div>
              ) : (
                responses.map((response) => (
                  <div 
                    key={response.id} 
                    className={`response-card ${response.isError ? 'error' : ''}`}
                  >
                    <div className="response-header">
                      <span className="response-function">
                        {functions.find(f => f.id === response.function)?.icon || '💬'} 
                        {functions.find(f => f.id === response.function)?.name || 'Consulta personalizada'}
                      </span>
                      <span className="response-time">{response.timestamp}</span>
                    </div>
                    <div className="response-question">
                      <strong>Pregunta:</strong> {response.question.length > 100 ? 
                        response.question.substring(0, 100) + '...' : response.question}
                    </div>
                    <div className="response-answer">
                      <strong>Respuesta:</strong> {response.answer.length > 200 ? 
                        response.answer.substring(0, 200) + '...' : response.answer}
                    </div>
                    <button 
                      className="view-full-response"
                      onClick={() => setModalResponse({
                        title: `${functions.find(f => f.id === response.function)?.icon || '💬'} ${functions.find(f => f.id === response.function)?.name || 'Consulta personalizada'}`,
                        content: response.answer,
                        timestamp: response.fullTimestamp,
                        question: response.question
                      })}
                    >
                      👁️ Ver completo
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>

          <div className="links-section">
            <h3>🔗 Enlaces del Sistema</h3>
            <div className="links-grid">
              <a href="http://127.0.0.1:8000" target="_blank" rel="noopener noreferrer">
                📡 API Backend
              </a>
              <a href="http://127.0.0.1:8000/docs" target="_blank" rel="noopener noreferrer">
                📚 Documentación API
              </a>
              <a href="http://127.0.0.1:8000/system-status" target="_blank" rel="noopener noreferrer">
                🔍 Estado del Sistema
              </a>
            </div>
          </div>
        </div>
      </header>

      {/* Modal para mostrar respuestas completas */}
      {modalResponse && (
        <div className="modal-overlay" onClick={() => setModalResponse(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>{modalResponse.title}</h2>
              <button 
                className="modal-close" 
                onClick={() => setModalResponse(null)}
              >
                ✕
              </button>
            </div>
            
            {modalResponse.question && (
              <div className="modal-question">
                <strong>📝 Pregunta:</strong>
                <p>{modalResponse.question}</p>
              </div>
            )}
            
            <div className="modal-response">
              <strong>🤖 Respuesta del Sistema:</strong>
              <div className="modal-content-text">
                {modalResponse.content}
              </div>
            </div>
            
            <div className="modal-footer">
              <span className="modal-timestamp">
                📅 {modalResponse.timestamp}
              </span>
              <button 
                className="modal-copy"
                onClick={() => {
                  navigator.clipboard.writeText(modalResponse.content);
                  alert('📋 Respuesta copiada al portapapeles');
                }}
              >
                📋 Copiar respuesta
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
