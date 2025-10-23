// src/App.tsx
import React, { useState, useEffect } from 'react';
import './App.css';
import Dashboard from './components/Dashboard/Dashboard';
import { HybridPredictorDashboard } from './components/predictor/HybridPredictorDashboard';

function App() {
  const [systemStatus, setSystemStatus] = useState(null);
  const [currentSection, setCurrentSection] = useState('dashboard'); // 'dashboard' o 'predictor'

  // Verificar estado del backend al cargar
  useEffect(() => {
    checkBackendStatus();
    const interval = setInterval(checkBackendStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const checkBackendStatus = async () => {
    try {
      const response = await fetch('http://localhost:8000/system-status');
      const data = await response.json();
      setSystemStatus(data);
    } catch (error) {
      console.log('Backend no disponible:', error);
      setSystemStatus(null);
    }
  };

  return (
    <div className="App">
      {/* Dashboard Principal - Solo las 5 tarjetas de agentes */}
      {currentSection === 'dashboard' && (
        <Dashboard onNavigateToPredictor={() => setCurrentSection('predictor')} />
      )}

      {/* Predictor Híbrido */}
      {currentSection === 'predictor' && (
        <div>
          <button 
            onClick={() => setCurrentSection('dashboard')}
            style={{
              position: 'fixed',
              top: '20px',
              left: '20px',
              padding: '10px 20px',
              backgroundColor: '#6366f1',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer',
              zIndex: 1000,
              fontWeight: 600,
              boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
            }}
          >
            ← Volver al Dashboard
          </button>
          <HybridPredictorDashboard />
        </div>
      )}
    </div>
  );
}

export default App;
