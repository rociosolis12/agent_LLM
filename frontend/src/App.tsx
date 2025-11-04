// src/App.tsx
import React, { useState, useEffect } from 'react';
import './App.css';
import Dashboard from './components/Dashboard/Dashboard';
import { HybridPredictorDashboard } from './components/predictor/HybridPredictorDashboard';

export default function App() {
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
      <header className="app-header">
        <h1>Financial Analysis System</h1>
        <nav className="nav-tabs">
          <button
            className={currentSection === 'dashboard' ? 'tab active' : 'tab'}
            onClick={() => setCurrentSection('dashboard')}
          >
            Dashboard
          </button>
          <button
            className={currentSection === 'predictor' ? 'tab active' : 'tab'}
            onClick={() => setCurrentSection('predictor')}
          >
            Predictor
          </button>
        </nav>
        <div className="status-indicator">
          {systemStatus ? (
            <span className="status-online">● Sistema en línea</span>
          ) : (
            <span className="status-offline">● Sistema offline</span>
          )}
        </div>
      </header>

      <main className="app-main">
        {currentSection === 'dashboard' && <Dashboard />}
        {currentSection === 'predictor' && <HybridPredictorDashboard />}
      </main>
    </div>
  );
}
