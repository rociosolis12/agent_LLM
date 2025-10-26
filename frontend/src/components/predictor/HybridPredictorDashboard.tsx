// src/components/predictor/HybridPredictorDashboard.tsx

import React, { useState } from 'react';
import { usePredictorData } from '../../hooks/usePredictorData';
import { PredictionCharts } from './PredictionCharts';
import { ValidationMetrics } from './ValidationMetrics';
import { RecommendationsList } from './RecommendationsList';
import { PipelineStatus } from './PipelineStatus';
import './HybridPredictorDashboard.css';

interface HybridPredictorDashboardProps {
  bankSymbol?: string;
}

export const HybridPredictorDashboard: React.FC<HybridPredictorDashboardProps> = ({
  bankSymbol = 'BBVA.MC',
}) => {
  const {
    predictions,
    pipelineStatus,
    recommendations,
    loading,
    error,
    loadData,
    runAnalysis,
  } = usePredictorData(true);

  console.log('🔍 HybridPredictorDashboard render:', {
  predictions,
  predictionsType: typeof predictions,
  predictionsIsNull: predictions === null,
  predictionsKeys: predictions ? Object.keys(predictions) : 'no keys',
  mlPredictionsLength: predictions?.ml_predictions?.length,
  confidenceLevel: predictions?.confidence_level,
  pipelineStatus,
  recommendations,
  recsStrategic: recommendations?.strategic,
  recsTactical: recommendations?.tactical,
  loading,
  error
});

  const [activeTab, setActiveTab] = useState<'overview' | 'predictions' | 'validation' | 'recommendations'>('overview');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState<string>('');

  const handleRunAnalysis = async () => {
    setIsAnalyzing(true);
    setAnalysisProgress('🔄 Iniciando análisis predictor híbrido...');

    try {
      // Simular progreso (opcional - puedes implementar SSE para progreso real)
      const progressInterval = setInterval(() => {
        const progressMessages = [
          '📄 Extrayendo datos financieros del PDF...',
          '🧮 Generando predicciones con Prophet y XGBoost...',
          '✅ Ejecutando validación walk-forward...',
          '🤖 Analizando con LLM híbrido...',
          '📊 Aplicando configuración regulatoria...',
        ];
        const randomMessage = progressMessages[Math.floor(Math.random() * progressMessages.length)];
        setAnalysisProgress(randomMessage);
      }, 3000);

      await runAnalysis(bankSymbol);
      
      clearInterval(progressInterval);
      setAnalysisProgress('✅ Análisis completado exitosamente');
      
      // Mostrar notificación de éxito
      setTimeout(() => {
        setAnalysisProgress('');
      }, 3000);

    } catch (err) {
      console.error('Error en handleRunAnalysis:', err);
      setAnalysisProgress('');
      // El error ya está manejado en el hook y se muestra en la UI
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="hybrid-predictor-dashboard">
      {/* Header */}
      <div className="dashboard-header">
        <h1>🚀 Predictor Híbrido - {bankSymbol}</h1>
        <div className="header-actions">
          <button 
            onClick={loadData} 
            disabled={loading || isAnalyzing}
            className="btn-secondary"
          >
            🔄 {loading ? 'Actualizando...' : 'Actualizar'}
          </button>
          <button 
            onClick={handleRunAnalysis} 
            disabled={loading || isAnalyzing}
            className="btn-primary"
          >
            {isAnalyzing ? '⏳ Ejecutando...' : '▶️ Ejecutar Análisis'}
          </button>
        </div>
      </div>

      {/* Progress Indicator */}
      {isAnalyzing && analysisProgress && (
        <div className="progress-banner">
          <div className="progress-spinner"></div>
          <span>{analysisProgress}</span>
          <span className="progress-time">⏱️ Esto puede tardar 3-5 minutos...</span>
        </div>
      )}

      {/* Pipeline Status */}
      {pipelineStatus && <PipelineStatus status={pipelineStatus} />}

      {/* Error Display */}
      {error && (
        <div className="error-banner">
          <div className="error-content">
            <span className="error-icon">⚠️</span>
            <span className="error-message">{error}</span>
          </div>
          <div className="error-actions">
            <button onClick={loadData} className="btn-retry">
              🔄 Reintentar
            </button>
            <button onClick={() => window.location.reload()} className="btn-reload">
              ↻ Recargar Página
            </button>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="dashboard-tabs">
        <button
          className={`tab ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          📊 Resumen
        </button>
        <button
          className={`tab ${activeTab === 'predictions' ? 'active' : ''}`}
          onClick={() => setActiveTab('predictions')}
        >
          📈 Predicciones ML
        </button>
        <button
          className={`tab ${activeTab === 'validation' ? 'active' : ''}`}
          onClick={() => setActiveTab('validation')}
        >
          ✅ Validación
        </button>
        <button
          className={`tab ${activeTab === 'recommendations' ? 'active' : ''}`}
          onClick={() => setActiveTab('recommendations')}
        >
          💡 Recomendaciones
        </button>
      </div>

      {/* Content */}
      <div className="dashboard-content">
        {loading && !isAnalyzing && (
          <div className="loading-state">
            <div className="spinner"></div>
            <p>⏳ Cargando datos...</p>
          </div>
        )}

        {!loading && (
          <>
            {activeTab === 'overview' && (
              <div className="overview-grid">
                <div className="metric-card">
                  <h3>Nivel de Confianza</h3>
                  <div className="metric-value">
                    {predictions?.confidence_level 
                      ? `${(predictions.confidence_level * 100).toFixed(1)}%` 
                      : 'N/A'}
                  </div>
                </div>
                <div className="metric-card">
                  <h3>Predicciones ML</h3>
                  <div className="metric-value">
                    {predictions?.ml_predictions?.length || 0}
                  </div>
                </div>
                <div className="metric-card">
                  <h3>Métricas Validadas</h3>
                  <div className="metric-value">
                    {Object.keys(predictions?.validation_results || {}).length}
                  </div>
                </div>
                <div className="metric-card">
                  <h3>Recomendaciones</h3>
                  <div className="metric-value">
                    {(recommendations?.strategic?.length || 0) +
                      (recommendations?.tactical?.length || 0)}
                  </div>
                </div>

                {predictions && <PredictionCharts data={predictions} />}
              </div>
            )}

            {activeTab === 'predictions' && predictions && (
              <PredictionCharts data={predictions} />
            )}

            {activeTab === 'validation' && predictions?.validation_results && (
              <ValidationMetrics metrics={predictions.validation_results} />
            )}

            {activeTab === 'recommendations' && recommendations && (
              <RecommendationsList recommendations={recommendations} />
            )}

            {/* Empty state */}
            {!predictions && !loading && !isAnalyzing && (
              <div className="empty-state">
                <div className="empty-icon">📊</div>
                <h3>No hay datos disponibles</h3>
                <p>Haz clic en "Ejecutar Análisis" para generar predicciones</p>
                <button onClick={handleRunAnalysis} className="btn-primary">
                  ▶️ Ejecutar Análisis Ahora
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};
