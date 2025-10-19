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

  const [activeTab, setActiveTab] = useState<'overview' | 'predictions' | 'validation' | 'recommendations'>('overview');
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleRunAnalysis = async () => {
    setIsAnalyzing(true);
    try {
      await runAnalysis(bankSymbol);
      alert('✅ Análisis híbrido completado exitosamente');
    } catch (err) {
      alert('❌ Error ejecutando análisis: ' + (err as Error).message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="hybrid-predictor-dashboard">
      {/* Header */}
      <div className="dashboard-header">
        <div className="header-content">
          <h1>🚀 Predictor Híbrido - {bankSymbol}</h1>
          <div className="header-actions">
            <button
              onClick={loadData}
              disabled={loading}
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

        {/* Pipeline Status */}
        {pipelineStatus && <PipelineStatus status={pipelineStatus} />}
      </div>

      {/* Error Display */}
      {error && (
        <div className="error-banner">
          <span>⚠️ {error}</span>
          <button onClick={() => window.location.reload()}>Recargar</button>
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
        {loading && (
          <div className="loading-spinner">
            ⏳ Cargando datos...
          </div>
        )}

        {!loading && (
          <>
            {activeTab === 'overview' && (
              <div className="overview-content">
                <div className="metrics-grid">
                  <div className="metric-card">
                    <h3>Nivel de Confianza</h3>
                    <div className={`metric-value confidence-${predictions?.confidence_level?.toLowerCase()}`}>
                      {predictions?.confidence_level || 'N/A'}
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
                </div>

                {predictions && <PredictionCharts data={predictions} />}
              </div>
            )}

            {activeTab === 'predictions' && predictions && (
              <PredictionCharts data={predictions} />
            )}

            {activeTab === 'validation' && predictions?.validation_results && (
              <ValidationMetrics data={predictions.validation_results} />
            )}

            {activeTab === 'recommendations' && recommendations && (
              <RecommendationsList data={recommendations} />
            )}
          </>
        )}
      </div>
    </div>
  );
};
