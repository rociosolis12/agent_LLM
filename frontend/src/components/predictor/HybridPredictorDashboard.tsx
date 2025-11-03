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

  // ===== LOGS DE DEBUGGING =====
  console.log('🔍 HybridPredictorDashboard render:', {
    predictions,
    predictionsType: typeof predictions,
    predictionsIsNull: predictions === null,
    predictionsKeys: predictions ? Object.keys(predictions) : 'no keys',
    mlPredictionsLength: predictions?.ml_predictions?.length,
    confidenceLevel: predictions?.confidence_level,
    pipelineStatus,
    recommendations,
    recommendationsType: typeof recommendations,
    recommendationsIsNull: recommendations === null,
    recsStrategic: recommendations?.strategic,
    recsStrategicLength: recommendations?.strategic?.length,
    recsTactical: recommendations?.tactical,
    recsTacticalLength: recommendations?.tactical?.length,
    loading,
    error
  });

  const [activeTab, setActiveTab] = useState<'overview' | 'predictions' | 'validation' | 'recommendations'>('overview');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState('');

  const handleRunAnalysis = async () => {
    setIsAnalyzing(true);
    setAnalysisProgress('🔄 Iniciando análisis predictor híbrido...');

    try {
      // Simular progreso
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
      setAnalysisProgress('❌ Error en el análisis');
      setTimeout(() => {
        setAnalysisProgress('');
      }, 3000);
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
          <button onClick={loadData} disabled={loading} className="btn-update">
            🔄 {loading ? 'Actualizando...' : 'Actualizar'}
          </button>
          <button 
            onClick={handleRunAnalysis} 
            disabled={isAnalyzing} 
            className="btn-analyze"
          >
            {isAnalyzing ? '⏳ Ejecutando...' : '▶️ Ejecutar Análisis'}
          </button>
        </div>
      </div>

      {/* Progress Indicator */}
      {isAnalyzing && analysisProgress && (
        <div className="analysis-progress">
          <div className="progress-message">{analysisProgress}</div>
          <div className="progress-info">⏱️ Esto puede tardar 3-5 minutos...</div>
        </div>
      )}

      {/* Pipeline Status */}
      {pipelineStatus && <PipelineStatus status={pipelineStatus} />}

      {/* Error Display */}
      {error && (
        <div className="error-display">
          <span className="error-icon">⚠️</span>
          <span className="error-message">{error}</span>
          <button onClick={loadData} className="btn-retry">
            🔄 Reintentar
          </button>
          <button onClick={() => window.location.reload()} className="btn-reload">
            ↻ Recargar Página
          </button>
        </div>
      )}

      {/* Tabs */}
      <div className="dashboard-tabs">
        <button
          onClick={() => setActiveTab('overview')}
          className={activeTab === 'overview' ? 'active' : ''}
        >
          📊 Resumen
        </button>
        <button
          onClick={() => setActiveTab('predictions')}
          className={activeTab === 'predictions' ? 'active' : ''}
        >
          📈 Predicciones ML
        </button>
        <button
          onClick={() => setActiveTab('validation')}
          className={activeTab === 'validation' ? 'active' : ''}
        >
          ✅ Validación
        </button>
        <button
          onClick={() => setActiveTab('recommendations')}
          className={activeTab === 'recommendations' ? 'active' : ''}
        >
          💡 Recomendaciones
        </button>
      </div>

      {/* Content */}
      <div className="dashboard-content">
        {loading && !isAnalyzing && (
          <div className="loading-state">⏳ Cargando datos...</div>
        )}

        {!loading && (
          <>
            {/* TAB: OVERVIEW */}
            {activeTab === 'overview' && (
              <div className="overview-tab">
                <div className="metrics-grid">
                  <div className="metric-card">
                    <div className="metric-label">Nivel de Confianza</div>
                    <div className="metric-value">
                      {predictions?.confidence_level
                        ? `${(predictions.confidence_level * 100).toFixed(1)}%`
                        : 'N/A'}
                    </div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-label">Predicciones ML</div>
                    <div className="metric-value">
                      {predictions?.ml_predictions?.length || 0}
                    </div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-label">Métricas Validadas</div>
                    <div className="metric-value">
                      {Object.keys(predictions?.validation_results || {}).length}
                    </div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-label">Recomendaciones</div>
                    <div className="metric-value">
                      {(recommendations?.strategic?.length || 0) +
                        (recommendations?.tactical?.length || 0)}
                    </div>
                  </div>
                </div>

                {predictions && <PredictionCharts data={predictions} />}
              </div>
            )}

            {/* TAB: PREDICTIONS */}
            {activeTab === 'predictions' && (
              <>
                {!predictions ? (
                  <div className="empty-state">
                    📊 No hay predicciones disponibles
                    <p>Ejecuta un análisis para generar predicciones</p>
                  </div>
                ) : (
                  <PredictionCharts data={predictions} />
                )}
              </>
            )}

            {/* TAB: VALIDATION */}
            {activeTab === 'validation' && (
              <>
                {!predictions?.validation_results ? (
                  <div className="empty-state">
                    📊 No hay métricas de validación disponibles
                    <p>Ejecuta un análisis para generar métricas</p>
                  </div>
                ) : (
                  <ValidationMetrics data={predictions.validation_results} />
                )}
              </>
            )}

            {/* TAB: RECOMMENDATIONS - ✅ SECCIÓN CORREGIDA */}
            {activeTab === 'recommendations' && (
              <>
                {console.log('🎯 Renderizando tab recommendations:', {
                  recommendations,
                  recommendationsType: typeof recommendations,
                  isNull: recommendations === null,
                  isUndefined: recommendations === undefined
                })}
                
                {!recommendations || 
                 ((!recommendations?.strategic || recommendations.strategic.length === 0) &&
                  (!recommendations?.tactical || recommendations.tactical.length === 0)) ? (
                  <div className="empty-state">
                    💡 No hay recomendaciones disponibles
                    <p>Ejecuta un análisis para generar recomendaciones estratégicas y tácticas</p>
                  </div>
                ) : (
                  <RecommendationsList recommendations={recommendations} />
                )}
              </>
            )}

            {/* Empty state general */}
            {!predictions && !loading && !isAnalyzing && (
              <div className="empty-state-main">
                <div className="empty-icon">📊</div>
                <h3>No hay datos disponibles</h3>
                <p>Haz clic en "Ejecutar Análisis" para generar predicciones</p>
                <button onClick={handleRunAnalysis} className="btn-start-analysis">
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
