// src/components/predictor/RecommendationsList.tsx

import React from 'react';
import type { Recommendations } from '../../types/predictor.types';

interface RecommendationsListProps {
  data: Recommendations | null | undefined; // ✅ Acepta null/undefined
}

export const RecommendationsList: React.FC<RecommendationsListProps> = ({ data }) => {
  // ===== AGREGAR ESTE LOG AL INICIO =====
  console.log('📋 RecommendationsList recibió data:', data);
  console.log('📋 Type of data:', typeof data);
  console.log('📋 Data keys:', data ? Object.keys(data) : 'null');
  // ===== FIN DE LOGS =====

  // Si no hay data, mostrar mensaje
  if (!data) {
    console.log('⚠️ RecommendationsList: data es null/undefined');
    return (
      <div className="no-recommendations">
        <div className="info-icon">💡</div>
        <h3>No hay recomendaciones disponibles</h3>
        <p>Ejecuta un análisis híbrido para generar recomendaciones</p>
      </div>
    );
  }

  // ===== AGREGAR ESTE LOG ANTES DE hasRecommendations =====
  console.log('🔍 Verificando arrays:', {
    strategic: data.strategic,
    strategic_length: data.strategic?.length,
    tactical: data.tactical,
    tactical_length: data.tactical?.length,
    risk_mitigation: data.risk_mitigation,
    risk_length: data.risk_mitigation?.length
  });
  // ===== FIN DE LOG =====

  const hasRecommendations =
    (data.strategic && data.strategic.length > 0) ||
    (data.tactical && data.tactical.length > 0) ||
    (data.risk_mitigation && data.risk_mitigation.length > 0);

  // ===== AGREGAR ESTE LOG =====
  console.log('🔍 hasRecommendations:', hasRecommendations);
  // ===== FIN DE LOG =====

  if (!hasRecommendations) {
    console.log('⚠️ RecommendationsList: No hay recomendaciones para mostrar');
    return (
      <div className="no-recommendations">
        <div className="info-icon">💡</div>
        <h3>No hay recomendaciones disponibles</h3>
        <p>Ejecuta un análisis híbrido para generar recomendaciones</p>
      </div>
    );
  }

  // ===== AGREGAR ESTE LOG ANTES DEL RETURN =====
  console.log('✅ RecommendationsList: Renderizando recomendaciones');
  // ===== FIN DE LOG =====

  return (
    <div className="recommendations-list">
      {data.strategic && data.strategic.length > 0 && (
        <div className="recommendation-section">
          <h3 className="section-title">💼 Recomendaciones Estratégicas</h3>
          {data.strategic.map((rec, idx) => {
            // ===== AGREGAR ESTE LOG =====
            console.log(`📝 Renderizando strategic ${idx}:`, rec);
            // ===== FIN DE LOG =====
            
            return (
              <div key={idx} className="recommendation-card">
                <div className="recommendation-source">{rec.source}</div>
                <div className="recommendation-content">{rec.insight}</div>
              </div>
            );
          })}
        </div>
      )}

      {data.tactical && data.tactical.length > 0 && (
        <div className="recommendation-section">
          <h3 className="section-title">🎯 Recomendaciones Tácticas</h3>
          {data.tactical.map((rec, idx) => (
            <div key={idx} className="recommendation-card">
              {rec.metric && <div className="recommendation-metric">{rec.metric}</div>}
              <div className="recommendation-source">{rec.source}</div>
              <div className="recommendation-content">{rec.insight}</div>
            </div>
          ))}
        </div>
      )}

      {data.risk_mitigation && data.risk_mitigation.length > 0 && (
        <div className="recommendation-section">
          <h3 className="section-title">⚠️ Mitigación de Riesgos</h3>
          {data.risk_mitigation.map((rec, idx) => (
            <div key={idx} className="recommendation-card risk">
              {rec.priority && (
                <div className="recommendation-priority">
                  Prioridad: {rec.priority}
                </div>
              )}
              <div className="recommendation-content">{rec.insight}</div>
              {rec.risk_factors && rec.risk_factors.length > 0 && (
                <ul className="risk-factors">
                  {rec.risk_factors.map((factor, i) => (
                    <li key={i}>{factor}</li>
                  ))}
                </ul>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
