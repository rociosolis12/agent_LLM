// src/components/predictor/RecommendationsList.tsx

import React from 'react';
import type { Recommendations } from '../../types/predictor.types';

interface RecommendationsListProps {
  data: Recommendations;
}

export const RecommendationsList: React.FC<RecommendationsListProps> = ({ data }) => {
  const hasRecommendations = 
    (data.strategic && data.strategic.length > 0) ||
    (data.tactical && data.tactical.length > 0) ||
    (data.risk_mitigation && data.risk_mitigation.length > 0);

  if (!hasRecommendations) {
    return (
      <div className="no-data-message">
        <p>💡 No hay recomendaciones disponibles</p>
        <p>Ejecuta un análisis híbrido para generar recomendaciones</p>
      </div>
    );
  }

  return (
    <div className="recommendations-list">
      {data.strategic && data.strategic.length > 0 && (
        <section className="recommendation-section">
          <h3>💼 Recomendaciones Estratégicas</h3>
          {data.strategic.map((rec, idx) => (
            <div key={idx} className="recommendation-card strategic">
              <div className="rec-header">
                <span className="rec-source">{rec.source}</span>
              </div>
              <p>{rec.insight}</p>
            </div>
          ))}
        </section>
      )}

      {data.tactical && data.tactical.length > 0 && (
        <section className="recommendation-section">
          <h3>🎯 Recomendaciones Tácticas</h3>
          {data.tactical.map((rec, idx) => (
            <div key={idx} className="recommendation-card tactical">
              <div className="rec-header">
                {rec.metric && <span className="rec-metric">{rec.metric}</span>}
                <span className="rec-source">{rec.source}</span>
              </div>
              <p>{rec.insight}</p>
            </div>
          ))}
        </section>
      )}

      {data.risk_mitigation && data.risk_mitigation.length > 0 && (
        <section className="recommendation-section">
          <h3>⚠️ Mitigación de Riesgos</h3>
          {data.risk_mitigation.map((rec, idx) => (
            <div key={idx} className="recommendation-card risk">
              {rec.priority && (
                <div className="rec-header">
                  <span className={`rec-priority priority-${rec.priority.toLowerCase()}`}>
                    Prioridad: {rec.priority}
                  </span>
                </div>
              )}
              <p>{rec.insight}</p>
              {rec.risk_factors && rec.risk_factors.length > 0 && (
                <ul className="risk-factors">
                  {rec.risk_factors.map((factor, i) => (
                    <li key={i}>{factor}</li>
                  ))}
                </ul>
              )}
            </div>
          ))}
        </section>
      )}
    </div>
  );
};
