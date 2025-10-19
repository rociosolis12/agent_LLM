// src/components/predictor/ValidationMetrics.tsx

import React from 'react';
import type { ValidationMetric } from '../../types/predictor.types';

interface ValidationMetricsProps {
  data: Record<string, ValidationMetric>;
}

export const ValidationMetrics: React.FC<ValidationMetricsProps> = ({ data }) => {
  if (!data || Object.keys(data).length === 0) {
    return (
      <div className="no-data-message">
        <p>📊 No hay métricas de validación disponibles</p>
      </div>
    );
  }

  const getStatus = (r2: number): string => {
    if (r2 > 0.8) return 'excellent';
    if (r2 > 0.6) return 'good';
    return 'fair';
  };

  const getStatusLabel = (r2: number): string => {
    if (r2 > 0.8) return 'Excelente';
    if (r2 > 0.6) return 'Bueno';
    return 'Regular';
  };

  return (
    <div className="validation-metrics">
      <h3>✅ Métricas de Validación Walk-Forward</h3>
      <table className="metrics-table">
        <thead>
          <tr>
            <th>Métrica</th>
            <th>MAE</th>
            <th>RMSE</th>
            <th>R²</th>
            <th>Estado</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(data).map(([metric, values]) => (
            <tr key={metric}>
              <td><strong>{metric}</strong></td>
              <td>{values.mae?.toFixed(4) || 'N/A'}</td>
              <td>{values.rmse?.toFixed(4) || 'N/A'}</td>
              <td>{values.r2?.toFixed(4) || 'N/A'}</td>
              <td>
                <span className={`status-badge ${getStatus(values.r2)}`}>
                  {getStatusLabel(values.r2)}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
