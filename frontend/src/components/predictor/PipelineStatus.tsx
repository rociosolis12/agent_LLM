// src/components/predictor/PipelineStatus.tsx

import React from 'react';
import type { PipelineStatus as PipelineStatusType } from '../../types/predictor.types';

interface PipelineStatusProps {
  status: PipelineStatusType;
}

export const PipelineStatus: React.FC<PipelineStatusProps> = ({ status }) => {
  // Usar optional chaining para evitar errores
  const hybridPredictorActive = status?.hybrid_predictor?.status === 'active';
  const mainPredictorActive = status?.main_predictor?.status === 'active';
  const validationActive = status?.validation_module?.status === 'active';

  return (
    <div className="pipeline-status-container">
      <h3>Estado del Pipeline</h3>
      <div className="status-grid">
        <StatusItem 
          label="Hybrid Predictor" 
          active={hybridPredictorActive} 
        />
        <StatusItem 
          label="Main Predictor" 
          active={mainPredictorActive} 
        />
        <StatusItem 
          label="Validation Module" 
          active={validationActive} 
        />
      </div>
      {status?.last_execution && (
        <div className="last-execution">
          Última ejecución: {new Date(status.last_execution).toLocaleString('es-ES')}
        </div>
      )}
    </div>
  );
};

const StatusItem: React.FC<{ label: string; active: boolean }> = ({ label, active }) => (
  <div className="status-item">
    <span className="status-icon">{active ? '✓' : '○'}</span>
    <span className="status-label">{label}</span>
  </div>
);
