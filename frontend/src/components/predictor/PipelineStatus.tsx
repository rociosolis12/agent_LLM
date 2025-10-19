// src/components/predictor/PipelineStatus.tsx

import React from 'react';
import type { PipelineStatus as PipelineStatusType } from '../../types/predictor.types';

interface PipelineStatusProps {
  status: PipelineStatusType;
}

export const PipelineStatus: React.FC<PipelineStatusProps> = ({ status }) => {
  return (
    <div className="pipeline-status">
      <div className="status-grid">
        <StatusItem label="ML Predictor" active={status.components.ml_predictor} />
        <StatusItem label="Validator" active={status.components.validator} />
        <StatusItem label="Hybrid Agent" active={status.components.hybrid_agent} />
        <StatusItem label="Regulatory" active={status.components.regulatory} />
      </div>
      {status.last_execution && (
        <div className="last-execution">
          Última ejecución: {new Date(status.last_execution).toLocaleString('es-ES')}
        </div>
      )}
    </div>
  );
};

const StatusItem: React.FC<{ label: string; active: boolean }> = ({ label, active }) => (
  <div className="status-item">
    <div className={`status-indicator ${active ? 'active' : 'inactive'}`}>
      {active ? '✓' : '○'}
    </div>
    <span>{label}</span>
  </div>
);
