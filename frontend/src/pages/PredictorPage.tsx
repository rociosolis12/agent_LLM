// src/pages/PredictorPage.tsx

import React from 'react';
import { HybridPredictorDashboard } from '../components/predictor/HybridPredictorDashboard';

export const PredictorPage: React.FC = () => {
  return (
    <div className="page-container">
      <HybridPredictorDashboard bankSymbol="BBVA.MC" />
    </div>
  );
};
