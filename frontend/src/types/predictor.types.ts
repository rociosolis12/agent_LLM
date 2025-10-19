export interface PredictionDataPoint {
  period: string;
  ROA: number;
  ROA_lower: number;
  ROA_upper: number;
  ratio_solvencia: number;
  liquidez: number;
  [key: string]: any;
}

export interface ValidationMetric {
  mae: number;
  rmse: number;
  r2: number;
  status?: string;
}

export interface Recommendation {
  source: string;
  insight: string;
  metric?: string;
  priority?: string;
  forecast_trend?: any;
  risk_factors?: string[];
}

export interface Recommendations {
  strategic: Recommendation[];
  tactical: Recommendation[];
  risk_mitigation: Recommendation[];
}

export interface PredictionData {
  ml_predictions: PredictionDataPoint[];
  validation_results: Record<string, ValidationMetric>;
  recommendations: Recommendations;
  confidence_level: 'HIGH' | 'MEDIUM' | 'LOW';
  timestamp: string;
}

export interface PipelineStatus {
  status: 'idle' | 'running' | 'completed' | 'error';
  components: {
    ml_predictor: boolean;
    validator: boolean;
    hybrid_agent: boolean;
    regulatory: boolean;
  };
  last_execution: string | null;
  hybrid_predictor_available: boolean;
}

export interface RunAnalysisRequest {
  pdf_path?: string;
  bank_symbol?: string;
  generate_new_predictions?: boolean;
  question?: string;
}
