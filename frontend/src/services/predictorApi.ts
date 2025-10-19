// src/services/predictorApi.ts

import axios, { AxiosInstance } from 'axios';
import type { 
  PredictionData, 
  PipelineStatus, 
  Recommendations,
  RunAnalysisRequest 
} from '../types/predictor.types';

class PredictorApiService {
  private api: AxiosInstance;

  constructor() {
    const baseURL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
    
    this.api = axios.create({
      baseURL,
      headers: { 'Content-Type': 'application/json' },
      timeout: 60000,
    });

    this.api.interceptors.request.use(
      (config) => {
        console.log('🚀 API Request:', config.method?.toUpperCase(), config.url);
        return config;
      },
      (error) => {
        console.error('❌ Request Error:', error);
        return Promise.reject(error);
      }
    );

    this.api.interceptors.response.use(
      (response) => {
        console.log('✅ API Response:', response.status, response.config.url);
        return response;
      },
      (error) => {
        console.error('❌ Response Error:', error.response?.status);
        return Promise.reject(error);
      }
    );
  }

  async runHybridAnalysis(request: RunAnalysisRequest): Promise<any> {
    try {
      const response = await this.api.post('/api/predictor/run-hybrid-analysis', request);
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Error ejecutando análisis');
    }
  }

  async getLatestPredictions(): Promise<PredictionData> {
    try {
      const response = await this.api.get('/api/predictor/predictions/latest');
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Error obteniendo predicciones');
    }
  }

  async getPipelineStatus(): Promise<PipelineStatus> {
    try {
      const response = await this.api.get('/api/predictor/pipeline/status');
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Error obteniendo estado');
    }
  }

  async getRecommendations(): Promise<Recommendations> {
    try {
      const response = await this.api.get('/api/predictor/recommendations');
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Error obteniendo recomendaciones');
    }
  }
}

export const predictorApi = new PredictorApiService();
