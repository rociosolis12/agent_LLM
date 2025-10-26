// src/services/predictorApi.ts

import axios, { AxiosInstance, AxiosError } from 'axios';
import type {
  PredictionData,
  PipelineStatus,
  Recommendations,
  RunAnalysisRequest
} from '../types/predictor.types';

interface ApiResponse<T = any> {
  status: 'success' | 'error';
  message?: string;
  error?: string;
  error_details?: string;
  analysis?: T;
}

class PredictorApiService {
  private api: AxiosInstance;
  private analysisApi: AxiosInstance;

  constructor() {
    const baseURL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

    // API principal con timeout estándar (60 segundos)
    this.api = axios.create({
      baseURL,
      headers: { 'Content-Type': 'application/json' },
      timeout: 60000, // 60 segundos para GET requests
    });

    // API específica para análisis con timeout extendido (10 minutos)
    this.analysisApi = axios.create({
      baseURL,
      headers: { 'Content-Type': 'application/json' },
      timeout: 600000, // 10 minutos para análisis
    });

    // Interceptor de requests
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

    // Interceptor de responses con manejo de errores mejorado
    this.api.interceptors.response.use(
      (response) => {
        console.log('✅ API Response:', response.status, response.config.url);
        return response;
      },
      (error: AxiosError) => {
        return Promise.reject(this.handleAxiosError(error));
      }
    );

    // Mismo interceptor para analysisApi
    this.analysisApi.interceptors.request.use(
      (config) => {
        console.log('🚀 Analysis Request:', config.method?.toUpperCase(), config.url);
        console.log('⏱️ Timeout configurado:', config.timeout, 'ms');
        return config;
      },
      (error) => {
        console.error('❌ Analysis Request Error:', error);
        return Promise.reject(error);
      }
    );

    this.analysisApi.interceptors.response.use(
      (response) => {
        console.log('✅ Analysis Response:', response.status, response.config.url);
        return response;
      },
      (error: AxiosError) => {
        return Promise.reject(this.handleAxiosError(error));
      }
    );
  }

  /**
   * Maneja errores de Axios de manera uniforme
   */
  private handleAxiosError(error: AxiosError): Error {
    console.error('❌ Axios Error:', {
      message: error.message,
      code: error.code,
      status: error.response?.status,
      data: error.response?.data,
    });

    // Timeout error
    if (error.code === 'ECONNABORTED' || error.message.includes('timeout')) {
      return new Error(
        '⏱️ El análisis excedió el tiempo límite de 10 minutos. ' +
        'Esto puede deberse a:\n' +
        '• El análisis es muy complejo\n' +
        '• El servidor está sobrecargado\n' +
        '• Rate limits de Azure OpenAI\n\n' +
        'Intenta nuevamente en unos momentos.'
      );
    }

    // Network error
    if (error.message === 'Network Error' || !error.response) {
      return new Error(
        '🔌 No se pudo conectar al servidor.\n' +
        'Verifica que:\n' +
        '• El backend esté ejecutándose en http://localhost:8000\n' +
        '• No haya firewall bloqueando la conexión\n' +
        '• El servidor no esté caído'
      );
    }

    // API error con respuesta
    if (error.response?.data) {
      const data = error.response.data as ApiResponse;
      return new Error(
        data.message || 
        data.error_details || 
        data.error || 
        `Error HTTP ${error.response.status}`
      );
    }

    // Error genérico
    return new Error(error.message || 'Error desconocido en la petición');
  }

  /**
   * Ejecuta el análisis híbrido completo (con timeout de 10 minutos)
   */
  async runHybridAnalysis(request: RunAnalysisRequest): Promise<ApiResponse> {
    console.log('🚀 Ejecutando análisis híbrido...');
    console.log('📦 Request data:', request);

    try {
      const response = await this.analysisApi.post<ApiResponse>(
        '/api/predictor/run-hybrid-analysis',
        request
      );

      console.log('📊 Analysis response:', response.data);

      // Verificar respuesta exitosa
      if (response.data.status === 'error') {
        throw new Error(
          response.data.message || 
          response.data.error || 
          'El análisis falló en el servidor'
        );
      }

      return response.data;

    } catch (error: any) {
      console.error('❌ Error en runHybridAnalysis:', error);
      throw error;
    }
  }

  /**
   * Obtiene las últimas predicciones
   * Nota: Este endpoint puede no existir, maneja el error gracefully
   */
  async getLatestPredictions(): Promise<PredictionData | null> {
    try {
      const response = await this.api.get<PredictionData>(
        '/api/predictor/predictions/latest'
      );
      return response.data;
    } catch (error: any) {
      console.warn(' No se pudieron cargar predicciones previas:', error.message);
      // No lanzar error, retornar null si no hay datos previos
      return null;
    }
  }

  /**
   * Obtiene el estado del pipeline
   */
  async getPipelineStatus(): Promise<PipelineStatus | null> {
    try {
      const response = await this.api.get<PipelineStatus>(
        '/api/predictor/pipeline/status'
      );
      return response.data;
    } catch (error: any) {
      console.warn(' No se pudo cargar estado del pipeline:', error.message);
      return null;
    }
  }

  /**
   * Obtiene las recomendaciones
   */
  async getRecommendations(): Promise<Recommendations | null> {
    try {
      const response = await this.api.get<Recommendations>(
        '/api/predictor/recommendations'
      );
      return response.data;
    } catch (error: any) {
      console.warn(' No se pudieron cargar recomendaciones:', error.message);
      return null;
    }
  }

  /**
   * Verifica el estado del servidor
   */
  async checkServerHealth(): Promise<boolean> {
    try {
      const response = await this.api.get('/api/system-status', {
        timeout: 5000, // 5 segundos para health check
      });
      
      console.log(' Server health check passed:', response.data);
      return response.status === 200;

    } catch (error) {
      console.error(' Server health check failed:', error);
      return false;
    }
  }
}

export const predictorApi = new PredictorApiService();
