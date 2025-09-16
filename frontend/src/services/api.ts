import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface QuestionRequest {
  question: string;
  include_predictions?: boolean;
  session_id?: string;
}

export interface PipelineRequest {
  include_predictions?: boolean;
  session_id?: string;
}

export const financialApi = {
  // Estado del sistema
  getSystemStatus: () => apiClient.get('/system-status'),
  
  // Subir PDF
  uploadPdf: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return apiClient.post('/upload-pdf', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
  },
  
  // Hacer pregunta
  askQuestion: (request: QuestionRequest) => 
    apiClient.post('/ask-question', request),
  
  // Ejecutar pipeline completo
  runPipeline: (request: PipelineRequest) =>
    apiClient.post('/run-pipeline', request),
  
  // Descargar resultados
  downloadResults: (sessionId: string) =>
    apiClient.get(`/download-results/${sessionId}`, { responseType: 'blob' }),
  
  // Sesiones activas
  getActiveSessions: () => apiClient.get('/sessions'),
};
