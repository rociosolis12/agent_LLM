// src/hooks/usePredictorData.ts

import { useState, useEffect, useCallback } from 'react';
import { predictorApi } from '../services/predictorApi';
import type { PredictionData, PipelineStatus, Recommendations } from '../types/predictor.types';

export const usePredictorData = (autoLoad: boolean = true) => {
  const [predictions, setPredictions] = useState<PredictionData | null>(null);
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatus | null>(null);
  const [recommendations, setRecommendations] = useState<Recommendations | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    console.log('🔄 Loading predictor data...');
    setLoading(true);
    setError(null);

    try {
      // Intentar cargar datos existentes (pueden no existir aún)
      const [predsData, statusData, recsData] = await Promise.all([
        predictorApi.getLatestPredictions().catch(() => {
          console.log('ℹ️ No predictions data available yet');
          return null;
        }),
        predictorApi.getPipelineStatus().catch(() => {
          console.log('ℹ️ No pipeline status available yet');
          return null;
        }),
        predictorApi.getRecommendations().catch(() => {
          console.log('ℹ️ No recommendations available yet');
          return null;
        }),
      ]);

      setPredictions(predsData);
      setPipelineStatus(statusData);
      setRecommendations(recsData);
      
      console.log('✅ Data loaded successfully');

    } catch (err: any) {
      console.error('❌ Error loading data:', err);
      // No establecer error aquí, solo en runAnalysis
      // porque la carga inicial puede fallar si no hay datos previos
    } finally {
      setLoading(false);
    }
  }, []);

  const runAnalysis = useCallback(async (bankSymbol: string = 'BBVA.MC') => {
    console.log(`🚀 Running analysis for ${bankSymbol}...`);
    setLoading(true);
    setError(null);

    try {
      // Verificar salud del servidor primero
      const serverHealthy = await predictorApi.checkServerHealth();
      
      if (!serverHealthy) {
        throw new Error(
          'El servidor no está disponible. Verifica que el backend esté ejecutándose en http://localhost:8000'
        );
      }

      // Ejecutar análisis
      const result = await predictorApi.runHybridAnalysis({
        bank_symbol: bankSymbol,
        generate_new_predictions: true,
      });

      if (result.status === 'success') {
        console.log('✅ Analysis completed successfully');
        
        // Actualizar estado con los resultados del análisis
        if (result.analysis) {
          setPredictions(result.analysis.predictions || null);
          setPipelineStatus(result.analysis.pipeline_status || null);
          setRecommendations(result.analysis.recommendations || null);
        } else {
          // Cargar datos desde archivos
          await loadData();
        }
        
        return result;
      } else {
        throw new Error(result.message || 'Analysis failed');
      }

    } catch (err: any) {
      console.error('❌ Error running analysis:', err);
      const errorMessage = err.message || 'Error ejecutando análisis';
      setError(errorMessage);
      throw err;

    } finally {
      setLoading(false);
    }
  }, [loadData]);

  // Cargar datos automáticamente al montar
  useEffect(() => {
    if (autoLoad) {
      loadData();
    }
  }, [autoLoad, loadData]);

  return {
    predictions,
    pipelineStatus,
    recommendations,
    loading,
    error,
    loadData,
    runAnalysis,
  };
};
