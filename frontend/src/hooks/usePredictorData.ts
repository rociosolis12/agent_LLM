// src/hooks/usePredictorData.ts

import { useState, useEffect, useCallback } from 'react';
import { predictorApi } from '../services/predictorApi';
import type { PredictionData, PipelineStatus, Recommendations } from '../types/predictor.types';

export const usePredictorData = (autoLoad: boolean = true) => {
  // ✅ CAMBIO CRÍTICO: loading comienza en true para evitar render antes de datos
  const [predictions, setPredictions] = useState<PredictionData | null>(null);
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatus | null>(null);
  const [recommendations, setRecommendations] = useState<Recommendations | null>(null);
  const [loading, setLoading] = useState<boolean>(true); // ⚠️ CAMBIO: true en lugar de false
  const [error, setError] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    console.log('🔄 Loading predictor data...');
    setLoading(true);
    setError(null);

    try {
      // Cargar datos en paralelo con manejo individual de errores
      const [predsData, statusData, recsData] = await Promise.all([
        predictorApi.getLatestPredictions().catch((err) => {
          console.log('ℹ️ No predictions data available yet:', err.message);
          return null;
        }),
        predictorApi.getPipelineStatus().catch((err) => {
          console.log('ℹ️ No pipeline status available yet:', err.message);
          return null;
        }),
        predictorApi.getRecommendations().catch((err) => {
          console.log('ℹ️ No recommendations available yet:', err.message);
          return null;
        }),
      ]);

      console.log('📊 Data loaded:', {
        predictions: predsData ? 'loaded' : 'null',
        pipeline: statusData ? 'loaded' : 'null',
        recommendations: recsData ? 'loaded' : 'null'
      });

      // ✅ Actualizar estados de forma atómica
      setPredictions(predsData);
      setPipelineStatus(statusData);
      setRecommendations(recsData);

      console.log('✅ State updated successfully');

    } catch (err: any) {
      console.error('❌ Error loading data:', err);
      // Mantener estados como null pero definidos
      setPredictions(null);
      setPipelineStatus(null);
      setRecommendations(null);
    } finally {
      // ✅ Solo marcar loading false cuando TODO esté listo
      setLoading(false);
    }
  }, []);

  const runAnalysis = useCallback(async (bankSymbol: string = 'BBVA.MC') => {
    console.log(`🚀 Running analysis for ${bankSymbol}...`);
    setLoading(true);
    setError(null);

    try {
      // Verificar salud del servidor
      const serverHealthy = await predictorApi.checkServerHealth();
      
      if (!serverHealthy) {
        throw new Error(
          'El servidor no está disponible. Verifica que el backend esté ejecutándose en http://localhost:8000'
        );
      }

      console.log('✅ Server is healthy, starting analysis...');

      // Ejecutar análisis
      const result = await predictorApi.runHybridAnalysis({
        bank_symbol: bankSymbol,
        generate_new_predictions: true,
      });

      if (result.status === 'success') {
        console.log('✅ Analysis completed successfully');
        console.log('📊 Analysis result:', result);

        // ✅ IMPORTANTE: Aumentar el tiempo de espera a 3 segundos
        // para que los archivos JSON se escriban completamente
        console.log('⏳ Waiting 3 seconds before loading results...');
        await new Promise(resolve => setTimeout(resolve, 3000));

        // Recargar datos desde los archivos JSON generados
        console.log('🔄 Loading fresh data from server...');
        await loadData();
        console.log('✅ Data refresh completed');

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
      console.log('🎬 Component mounted, loading data...');
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
