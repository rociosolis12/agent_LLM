// src/hooks/usePredictorData.ts

import { useState, useEffect, useCallback } from 'react';
import { predictorApi } from '../services/predictorApi';
import type { PredictionData, PipelineStatus, Recommendations } from '../types/predictor.types';

export const usePredictorData = (autoLoad: boolean = true) => {
  const [predictions, setPredictions] = useState<PredictionData | null>(null);
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatus | null>(null);
  const [recommendations, setRecommendations] = useState<Recommendations | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    console.log('🔄 Loading predictor data...');
    setLoading(true);
    setError(null);
    
    try {
      const [predsData, statusData, recsData] = await Promise.all([
        predictorApi.getLatestPredictions().catch(() => null),
        predictorApi.getPipelineStatus().catch(() => null),
        predictorApi.getRecommendations().catch(() => null),
      ]);
      
      setPredictions(predsData);
      setPipelineStatus(statusData);
      setRecommendations(recsData);
      
      console.log('✅ Data loaded successfully');
    } catch (err: any) {
      console.error('❌ Error loading data:', err);
      setError(err.message || 'Error loading data');
    } finally {
      setLoading(false);
    }
  }, []);

  const runAnalysis = useCallback(async (bankSymbol: string = 'BBVA.MC') => {
    console.log(`🚀 Running analysis for ${bankSymbol}...`);
    setLoading(true);
    setError(null);
    
    try {
      const result = await predictorApi.runHybridAnalysis({
        bank_symbol: bankSymbol,
        generate_new_predictions: true,
      });
      
      if (result.status === 'success') {
        console.log('✅ Analysis completed');
        await loadData();
        return result;
      } else {
        throw new Error(result.message || 'Analysis failed');
      }
    } catch (err: any) {
      console.error('❌ Error running analysis:', err);
      setError(err.message || 'Error running analysis');
      throw err;
    } finally {
      setLoading(false);
    }
  }, [loadData]);

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
