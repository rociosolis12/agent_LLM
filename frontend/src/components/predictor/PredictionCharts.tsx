// src/components/predictor/PredictionCharts.tsx

import React from 'react';
import {
  LineChart, Line, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import type { PredictionData } from '../../types/predictor.types';

interface PredictionChartsProps {
  data: PredictionData | null | undefined;
}

export const PredictionCharts: React.FC<PredictionChartsProps> = ({ data }) => {
  // Log para debugging
  console.log('📊 PredictionCharts data:', data);

  if (!data?.ml_predictions || data.ml_predictions.length === 0) {
    return (
      <div className="no-charts">
        <div className="info-icon">📊</div>
        <h3>No hay predicciones disponibles</h3>
        <p>Ejecuta un análisis híbrido para generar predicciones</p>
      </div>
    );
  }

  // ===== PROCESAR DATOS PARA RECHARTS =====
  // Agrupar predicciones por timestep
  const groupedData: { [key: number]: any } = {};
  
  data.ml_predictions.forEach((pred) => {
    const timestep = pred.timestep;
    
    if (!groupedData[timestep]) {
      groupedData[timestep] = { timestep };
    }
    
    const metric = pred.metric;
    groupedData[timestep][metric] = pred.prediction;
    groupedData[timestep][`${metric}_lower`] = pred.lower;
    groupedData[timestep][`${metric}_upper`] = pred.upper;
  });

  // Convertir a array y ordenar por timestep
  const chartData = Object.values(groupedData).sort((a, b) => a.timestep - b.timestep);
  
  console.log('📈 Chart data processed:', chartData);

  // Detectar qué métricas existen en los datos
  const availableMetrics = new Set<string>();
  data.ml_predictions.forEach((pred) => availableMetrics.add(pred.metric));
  
  const hasROA = availableMetrics.has('ROA');
  const hasSolvencia = availableMetrics.has('ratio_solvencia') || availableMetrics.has('solvencia');
  const hasLiquidez = availableMetrics.has('liquidez');

  return (
    <div className="charts-container">
      {/* Gráfico de ROA con intervalos de confianza */}
      {hasROA && chartData.length > 0 && (
        <div className="chart-section">
          <h3 className="chart-title">📈 Predicción de ROA con Intervalos de Confianza</h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="timestep" 
                label={{ value: 'Período', position: 'insideBottom', offset: -5 }}
              />
              <YAxis 
                label={{ value: 'ROA (%)', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip />
              <Legend />
              
              {/* Área del intervalo de confianza */}
              <Area
                type="monotone"
                dataKey="ROA_upper"
                stackId="1"
                stroke="#82ca9d"
                fill="#82ca9d"
                fillOpacity={0.2}
                name="Superior"
              />
              <Area
                type="monotone"
                dataKey="ROA_lower"
                stackId="1"
                stroke="#82ca9d"
                fill="#ffffff"
                fillOpacity={0}
                name="Inferior"
              />
              
              {/* Línea de predicción */}
              <Line
                type="monotone"
                dataKey="ROA"
                stroke="#8884d8"
                strokeWidth={3}
                name="ROA"
                dot={{ r: 5 }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Gráfico de Ratio de Solvencia */}
      {hasSolvencia && chartData.length > 0 && (
        <div className="chart-section">
          <h3 className="chart-title">Ratio de Solvencia</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="timestep" 
                label={{ value: 'Período', position: 'insideBottom', offset: -5 }}
              />
              <YAxis 
                label={{ value: 'Solvencia (%)', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip />
              <Legend />
              
              <Line
                type="monotone"
                dataKey="ratio_solvencia"
                stroke="#82ca9d"
                strokeWidth={2}
                name="Solvencia (%)"
                dot={{ r: 4 }}
              />
              <Line
                type="monotone"
                dataKey="solvencia"
                stroke="#82ca9d"
                strokeWidth={2}
                name="Solvencia (%)"
                dot={{ r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Gráfico de Liquidez */}
      {hasLiquidez && chartData.length > 0 && (
        <div className="chart-section">
          <h3 className="chart-title">Liquidez</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="timestep" 
                label={{ value: 'Período', position: 'insideBottom', offset: -5 }}
              />
              <YAxis 
                label={{ value: 'Liquidez', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip />
              <Legend />
              
              <Line
                type="monotone"
                dataKey="liquidez"
                stroke="#ffc658"
                strokeWidth={2}
                name="Liquidez"
                dot={{ r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Mensaje si no hay métricas */}
      {!hasROA && !hasSolvencia && !hasLiquidez && (
        <div className="no-charts">
          <div className="info-icon">📊</div>
          <h3>No hay métricas para graficar</h3>
          <p>Las predicciones no contienen métricas conocidas (ROA, solvencia, liquidez)</p>
        </div>
      )}
    </div>
  );
};
