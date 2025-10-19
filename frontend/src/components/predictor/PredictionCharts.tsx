// src/components/predictor/PredictionCharts.tsx

import React from 'react';
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import type { PredictionData } from '../../types/predictor.types';

interface PredictionChartsProps {
  data: PredictionData;
}

export const PredictionCharts: React.FC<PredictionChartsProps> = ({ data }) => {
  if (!data?.ml_predictions || data.ml_predictions.length === 0) {
    return (
      <div className="no-data-message">
        <p>📊 No hay predicciones disponibles</p>
        <p>Ejecuta un análisis híbrido para generar predicciones</p>
      </div>
    );
  }

  const chartData = data.ml_predictions;

  return (
    <div className="prediction-charts">
      <div className="chart-container">
        <h3>📈 Predicción de ROA con Intervalos de Confianza</h3>
        <ResponsiveContainer width="100%" height={350}>
          <AreaChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="period" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Area type="monotone" dataKey="ROA_upper" stroke="#82ca9d" fill="#82ca9d" fillOpacity={0.2} name="Superior" />
            <Area type="monotone" dataKey="ROA" stroke="#8884d8" fill="#8884d8" strokeWidth={2} name="ROA" />
            <Area type="monotone" dataKey="ROA_lower" stroke="#ffc658" fill="#ffc658" fillOpacity={0.2} name="Inferior" />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="charts-grid">
        <div className="chart-item">
          <h4>Ratio de Solvencia</h4>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="period" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="ratio_solvencia" stroke="#8884d8" strokeWidth={2} name="Solvencia (%)" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-item">
          <h4>Liquidez</h4>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="period" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="liquidez" fill="#82ca9d" name="Liquidez" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
