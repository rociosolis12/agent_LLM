import React, { useState } from 'react';
import {
  Box,
  TextField,
  Button,
  FormControlLabel,
  Switch,
  Typography,
  Paper,
  Chip,
  CircularProgress,
  Alert
} from '@mui/material';
import { Send as SendIcon, Psychology as PredictionIcon } from '@mui/icons-material';

import { financialApi } from '../../services/api';

const QuestionInterface: React.FC = () => {
  const [question, setQuestion] = useState('');
  const [includePredictions, setIncludePredictions] = useState(true);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setError(null);
    try {
      const response = await financialApi.askQuestion({
        question: question.trim(),
        include_predictions: includePredictions
      });
      setResult(response);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Error procesando la pregunta');
    } finally {
      setLoading(false);
    }
  };

  const suggestedQuestions = [
    "¿Cuál es el total de activos en 2023?",
    "¿Cuál fue el beneficio neto del año?",
    "¿Qué efectivo generaron las operaciones?",
    "¿Cómo cambió el patrimonio?",
    "¿Cuáles son las predicciones de crecimiento?",
    "¿Qué riesgos financieros se anticipan?"
  ];

  return (
    <Box>
      <form onSubmit={handleSubmit}>
        <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
          <TextField
            fullWidth
            variant="outlined"
            placeholder="Haz una pregunta sobre los estados financieros..."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            disabled={loading}
            multiline
            rows={2}
          />
          <Button
            type="submit"
            variant="contained"
            disabled={loading || !question.trim()}
            sx={{ minWidth: 120 }}
            startIcon={loading ? <CircularProgress size={20} /> : <SendIcon />}
          >
            {loading ? 'Analizando...' : 'Preguntar'}
          </Button>
        </Box>

        <FormControlLabel
          control={
            <Switch
              checked={includePredictions}
              onChange={(e) => setIncludePredictions(e.target.checked)}
              disabled={loading}
            />
          }
          label={
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <PredictionIcon fontSize="small" />
              Incluir análisis predictivo
            </Box>
          }
        />
      </form>

      {/* Preguntas sugeridas */}
      <Box sx={{ mt: 2, mb: 2 }}>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          Preguntas sugeridas:
        </Typography>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
          {suggestedQuestions.map((q, index) => (
            <Chip
              key={index}
              label={q}
              variant="outlined"
              size="small"
              onClick={() => setQuestion(q)}
              disabled={loading}
              sx={{ cursor: 'pointer' }}
            />
          ))}
        </Box>
      </Box>

      {/* Error */}
      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}

      {/* Resultado */}
      {result && (
        <Paper sx={{ p: 3, mt: 3, bgcolor: 'grey.50' }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">
              📋 Respuesta
            </Typography>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Chip 
                label={`Agente: ${result.agent_used}`} 
                size="small" 
                color="primary" 
                variant="outlined" 
              />
              <Chip 
                label={`Confianza: ${(result.confidence * 100).toFixed(0)}%`} 
                size="small" 
                color={result.confidence > 0.8 ? 'success' : result.confidence > 0.6 ? 'warning' : 'error'}
                variant="outlined" 
              />
            </Box>
          </Box>

          <Typography variant="body1" sx={{ mb: 2, whiteSpace: 'pre-wrap' }}>
            {result.answer}
          </Typography>

          {result.predictions_included && result.prediction_data && (
            <Box sx={{ mt: 2, p: 2, bgcolor: 'primary.50', borderRadius: 1 }}>
              <Typography variant="subtitle2" sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                <PredictionIcon fontSize="small" />
                Análisis Predictivo Incluido
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Se generaron predicciones adicionales basadas en los datos históricos.
              </Typography>
            </Box>
          )}

          {result.files_generated > 0 && (
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              📁 {result.files_generated} archivo(s) generado(s) para análisis detallado
            </Typography>
          )}
        </Paper>
      )}
    </Box>
  );
};

export default QuestionInterface;
