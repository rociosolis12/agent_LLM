import React, { useState, useEffect } from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Box,
  Alert,
  Fab,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Chip,
  LinearProgress
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  CloudUpload as UploadIcon,
  Assessment as AnalysisIcon,
  TrendingUp as PredictionIcon
} from '@mui/icons-material';

import FileUpload from '../FileUpload/FileUpload';
import QuestionInterface from '../QuestionInterface/QuestionInterface';
import PipelineStatus from '../PipelineStatus/PipelineStatus';
import ResultsViewer from '../ResultsViewer/ResultsViewer';
import SystemStatus from '../SystemStatus/SystemStatus';
import { useSystemStatus } from '../../hooks/useSystemStatus';
import { financialApi } from '../../services/api';

const Dashboard: React.FC = () => {
  const { systemStatus, loading: statusLoading, refresh: refreshStatus } = useSystemStatus();
  const [pipelineOpen, setPipelineOpen] = useState(false);
  const [pipelineRunning, setPipelineRunning] = useState(false);
  const [pipelineResult, setPipelineResult] = useState<any>(null);

  const handleRunPipeline = async () => {
    setPipelineRunning(true);
    try {
      const result = await financialApi.runPipeline({ include_predictions: true });
      setPipelineResult(result);
      refreshStatus();
    } catch (error) {
      console.error('Error running pipeline:', error);
    } finally {
      setPipelineRunning(false);
    }
  };

  const getPipelineStatusColor = () => {
    if (!systemStatus) return 'default';
    const health = systemStatus.system_health?.status;
    switch (health) {
      case 'healthy': return 'success';
      case 'degraded': return 'warning';
      case 'unhealthy': return 'error';
      default: return 'default';
    }
  };

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          🤖 Sistema Multi-Agente Financiero
        </Typography>
        <Typography variant="subtitle1" color="text.secondary">
          Análisis inteligente de estados financieros con predicciones avanzadas
        </Typography>
        
        {systemStatus && (
          <Box sx={{ mt: 2, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            <Chip 
              icon={<AnalysisIcon />}
              label={`${systemStatus.agents_status.loaded} Agentes Activos`}
              color={systemStatus.agents_status.loaded >= 4 ? 'success' : 'warning'}
              variant="outlined"
            />
            <Chip 
              icon={<PredictionIcon />}
              label={systemStatus.agents_status.predictor_available ? 'Predicciones ON' : 'Predicciones OFF'}
              color={systemStatus.agents_status.predictor_available ? 'success' : 'default'}
              variant="outlined"
            />
            <Chip 
              label={`Estado: ${systemStatus.system_health?.status || 'Unknown'}`}
              color={getPipelineStatusColor()}
              variant="outlined"
            />
          </Box>
        )}
      </Box>

      <Grid container spacing={3}>
        {/* Panel de Carga de Archivos */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, height: '400px' }}>
            <Typography variant="h6" gutterBottom>
              📄 Cargar Estados Financieros
            </Typography>
            <FileUpload onUploadSuccess={refreshStatus} />
          </Paper>
        </Grid>

        {/* Estado del Sistema */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, height: '400px' }}>
            <Typography variant="h6" gutterBottom>
              🔧 Estado del Sistema
            </Typography>
            <SystemStatus />
          </Paper>
        </Grid>

        {/* Interfaz de Preguntas */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              💬 Consulta Inteligente
            </Typography>
            <QuestionInterface />
          </Paper>
        </Grid>

        {/* Estado del Pipeline */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              🔄 Pipeline de Análisis
            </Typography>
            <PipelineStatus result={pipelineResult} />
          </Paper>
        </Grid>

        {/* Resultados */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              📊 Resultados
            </Typography>
            <ResultsViewer />
          </Paper>
        </Grid>
      </Grid>

      {/* FAB para ejecutar pipeline completo */}
      <Fab
        color="primary"
        aria-label="run pipeline"
        sx={{ position: 'fixed', bottom: 16, right: 16 }}
        onClick={() => setPipelineOpen(true)}
        disabled={!systemStatus?.pdf_status.extracted_exists}
      >
        <PlayIcon />
      </Fab>

      {/* Dialog para pipeline completo */}
      <Dialog open={pipelineOpen} onClose={() => setPipelineOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>🚀 Ejecutar Pipeline Completo</DialogTitle>
        <DialogContent>
          <Typography variant="body1" gutterBottom>
            El pipeline completo ejecutará todos los agentes especializados y generará predicciones:
          </Typography>
          <Box sx={{ mt: 2, mb: 2 }}>
            <Typography variant="body2" color="text.secondary">
              • Extractor PDF → Balance Agent → Income Agent → Cashflows Agent → Equity Agent → Predictor Agent
            </Typography>
          </Box>
          {pipelineRunning && (
            <Box sx={{ mt: 2 }}>
              <LinearProgress />
              <Typography variant="body2" sx={{ mt: 1 }}>
                Ejecutando pipeline...
              </Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setPipelineOpen(false)}>Cancelar</Button>
          <Button 
            onClick={handleRunPipeline} 
            variant="contained" 
            disabled={pipelineRunning}
            startIcon={<PlayIcon />}
          >
            Ejecutar Pipeline
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default Dashboard;
