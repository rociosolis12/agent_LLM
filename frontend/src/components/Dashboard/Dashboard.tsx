import React, { useState, useEffect } from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Box,
  Chip,
  Card,
  CardContent,
  CardActions,
  Button,
  CircularProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton
} from '@mui/material';
import {
  AccountBalance as BalanceIcon,
  BarChart as IncomeIcon,
  AttachMoney as CashFlowIcon,
  ShowChart as EquityIcon,
  AutoAwesome as ForecastIcon,
  Close as CloseIcon
} from '@mui/icons-material';

interface AgentCard {
  id: string;
  title: string;
  description: string;
  icon: React.ReactElement;
  color: string;
  endpoint: string;
  isPredictor?: boolean;
}

interface DashboardProps {
  onNavigateToPredictor?: () => void;
}

const Dashboard: React.FC<DashboardProps> = ({ onNavigateToPredictor }) => {
  const [systemStatus, setSystemStatus] = useState<any>(null);
  const [agentLoading, setAgentLoading] = useState<string | null>(null);
  const [agentsActive, setAgentsActive] = useState(5);
  const [analysisResult, setAnalysisResult] = useState<any>(null);
  const [openModal, setOpenModal] = useState(false);

  // Definición de los 5 agentes especializados
  const agents: AgentCard[] = [
    {
      id: 'balance',
      title: 'Balance General',
      description: 'Activos, pasivos y patrimonio',
      icon: <BalanceIcon sx={{ fontSize: 60 }} />,
      color: '#6366f1',
      endpoint: '/api/agents/balance-analysis'
    },
    {
      id: 'income',
      title: 'Estado de Resultados',
      description: 'Ingresos, gastos y rentabilidad',
      icon: <IncomeIcon sx={{ fontSize: 60 }} />,
      color: '#ec4899',
      endpoint: '/api/agents/income-analysis'
    },
    {
      id: 'cashflow',
      title: 'Flujos de Efectivo',
      description: 'Entradas y salidas de efectivo',
      icon: <CashFlowIcon sx={{ fontSize: 60 }} />,
      color: '#14b8a6',
      endpoint: '/api/agents/cashflow-analysis'
    },
    {
      id: 'equity',
      title: 'Estado de Patrimonio',
      description: 'Cambios en el capital',
      icon: <EquityIcon sx={{ fontSize: 60 }} />,
      color: '#10b981',
      endpoint: '/api/agents/equity-analysis'
    },
    {
      id: 'forecast',
      title: 'Pronóstico AI',
      description: 'Predictor híbrido ML + LLM',
      icon: <ForecastIcon sx={{ fontSize: 60 }} />,
      color: '#f59e0b',
      endpoint: '',
      isPredictor: true
    }
  ];

  useEffect(() => {
    checkSystemStatus();
  }, []);

  const checkSystemStatus = async () => {
    try {
      const response = await fetch('http://localhost:8000/system-status');
      const data = await response.json();
      setSystemStatus(data);
      setAgentsActive(data.system_available ? 5 : 0);
    } catch (error) {
      console.error('Error checking system status:', error);
      setAgentsActive(0);
    }
  };

  const handleAgentClick = async (agent: AgentCard) => {
    // Si es el pronóstico AI, navegar al dashboard del predictor
    if (agent.isPredictor && onNavigateToPredictor) {
      onNavigateToPredictor();
      return;
    }

    setAgentLoading(agent.id);

    try {
      const response = await fetch(`http://localhost:8000${agent.endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          company: 'BBVA',
          generate_analysis: true
        })
      });

      const data = await response.json();
      console.log(`${agent.title} result:`, data);
      
      // Guardar resultado y abrir modal
      setAnalysisResult({
        agent: agent.title,
        color: agent.color,
        data: data
      });
      setOpenModal(true);
      
    } catch (error) {
      console.error(`Error ejecutando ${agent.title}:`, error);
      setAnalysisResult({
        agent: agent.title,
        color: agent.color,
        data: { error: 'Error ejecutando análisis' }
      });
      setOpenModal(true);
    } finally {
      setAgentLoading(null);
    }
  };

  const handleCloseModal = () => {
    setOpenModal(false);
    setAnalysisResult(null);
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #1e293b 0%, #334155 100%)',
        py: 6
      }}
    >
      <Container maxWidth="xl">
        {/* Header */}
        <Box sx={{ mb: 6, textAlign: 'center' }}>
          <Typography
            variant="h2"
            component="h1"
            gutterBottom
            sx={{
              color: '#60a5fa',
              fontWeight: 700,
              fontSize: { xs: '2rem', md: '3rem' }
            }}
          >
            🤖 Sistema Multi-Agente Financiero
          </Typography>
          
          <Box sx={{ mt: 3 }}>
            <Chip
              label={`✅ Sistema operativo - ${agentsActive} agentes activos`}
              sx={{
                bgcolor: agentsActive > 0 ? '#10b981' : '#ef4444',
                color: 'white',
                fontSize: '1rem',
                py: 2,
                px: 1
              }}
            />
          </Box>
          
          <Typography
            variant="h6"
            sx={{ color: '#94a3b8', mt: 2 }}
          >
            Salud del sistema: {agentsActive > 0 ? '100%' : '0%'}
          </Typography>
        </Box>

        {/* Tarjetas de Agentes */}
        <Grid container spacing={4} justifyContent="center">
          {agents.map((agent) => (
            <Grid item xs={12} sm={6} md={4} lg={2.4} key={agent.id}>
              <Card
                sx={{
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  background: 'rgba(255, 255, 255, 0.05)',
                  backdropFilter: 'blur(10px)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  borderTop: `4px solid ${agent.color}`,
                  cursor: 'pointer',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    transform: 'translateY(-12px)',
                    boxShadow: '0 20px 40px rgba(0, 0, 0, 0.5)',
                    background: 'rgba(255, 255, 255, 0.08)'
                  }
                }}
                onClick={() => handleAgentClick(agent)}
              >
                <CardContent sx={{ flexGrow: 1, textAlign: 'center', py: 4 }}>
                  <Box
                    sx={{
                      display: 'flex',
                      justifyContent: 'center',
                      alignItems: 'center',
                      mb: 3,
                      p: 3,
                      bgcolor: `${agent.color}20`,
                      borderRadius: 3,
                      color: agent.color,
                      mx: 'auto',
                      width: 'fit-content'
                    }}
                  >
                    {agent.icon}
                  </Box>
                  
                  <Typography
                    variant="h6"
                    component="h3"
                    gutterBottom
                    sx={{ color: 'white', fontWeight: 600 }}
                  >
                    {agent.title}
                  </Typography>
                  
                  <Typography
                    variant="body2"
                    sx={{ color: '#cbd5e1', mt: 1 }}
                  >
                    {agent.description}
                  </Typography>
                </CardContent>
                
                <CardActions sx={{ justifyContent: 'center', pb: 3 }}>
                  <Button
                    variant="contained"
                    fullWidth
                    disabled={agentLoading === agent.id}
                    sx={{
                      bgcolor: agent.color,
                      mx: 2,
                      py: 1.5,
                      fontWeight: 600,
                      '&:hover': {
                        bgcolor: agent.color,
                        opacity: 0.9
                      }
                    }}
                  >
                    {agentLoading === agent.id ? (
                      <CircularProgress size={24} sx={{ color: 'white' }} />
                    ) : agent.isPredictor ? (
                      'Ir al Pronóstico'
                    ) : (
                      'Ejecutar Análisis'
                    )}
                  </Button>
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>

        {/* Consulta Personalizada */}
        <Paper
          sx={{
            mt: 6,
            p: 4,
            background: 'rgba(255, 255, 255, 0.05)',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255, 255, 255, 0.1)'
          }}
        >
          <Typography variant="h5" gutterBottom sx={{ color: 'white' }}>
            💬 Consulta Personalizada
          </Typography>
          <Typography variant="body2" sx={{ color: '#94a3b8', mb: 2 }}>
            Selecciona una función para ejecutar análisis financiero especializado
          </Typography>
        </Paper>
      </Container>

      {/* Modal de Resultados */}
      <Dialog
        open={openModal}
        onClose={handleCloseModal}
        maxWidth="lg"
        fullWidth
        PaperProps={{
          sx: {
            borderTop: `4px solid ${analysisResult?.color || '#6366f1'}`,
            maxHeight: '80vh'
          }
        }}
      >
        <DialogTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h5" component="span">
            {analysisResult?.agent}
          </Typography>
          <IconButton onClick={handleCloseModal}>
            <CloseIcon />
          </IconButton>
        </DialogTitle>
        
        <DialogContent dividers sx={{ maxHeight: '70vh', overflow: 'auto' }}>
          {analysisResult?.data?.financial_analysis?.answer ? (
            <Box>
              {/* Respuesta principal del agente */}
              <Paper sx={{ p: 3, mb: 3, bgcolor: '#f8f9fa' }}>
                <Typography variant="h6" gutterBottom sx={{ color: analysisResult.color, fontWeight: 600 }}>
                  📊 Análisis Principal
                </Typography>
                <Typography 
                  variant="body1" 
                  component="pre" 
                  sx={{ 
                    whiteSpace: 'pre-wrap', 
                    fontFamily: 'inherit',
                    lineHeight: 1.8
                  }}
                >
                  {analysisResult.data.financial_analysis.answer}
                </Typography>
              </Paper>

              {/* Métricas de confianza y ejecución */}
              <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={12} sm={4}>
                  <Paper sx={{ p: 2, textAlign: 'center', bgcolor: '#e3f2fd' }}>
                    <Typography variant="caption" color="textSecondary">
                      Confianza
                    </Typography>
                    <Typography variant="h5" sx={{ fontWeight: 600, color: '#1976d2' }}>
                      {(analysisResult.data.financial_analysis.confidence * 100).toFixed(1)}%
                    </Typography>
                  </Paper>
                </Grid>
                <Grid item xs={12} sm={4}>
                  <Paper sx={{ p: 2, textAlign: 'center', bgcolor: '#f3e5f5' }}>
                    <Typography variant="caption" color="textSecondary">
                      Archivos Generados
                    </Typography>
                    <Typography variant="h5" sx={{ fontWeight: 600, color: '#7b1fa2' }}>
                      {analysisResult.data.financial_analysis.files_generated || 0}
                    </Typography>
                  </Paper>
                </Grid>
                <Grid item xs={12} sm={4}>
                  <Paper sx={{ p: 2, textAlign: 'center', bgcolor: '#e8f5e9' }}>
                    <Typography variant="caption" color="textSecondary">
                      Pasos Ejecutados
                    </Typography>
                    <Typography variant="h5" sx={{ fontWeight: 600, color: '#388e3c' }}>
                      {analysisResult.data.financial_analysis.steps_taken || 0}
                    </Typography>
                  </Paper>
                </Grid>
              </Grid>

              {/* Predicciones ML (si existen) */}
              {analysisResult.data.hybrid_predictions?.ml_predictions && 
              analysisResult.data.hybrid_predictions.ml_predictions.length > 0 && (
                <Paper sx={{ p: 3, mb: 3, bgcolor: '#fff3e0' }}>
                  <Typography variant="h6" gutterBottom sx={{ color: '#f57c00', fontWeight: 600 }}>
                    🔮 Predicciones ML
                  </Typography>
                  <Box sx={{ maxHeight: '300px', overflow: 'auto' }}>
                    {analysisResult.data.hybrid_predictions.ml_predictions.slice(0, 4).map((pred: any, idx: number) => (
                      <Box key={idx} sx={{ mb: 2, p: 2, bgcolor: 'white', borderRadius: 1 }}>
                        <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                          {pred.metric} - Periodo {pred.timestep}
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          Predicción: <strong>{pred.prediction.toFixed(4)}</strong>
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          Rango: {pred.lower.toFixed(4)} - {pred.upper.toFixed(4)}
                        </Typography>
                      </Box>
                    ))}
                  </Box>
                </Paper>
              )}

              {/* Recomendaciones (si existen) */}
              {analysisResult.data.hybrid_predictions?.recommendations && 
              analysisResult.data.hybrid_predictions.recommendations.length > 0 && (
                <Paper sx={{ p: 3, mb: 3, bgcolor: '#e8eaf6' }}>
                  <Typography variant="h6" gutterBottom sx={{ color: '#3f51b5', fontWeight: 600 }}>
                    💡 Recomendaciones
                  </Typography>
                  {analysisResult.data.hybrid_predictions.recommendations.map((rec: any, idx: number) => (
                    <Box 
                      key={idx} 
                      sx={{ 
                        mb: 2, 
                        p: 2, 
                        bgcolor: rec.level === 'SUCCESS' ? '#c8e6c9' : '#fff9c4',
                        borderRadius: 1,
                        borderLeft: `4px solid ${rec.level === 'SUCCESS' ? '#4caf50' : '#fbc02d'}`
                      }}
                    >
                      <Typography variant="body2">
                        <strong>{rec.level}:</strong> {rec.message}
                      </Typography>
                    </Box>
                  ))}
                </Paper>
              )}

              {/* Información de extracción PDF */}
              {analysisResult.data.pdf_extraction && (
                <Paper sx={{ p: 3, bgcolor: '#fce4ec' }}>
                  <Typography variant="h6" gutterBottom sx={{ color: '#c2185b', fontWeight: 600 }}>
                    📄 Extracción PDF
                  </Typography>
                  <Typography variant="body2" gutterBottom>
                    <strong>Páginas procesadas:</strong> {analysisResult.data.pdf_extraction.total_pages_extracted} de {analysisResult.data.pdf_extraction.original_pages}
                  </Typography>
                  <Typography variant="body2" gutterBottom>
                    <strong>Páginas extraídas:</strong> {analysisResult.data.pdf_extraction.pages_extracted?.join(', ')}
                  </Typography>
                  <Typography variant="body2">
                    <strong>Archivo generado:</strong> {analysisResult.data.pdf_extraction.file_size_kb} KB
                  </Typography>
                </Paper>
              )}
            </Box>
          ) : (
            <Typography>No hay resultados disponibles</Typography>
          )}
        </DialogContent>

        
        <DialogActions>
          <Button onClick={handleCloseModal} variant="contained">
            Cerrar
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Dashboard;
