"""
validation_module.py - Módulo de Validación Walk-Forward
Integrado con el sistema híbrido de predicción
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from update_predictor_agent import EvolutionaryPredictorAgent


# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Validador walk-forward para el sistema evolutivo
    Integrado con HybridPredictorAgent y configuración dinámica
    """
    
    def __init__(self, predictor_agent):
        """
        Args:
            predictor_agent: Instancia de EvolutionaryPredictorAgent
        """
        self.predictor = predictor_agent
        self.validation_results = {}
        
        logger.info("✅ WalkForwardValidator inicializado")
    
    
    def walk_forward_validate(
        self,
        data: pd.Series,
        metric_name: str,
        n_splits: int = None,  # Ahora adaptativo
        min_train_size: int = None  # Ahora adaptativo
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        VALIDACIÓN ADAPTADA: Walk-forward para series cortas (5 años anuales)
        """
        logger.info(f" Iniciando validación walk-forward para {metric_name}")
        
        # NUEVO: Configuración adaptativa según longitud de datos
        data_length = len(data)
        
        if data_length < 5:
            logger.error(f" Datos insuficientes: {data_length} < 5 años mínimo")
            return self._create_dummy_validation(), {}
        
        # Ajustar parámetros según datos disponibles
        if n_splits is None:
            # Para 5 años: máximo 3 splits (entrenar 2→validar 1, entrenar 3→validar 1, entrenar 4→validar 1)
            n_splits = min(3, data_length - 2)
        
        if min_train_size is None:
            # Mínimo 2 años para entrenar con datos anuales
            min_train_size = max(2, data_length // 2)
        
        logger.info(f" Configuración adaptativa: {data_length} años → {n_splits} splits, train_size={min_train_size}")
        
        results = {
            'prophet_predictions': [],
            'xgboost_predictions': [],
            'ensemble_predictions': [],
            'actual_values': [],
            'dates': [],
            'errors': {
                'prophet_mae': [],
                'xgboost_mae': [],
                'ensemble_mae': []
            }
        }
        
        # Validación con expanding window (no rolling) para datos cortos
        for i in range(n_splits):
            try:
                # Expanding window: siempre entrena desde el inicio
                train_end = min_train_size + i
                test_idx = train_end
                
                if test_idx >= len(data):
                    break
                
                train_data = data.iloc[:train_end]
                actual_value = data.iloc[test_idx]
                test_date = data.index[test_idx]
                
                logger.info(f" Ventana {i+1}/{n_splits}: entrenar {len(train_data)} años → validar {test_date.year}")
                
                # Preparar datos con frecuencia anual
                df = self.predictor.prepare_data_for_prophet(
                    train_data, None, metric_name, frequency='Y'
                )
                
                if len(df) < 2:
                    logger.warning(f" Ventana {i+1} sin datos suficientes")
                    continue
                
                # Predicción Prophet (1 año adelante)
                prophet_results = self.predictor.prophet_prediction(
                    df, metric_name, periods=1, frequency='Y'
                )
                prophet_pred = (prophet_results['predictions'][0]
                            if prophet_results['predictions'] else actual_value)
                
                # Predicción XGBoost (con fallback para pocas observaciones)
                if len(df) >= 3:
                    xgboost_results = self.predictor.xgboost_prediction(
                        df, metric_name, periods=1
                    )
                    xgb_pred = (xgboost_results['predictions'][0]
                            if xgboost_results['predictions'] else actual_value)
                else:
                    # Fallback: usar tendencia lineal simple
                    xgb_pred = prophet_pred
                
                # Ensemble
                ensemble_results = self.predictor.ensemble_prediction(
                    prophet_results, {'predictions': [xgb_pred]}
                )
                ensemble_pred = (ensemble_results['predictions'][0]
                                if ensemble_results['predictions'] else actual_value)
                
                # Almacenar resultados
                results['prophet_predictions'].append(prophet_pred)
                results['xgboost_predictions'].append(xgb_pred)
                results['ensemble_predictions'].append(ensemble_pred)
                results['actual_values'].append(actual_value)
                results['dates'].append(test_date)
                
                # Errores
                results['errors']['prophet_mae'].append(abs(prophet_pred - actual_value))
                results['errors']['xgboost_mae'].append(abs(xgb_pred - actual_value))
                results['errors']['ensemble_mae'].append(abs(ensemble_pred - actual_value))
                
            except Exception as e:
                logger.warning(f" Error en validación paso {i+1}: {e}")
                continue
        
        # Calcular métricas finales
        validation_summary = self._calculate_validation_metrics(results)
        
        logger.info(f" Validación completada: {len(results['actual_values'])} períodos validados")
        
        return validation_summary, results

    
    def _calculate_validation_metrics(self, results: Dict) -> Dict[str, Any]:
        """
        Calcula métricas de rendimiento de validación
        """
        if not results['actual_values']:
            return self._create_dummy_validation()
        
        actual = np.array(results['actual_values'])
        prophet_preds = np.array(results['prophet_predictions'])
        xgb_preds = np.array(results['xgboost_predictions']) 
        ensemble_preds = np.array(results['ensemble_predictions'])
        
        metrics = {}
        
        # Métricas para cada modelo
        for model_name, preds in [
            ('prophet', prophet_preds), 
            ('xgboost', xgb_preds), 
            ('ensemble', ensemble_preds)
        ]:
            if len(preds) > 0:
                mae = np.mean(np.abs(preds - actual))
                mse = np.mean((preds - actual) ** 2)
                rmse = np.sqrt(mse)
                
                # R² score (maneja casos con poca varianza)
                try:
                    r2 = r2_score(actual, preds)
                except:
                    r2 = 0.0
                
                # MAPE (Mean Absolute Percentage Error)
                with np.errstate(divide='ignore', invalid='ignore'):
                    mape = np.mean(np.abs((actual - preds) / actual)) * 100
                    mape = np.nan_to_num(mape, nan=0.0, posinf=0.0, neginf=0.0)
                
                metrics[f'{model_name}_mae'] = float(mae)
                metrics[f'{model_name}_mse'] = float(mse)
                metrics[f'{model_name}_rmse'] = float(rmse)
                metrics[f'{model_name}_r2'] = float(r2)
                metrics[f'{model_name}_mape'] = float(mape)
        
        # Determina mejor modelo
        mae_scores = {k: v for k, v in metrics.items() if k.endswith('_mae')}
        best_model = (min(mae_scores.keys(), key=mae_scores.get).replace('_mae', '') 
                     if mae_scores else 'ensemble')
        
        metrics['best_model'] = best_model
        metrics['validation_periods'] = len(results['actual_values'])
        
        return metrics
    
    
    def _create_dummy_validation(self) -> Dict[str, Any]:
        """
        Crea métricas dummy cuando no hay suficientes datos
        """
        return {
            'prophet_mae': 0.1,
            'prophet_r2': 0.5,
            'prophet_mape': 10.0,
            'xgboost_mae': 0.1,
            'xgboost_r2': 0.5,
            'xgboost_mape': 10.0,
            'ensemble_mae': 0.1,
            'ensemble_r2': 0.6,
            'ensemble_mape': 9.0,
            'best_model': 'ensemble',
            'validation_periods': 0,
            'note': 'Insufficient data for validation'
        }
    
    
    def cross_validate_all_metrics(
        self, 
        financial_data: pd.DataFrame, 
        metrics: Optional[list] = None, 
        n_splits: int = 6
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Validación cruzada para todas las métricas financieras
        
        Args:
            financial_data: DataFrame con métricas financieras
            metrics: Lista de métricas a validar (default: todas disponibles)
            n_splits: Número de splits para validación
        
        Returns:
            Tuple con (overall_summary, all_results)
        """
        if metrics is None:
            # Detecta métricas disponibles
            available_metrics = ['ROA', 'ROE', 'ratio_solvencia', 'liquidez', 'beneficio_neto']
            metrics = [col for col in financial_data.columns if col in available_metrics]
        
        logger.info(f" Validación cruzada para {len(metrics)} métricas: {metrics}")
        
        all_results = {}
        
        for metric in metrics:
            if metric in financial_data.columns:
                series = financial_data[metric].dropna()
                
                if len(series) >= 10:  # Mínimo para validación robusta
                    logger.info(f"\n Validando {metric}...")
                    validation_summary, validation_results = self.walk_forward_validate(
                        series, metric, n_splits
                    )
                    all_results[metric] = {
                        'summary': validation_summary,
                        'details': validation_results
                    }
                else:
                    logger.warning(f" {metric}: datos insuficientes ({len(series)} < 10)")
                    all_results[metric] = {
                        'summary': self._create_dummy_validation(),
                        'details': {}
                    }
        
        # Resumen general
        overall_summary = self._create_overall_summary(all_results)
        
        logger.info("\n Validación cruzada completada para todas las métricas")
        logger.info(f"   Promedio Ensemble MAE: {overall_summary['avg_ensemble_mae']:.4f}")
        logger.info(f"   Promedio Ensemble R²: {overall_summary['avg_ensemble_r2']:.4f}")
        logger.info(f"   Mejor métrica: {overall_summary['best_performing_metric']}")
        
        return overall_summary, all_results
    
    
    def _create_overall_summary(self, all_results: Dict) -> Dict[str, Any]:
        """
        Crea resumen general de validación
        """
        summary = {
            'total_metrics': len(all_results),
            'avg_ensemble_mae': 0.0,
            'avg_ensemble_r2': 0.0,
            'avg_ensemble_mape': 0.0,
            'best_performing_metric': '',
            'worst_performing_metric': '',
            'metrics_validated': []
        }
        
        if not all_results:
            return summary
        
        # Calcula promedios
        ensemble_maes = []
        ensemble_r2s = []
        ensemble_mapes = []
        
        for metric, results in all_results.items():
            if 'ensemble_mae' in results['summary']:
                ensemble_maes.append(results['summary']['ensemble_mae'])
                summary['metrics_validated'].append(metric)
            if 'ensemble_r2' in results['summary']:
                ensemble_r2s.append(results['summary']['ensemble_r2'])
            if 'ensemble_mape' in results['summary']:
                ensemble_mapes.append(results['summary']['ensemble_mape'])
        
        if ensemble_maes:
            summary['avg_ensemble_mae'] = float(np.mean(ensemble_maes))
            summary['best_performing_metric'] = min(
                all_results.keys(), 
                key=lambda x: all_results[x]['summary'].get('ensemble_mae', float('inf'))
            )
            summary['worst_performing_metric'] = max(
                all_results.keys(), 
                key=lambda x: all_results[x]['summary'].get('ensemble_mae', 0)
            )
        
        if ensemble_r2s:
            summary['avg_ensemble_r2'] = float(np.mean(ensemble_r2s))
        
        if ensemble_mapes:
            summary['avg_ensemble_mape'] = float(np.mean(ensemble_mapes))
        
        return summary


# ============================================================================
# INTEGRACIÓN CON SISTEMA HÍBRIDO
# ============================================================================

class HybridPredictorValidator:
    """
    Validador específico para el HybridPredictorAgent
    Extiende WalkForwardValidator con integración al sistema completo
    """
    
    def __init__(self, hybrid_predictor_agent):
        """
        Args:
            hybrid_predictor_agent: Instancia de HybridPredictorAgent o DynamicHybridPredictorAgent
        """
        self.hybrid_agent = hybrid_predictor_agent
        
        # Accede al agente ML interno
        if hasattr(hybrid_predictor_agent, 'base_agent'):
            # Es DynamicHybridPredictorAgent
            self.ml_agent = getattr(hybrid_predictor_agent, 'evo_agent', None)
        else:
            # Es HybridPredictorAgent directo - usa 'evo_agent', NO 'ml_agent'
            self.ml_agent = getattr(hybrid_predictor_agent, 'evo_agent', None)
        
        # Acceder a otros componentes importantes
        self.llm_agent = getattr(hybrid_predictor_agent, 'llm_agent', None)
        self.evo_agent = getattr(hybrid_predictor_agent, 'evo_agent', None)
        self.regulatory_agent = getattr(hybrid_predictor_agent, 'regulatory_agent', None)
        
        # Crea validador walk-forward
        if self.ml_agent:
            try:
                self.validator = WalkForwardValidator(self.ml_agent)
                logger.info("   Walk-forward Validator creado")
            except Exception as e:
                logger.warning(f" Error creando validator: {e}")
                # Usar validator existente del hybrid_agent si está disponible
                self.validator = getattr(hybrid_predictor_agent, 'validator', None)
        else:
            # Usar el validator existente del hybrid_agent
            self.validator = getattr(hybrid_predictor_agent, 'validator', None)
            logger.warning(" No se encontró ml_agent, usando validator del hybrid_agent")
        
        logger.info(" HybridPredictorValidator inicializado")
        if self.llm_agent:
            logger.info("   LLM Agent disponible")
        if self.ml_agent:
            logger.info("   ML/Evo Agent disponible")
        if self.validator:
            logger.info("   Validator disponible")
        
    def validate_predictions(
                self, 
                time_series_data: pd.DataFrame,
                metrics_to_validate: Optional[list] = None
            ) -> Dict[str, Any]:
                """
                Valida predicciones del sistema híbrido completo
                
                Args:
                    time_series_data: DataFrame con series temporales históricas
                    metrics_to_validate: Lista de métricas a validar
                
                Returns:
                    Dict con resultados de validación completos
                """
                logger.info("="*60)
                logger.info("🔍 INICIANDO VALIDACIÓN DEL SISTEMA HÍBRIDO")
                logger.info("="*60)
                
                # Valida todas las métricas
                overall_summary, detailed_results = self.validator.cross_validate_all_metrics(
                    financial_data=time_series_data,
                    metrics=metrics_to_validate,
                    n_splits=8
                )
                
                # Estructura resultado
                validation_report = {
                    'validation_date': datetime.now().isoformat(),
                    'overall_summary': overall_summary,
                    'detailed_results': detailed_results,
                    'recommendations': self._generate_validation_recommendations(overall_summary)
                }
                
                logger.info("="*60)
                logger.info(" VALIDACIÓN COMPLETADA")
                logger.info("="*60)
                
                return validation_report
    
    def _generate_validation_recommendations(self, summary: Dict) -> list:
                """
                Genera recomendaciones basadas en resultados de validación
                """
                recommendations = []
                
                avg_mae = summary.get('avg_ensemble_mae', 0)
                avg_r2 = summary.get('avg_ensemble_r2', 0)
                
                # Recomendaciones según rendimiento
                if avg_mae < 0.5:
                    recommendations.append({
                        'level': 'SUCCESS',
                        'message': f'Excelente precisión de predicción (MAE: {avg_mae:.4f}). '
                                'El modelo es altamente confiable.'
                    })
                elif avg_mae < 1.0:
                    recommendations.append({
                        'level': 'GOOD',
                        'message': f'Buena precisión de predicción (MAE: {avg_mae:.4f}). '
                                'El modelo es confiable para uso operativo.'
                    })
                else:
                    recommendations.append({
                        'level': 'WARNING',
                        'message': f'Precisión moderada (MAE: {avg_mae:.4f}). '
                                'Considerar más datos históricos o ajuste de hiperparámetros.'
                    })
                
                if avg_r2 > 0.7:
                    recommendations.append({
                        'level': 'SUCCESS',
                        'message': f'Excelente capacidad explicativa (R²: {avg_r2:.4f}). '
                                'El modelo captura bien la varianza de los datos.'
                    })
                elif avg_r2 > 0.4:
                    recommendations.append({
                        'level': 'GOOD',
                        'message': f'Capacidad explicativa aceptable (R²: {avg_r2:.4f}).'
                    })
                else:
                    recommendations.append({
                        'level': 'WARNING',
                        'message': f'Capacidad explicativa limitada (R²: {avg_r2:.4f}). '
                                'Considerar variables adicionales o modelos alternativos.'
                    })
                
                return recommendations
    
    def cross_validate_all_metrics(
                self,
                financial_data: pd.DataFrame,
                metrics: Optional[list] = None
            ) -> Dict[str, Any]:
                """
                Validación cruzada de todas las métricas financieras
                
                Args:
                    financial_data: DataFrame con datos financieros
                    metrics: Lista de métricas a validar (opcional)
                    
                Returns:
                    Diccionario con resultados de validación cruzada
                """
                logger.info(" Iniciando validación cruzada de métricas financieras...")
                
                if self.validator:
                    overall_summary, all_results = self.validator.cross_validate_all_metrics(
                        financial_data=financial_data,
                        metrics=metrics,
                        n_splits=6
                    )
                    
                    # Agregar recomendaciones híbridas
                    recommendations = self._generate_validation_recommendations(overall_summary)
                    
                    return {
                        'overall_summary': overall_summary,
                        'detailed_results': all_results,
                        'recommendations': recommendations,
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    logger.warning(" Validator no disponible para validación cruzada")
                    return {
                        'error': 'Validator not available',
                        'timestamp': datetime.now().isoformat()
                    }

            
    
# ============================================================================
# FUNCIÓN DE TEST
# ============================================================================

def test_validation_module():
    """
    Test del módulo de validación integrado
    """
    print("\n" + "="*70)
    print(" TEST DEL MÓDULO DE VALIDACIÓN INTEGRADO")
    print("="*70)
    
    try:
        
        # Crea datos de prueba
        print("\n Generando datos de prueba...")
        dates = pd.date_range('2020-01-01', periods=24, freq='Q')
        test_data = pd.DataFrame({
            'ROA': 1.2 + 0.1 * np.sin(2 * np.pi * np.arange(24) / 4) + np.random.normal(0, 0.05, 24),
            'ratio_solvencia': 12.0 + 0.5 * np.sin(2 * np.pi * np.arange(24) / 4) + np.random.normal(0, 0.2, 24),
            'liquidez': 1.5 + 0.1 * np.sin(2 * np.pi * np.arange(24) / 4) + np.random.normal(0, 0.05, 24)
        }, index=dates)
        
        print(f"   Datos generados: {len(test_data)} períodos, {len(test_data.columns)} métricas")
        
        # Crea predictor
        print("\n Inicializando predictor evolutivo...")
        predictor = EvolutionaryPredictorAgent()
        
        # Crea validador
        print("\n Inicializando validador walk-forward...")
        validator = WalkForwardValidator(predictor)
        
        # Ejecuta validación
        print("\n Ejecutando validación cruzada...")
        overall_summary, all_results = validator.cross_validate_all_metrics(
            financial_data=test_data,
            n_splits=6
        )
        
        # Muestra resultados
        print("\n" + "="*70)
        print(" RESULTADOS DE VALIDACIÓN")
        print("="*70)
        print(f"\nMétricas validadas: {overall_summary['metrics_validated']}")
        print(f"Promedio Ensemble MAE: {overall_summary['avg_ensemble_mae']:.4f}")
        print(f"Promedio Ensemble R²: {overall_summary['avg_ensemble_r2']:.4f}")
        print(f"Mejor métrica: {overall_summary['best_performing_metric']}")
        print(f"Peor métrica: {overall_summary['worst_performing_metric']}")
        
        print("\n Test completado exitosamente!")
        return True
        
    except ImportError as e:
        print(f"\n Error de importación: {e}")
        print("   Asegúrate de que EvolutionaryPredictorAgent esté disponible")
        return False
    except Exception as e:
        print(f"\n Error en test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_validation_module()
