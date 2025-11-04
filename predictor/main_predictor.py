"""
main_predictor.py - Orquestador Principal del Sistema Predictor Híbrido
========================================================================
Integra todos los componentes del módulo predictor:
1. EvolutionaryPredictorAgent (ML predictions con Prophet + XGBoost)
2. HybridPredictorAgent (Análisis híbrido LLM + ML + Regulatory)
3. WalkForwardValidator (Validación temporal robusta)
4. RegulatoryConfigAgent (Configuración regulatoria dinámica)

Pipeline completo:
- Configuración regulatoria
- Predicción ML evolutiva
- Validación walk-forward
- Análisis híbrido cualitativo/cuantitativo
- Consolidación y exportación de resultados
"""

import os
import sys
import asyncio
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd

# Imports de los módulos del predictor
from update_predictor_agent import EvolutionaryPredictorAgent
from hybrid_predictor_agent import HybridPredictorAgent
from validation_module import WalkForwardValidator, HybridPredictorValidator

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('predictor_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PredictorOrchestrator:
    """
    Orquestador principal del sistema predictor multi-agente
    Coordina predicción ML, validación, análisis híbrido y exportación
    """
    
    def __init__(
        self,
        bank_symbol: str = "GARAN.IS",
        jurisdiction: str = "TR",
        parent_bank: str = "BBVA",
        alpha_vantage_key: Optional[str] = None,
        data_dir: str = "./data_outputs",
        output_dir: str = "./data_outputs",
        use_regulatory_config: bool = True,
        always_generate_new: bool = True
    ):
        """
        Inicializa el orquestador con todos los componentes necesarios
        
        Args:
            bank_symbol: Símbolo del banco (ej: GARAN.IS)
            jurisdiction: Jurisdicción regulatoria (ej: TR para Turquía)
            parent_bank: Banco matriz (ej: BBVA)
            alpha_vantage_key: API key para Alpha Vantage
            data_dir: Directorio de entrada de datos
            output_dir: Directorio de salida de resultados
            use_regulatory_config: Si usar configuración regulatoria
        """
        self.bank_symbol = bank_symbol
        self.jurisdiction = jurisdiction
        self.parent_bank = parent_bank
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.alpha_vantage_key = alpha_vantage_key
        self.always_generate_new = always_generate_new

        # Crear directorios si no existen
        # SIEMPRE usar la ruta del predictor, no la que se pasa desde main_system
        predictor_base_dir = Path(__file__).parent  # Directorio predictor/
        default_data_outputs = predictor_base_dir / "data_outputs"

        self.data_dir = default_data_outputs
        self.output_dir = default_data_outputs
        self.alpha_vantage_key = alpha_vantage_key
        self.always_generate_new = always_generate_new

        # Crear directorios si no existen
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        
        # Inicializar componentes del sistema predictor
        logger.info("=" * 80)
        logger.info("Inicializando Orquestador del Sistema Predictor Híbrido")
        logger.info("=" * 80)
        
        # 1. Agente Predictor Evolutivo (ML)
        logger.info("Inicializando EvolutionaryPredictorAgent...")
        self.evo_predictor = EvolutionaryPredictorAgent(
            alpha_vantage_key=alpha_vantage_key
        )
        
        # 2. Validador Walk-Forward
        logger.info("Inicializando WalkForwardValidator...")
        self.validator = WalkForwardValidator(self.evo_predictor)
        
        # 3. Agente Híbrido (LLM + ML + Regulatory)
        logger.info("Inicializando HybridPredictorAgent...")
        self.hybrid_agent = HybridPredictorAgent(
            bank_symbol=bank_symbol,
            alpha_vantage_key=alpha_vantage_key,
            jurisdiction=jurisdiction,
            parent_bank=parent_bank,
            use_regulatory_config=use_regulatory_config,
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir)
        )
        
        # 4. Validador híbrido avanzado
        logger.info(" Inicializando HybridPredictorValidator...")
        self.hybrid_validator = HybridPredictorValidator(self.hybrid_agent)
        
        # Almacenamiento de resultados consolidados
        self.consolidated_results = {
            'metadata': {
                'execution_timestamp': datetime.now().isoformat(),
                'bank_symbol': bank_symbol,
                'jurisdiction': jurisdiction,
                'parent_bank': parent_bank,
                'pipeline_version': '2.0'
            },
            'ml_predictions': {},
            'validation_results': {},
            'hybrid_analysis': {},
            'recommendations': {}
        }
        
        logger.info(" Orquestador inicializado correctamente")
        logger.info("=" * 80)

    def run_ml_predictions(
        self,
        agent_results: Optional[Dict[str, Any]] = None,
        generate_new: bool = True  # ← CAMBIO 1: True por defecto
    ) -> pd.DataFrame:
        """
        Ejecuta predicciones ML usando datos de agentes financieros
        
        Args:
            agent_results: Diccionario con resultados de balance, income, cashflows, equity
            generate_new: Si generar nuevas predicciones
            
        Returns:
            DataFrame con predicciones ML
        """
        logger.info("🤖 Ejecutando predicciones ML evolutivas...")
        
        csv_path = Path(self.output_dir) / "evolutionary_predictions.csv"
        
        # ✅ CAMBIO 2: SIEMPRE REGENERAR SI HAY agent_results
        should_generate = (agent_results is not None) or generate_new
        
        if should_generate:
            logger.info("✨ Generando NUEVAS predicciones con datos de agentes...")
            
            # ✅ CAMBIO 3: EXTRAER DATOS DE AGENTES CORRECTAMENTE
            if agent_results:
                logger.info("📊 Extrayendo datos financieros de agentes...")
                
                # Usar el método de extracción del EvolutionaryPredictorAgent
                financial_data = self.evo_predictor.extract_financial_data_from_agents(
                    agent_results
                )
                
                if not financial_data.empty:
                    logger.info(f"   ✅ Datos extraídos: {financial_data.shape[0]} periodos, {financial_data.shape[1]} métricas")
                    logger.info(f"   Métricas disponibles: {list(financial_data.columns)}")
                else:
                    logger.warning("   ⚠️ No se extrajeron datos de agentes, usando mock")
                    financial_data = None
            else:
                logger.warning("⚠️ No hay agent_results, usando datos mock")
                financial_data = None
            
            # ✅ CAMBIO 4: LLAMAR AL PREDICTOR CON LOS DATOS EXTRAÍDOS
            try:
                logger.info("🔮 Generando predicciones con Prophet + XGBoost...")
                
                results = self.evo_predictor.predict_financial_metrics(
                    financial_data=financial_data,  # ← Pasar datos extraídos
                    agent_results=agent_results,  # ← También pasar agent_results
                    bank_symbol="GARAN.IS",
                    periods=4
                )
                
                logger.info(f"   ✅ Predicciones generadas para {len(results)} métricas")
                
            except Exception as e:
                logger.error(f"   ❌ Error en predicción: {e}")
                raise
            
            # ✅ CAMBIO 5: CONVERTIR CORRECTAMENTE A DATAFRAME
            if results:
                try:
                    predictions_list = []
                    
                    for metric, res in results.items():
                        logger.info(f"   Procesando métrica: {metric}")
                        
                        # Extraer predicciones ensemble
                        if isinstance(res, dict) and 'ensemble' in res:
                            ensemble = res['ensemble']
                            predictions = ensemble.get('predictions', [])
                            lower = ensemble.get('lower_bound', [])
                            upper = ensemble.get('upper_bound', [])
                        else:
                            predictions = res if isinstance(res, list) else [res]
                            lower = [p * 0.9 for p in predictions]
                            upper = [p * 1.1 for p in predictions]
                        
                        # Crear registros
                        for i, pred in enumerate(predictions):
                            predictions_list.append({
                                'metric': metric,
                                'periodo': i + 1,
                                'prediction': float(pred) if pred is not None else 0,
                                'lower_bound': float(lower[i]) if i < len(lower) else 0,
                                'upper_bound': float(upper[i]) if i < len(upper) else 0
                            })
                    
                    ml_predictions = pd.DataFrame(predictions_list)
                    
                    # Guardar a CSV
                    ml_predictions.to_csv(csv_path, index=False)
                    logger.info(f"💾 Predicciones guardadas: {csv_path}")
                    logger.info(f"   Total registros: {len(ml_predictions)}")
                    
                except Exception as e:
                    logger.error(f"❌ Error convertiendo a DataFrame: {e}")
                    raise
            else:
                logger.error("❌ No se generaron resultados de predicción")
                raise ValueError("No prediction results generated")
        
        # ✅ CAMBIO 6: CARGAR SI YA EXISTEN (pero solo si NO hay agent_results nuevos)
        elif csv_path.exists():
            logger.info("📂 Cargando predicciones ML existentes...")
            try:
                ml_predictions = pd.read_csv(csv_path)
                logger.info(f"   ✅ Predicciones cargadas: {len(ml_predictions)} registros")
            except Exception as e:
                logger.error(f"❌ Error cargando predicciones: {e}")
                raise
        else:
            logger.error(f"❌ No hay predicciones en {csv_path}")
            raise FileNotFoundError(f"No predictions found at {csv_path}")
        
        # Guardar en resultados consolidados
        self.consolidated_results['ml_predictions'] = ml_predictions.to_dict('records')
        
        return ml_predictions



    def run_validation(
        self,
        ml_predictions: pd.DataFrame,
        metrics_to_validate: list = None
    ) -> Dict[str, Any]:
        """
        Ejecuta validación walk-forward en predicciones ML
        
        Args:
            ml_predictions: DataFrame con predicciones ML
            metrics_to_validate: Lista de métricas a validar
            
        Returns:
            Diccionario con resultados de validación
        """
        logger.info(" Ejecutando validación walk-forward...")
        
        if metrics_to_validate is None:
            metrics_to_validate = ["ROA", "ratio_solvencia", "liquidez"]
        
        validation_results = {}
        
        for metric in metrics_to_validate:
            if metric in ml_predictions.columns:
                logger.info(f"  ➤ Validando métrica: {metric}")
                
                try:
                    # Validación walk-forward
                    result = self.validator.walkforwardvalidate(
                        data=ml_predictions[metric],
                        metric_name=metric,
                        n_splits=8,
                        min_train_size=8
                    )
                    
                    validation_results[metric] = result
                    
                    # Log de resultados
                    logger.info(f"    ✓ MAE: {result['mae']:.4f}")
                    logger.info(f"    ✓ RMSE: {result['rmse']:.4f}")
                    logger.info(f"    ✓ R²: {result['r2']:.4f}")
                    
                except Exception as e:
                    logger.error(f"    ✗ Error validando {metric}: {e}")
                    validation_results[metric] = {'error': str(e)}
            else:
                logger.warning(f"  ⚠ Métrica {metric} no encontrada en predicciones")
        
        # Almacenar en resultados consolidados
        self.consolidated_results['validation_results'] = validation_results
        
        logger.info(" Validación completada")
        return validation_results
    
    async def run_hybrid_analysis(
        self,
        agent_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ejecuta análisis híbrido completo (LLM + ML + Regulatory)
        
        Args:
            agent_results: Resultados de agentes financieros especializados
            
        Returns:
            Diccionario con análisis híbrido completo
        """
        logger.info(" Ejecutando análisis híbrido completo...")
        
        try:
            # Ejecutar análisis híbrido
            hybrid_results = await self.hybrid_agent.run_full_analysis(
                agent_results=agent_results
            )
            
            # Almacenar en resultados consolidados
            self.consolidated_results['hybrid_analysis'] = hybrid_results
            
            logger.info("Análisis híbrido completado")
            return hybrid_results
            
        except Exception as e:
            logger.error(f" Error en análisis híbrido: {e}")
            raise
    
    def run_advanced_validation(
        self,
        ml_predictions: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Ejecuta validación avanzada con recomendaciones cualitativas
        
        Args:
            ml_predictions: DataFrame con predicciones ML
            
        Returns:
            Diccionario con validación avanzada y recomendaciones
        """
        logger.info(" Ejecutando validación avanzada híbrida...")
        
        try:
            # Validación cruzada de todas las métricas
            advanced_results = self.hybrid_validator.cross_validate_all_metrics(
                financial_data=ml_predictions
            )
            
            # Almacenar recomendaciones
            self.consolidated_results['recommendations'] = advanced_results.get(
                'recommendations', {}
            )
            
            logger.info(" Validación avanzada completada")
            return advanced_results
            
        except Exception as e:
            logger.error(f" Error en validación avanzada: {e}")
            return {'error': str(e)}
    
    def consolidate_and_export(
        self,
        export_format: str = 'json'
    ) -> Path:
        """
        Consolida todos los resultados y los exporta
        
        Args:
            export_format: Formato de exportación ('json' o 'csv')
            
        Returns:
            Path del archivo exportado
        """
        logger.info(" Consolidando y exportando resultados...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Exportar JSON consolidado
        json_path = self.output_dir / f"consolidated_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.consolidated_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f" Resultados JSON exportados: {json_path}")
        
        # Exportar CSV si se solicita
        if export_format == 'csv':
            csv_path = self.output_dir / f"consolidated_predictions_{timestamp}.csv"
            
            if self.consolidated_results.get('ml_predictions'):
                df = pd.DataFrame(self.consolidated_results['ml_predictions'])
                df.to_csv(csv_path, index=False)
                logger.info(f" Predicciones CSV exportadas: {csv_path}")
        
        # Exportar resumen ejecutivo
        summary_path = self.output_dir / f"executive_summary_{timestamp}.txt"
        self._export_executive_summary(summary_path)
        
        logger.info("=" * 80)
        logger.info(" PIPELINE COMPLETADO EXITOSAMENTE")
        logger.info(f" Resultados en: {self.output_dir}")
        logger.info("=" * 80)
        
        return json_path
    
    def _export_executive_summary(self, path: Path):
        """Exporta un resumen ejecutivo en texto plano"""
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RESUMEN EJECUTIVO - SISTEMA PREDICTOR HÍBRIDO\n")
            f.write("=" * 80 + "\n\n")
            
            # Metadata
            meta = self.consolidated_results.get('metadata', {
                    'timestamp': datetime.now().isoformat(),
                    'ticker': 'GARAN.IS',
                    'version': '1.0'
                })
        
            f.write(f"Fecha: {meta.get('execution_timestamp', datetime.now().isoformat())}\n")
            f.write(f"Banco: {meta.get('bank_symbol', 'GARAN.IS')}\n")
            f.write(f"Fecha: {meta.get('execution_timestamp', datetime.now().isoformat())}\n")
            f.write(f"Banco Matriz: {meta.get('parent_bank', 'BBVA.MC')}\n")
            
            # Predicciones ML
            f.write("PREDICCIONES ML\n")
            f.write("-" * 80 + "\n")
            ml_preds = self.consolidated_results.get('ml_predictions', [])
            f.write(f"Total de predicciones: {len(ml_preds)}\n\n")
            
            # Validación
            f.write("VALIDACIÓN WALK-FORWARD\n")
            f.write("-" * 80 + "\n")
            validations = self.consolidated_results.get('validation_results', {})
            for metric, result in validations.items():
                if 'error' not in result:
                    f.write(f"{metric}:\n")
                    f.write(f"  - MAE: {result.get('mae', 'N/A'):.4f}\n")
                    f.write(f"  - RMSE: {result.get('rmse', 'N/A'):.4f}\n")
                    f.write(f"  - R²: {result.get('r2', 'N/A'):.4f}\n\n")
            
            # Recomendaciones
            f.write("RECOMENDACIONES\n")
            f.write("-" * 80 + "\n")
            recs = self.consolidated_results.get('recommendations', {})
            if recs:
                f.write(json.dumps(recs, indent=2, ensure_ascii=False))
            
            f.write("\n" + "=" * 80 + "\n")
        
        logger.info(f" Resumen ejecutivo exportado: {path}")
    
    async def run_complete_pipeline(
        self,
        agent_results: Optional[Dict[str, Any]] = None,
        generate_new_predictions: bool = True,
        run_advanced_validation: bool = True
    ) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo con datos de agentes financieros
        
        Args:
            agent_results: Diccionario con resultados de balance, income, cashflows, equity
            generate_new_predictions: Si generar nuevas predicciones (siempre True si hay agent_results)
            run_advanced_validation: Si ejecutar validación avanzada
            
        Returns:
            Diccionario consolidado con predicciones, validación y recomendaciones
        """
        logger.info("=" * 80)
        logger.info(" INICIANDO PIPELINE COMPLETO DEL SISTEMA PREDICTOR")
        logger.info("=" * 80)
        
        logger.info(f" Parámetros recibidos:")
        logger.info(f"   agent_results disponibles: {agent_results is not None}")
        logger.info(f"   generate_new_predictions: {generate_new_predictions}")
        logger.info(f"   run_advanced_validation: {run_advanced_validation}")
        
        try:
            # ============================================================================
            # PASO 1: PREDICCIONES ML CON DATOS DE AGENTES
            # ============================================================================
            logger.info("")
            logger.info("=" * 80)
            logger.info("PASO 1/5: Ejecutando predicciones ML evolutivas...")
            logger.info("=" * 80)
            
            try:
                # Pasar agent_results al predictor
                ml_predictions = self.run_ml_predictions(
                    agent_results=agent_results,  # ← CLAVE: Pasar agent_results
                    generate_new=True  # ← SIEMPRE True para generar NUEVAS predicciones
                )
                
                logger.info(f" Predicciones ML completadas:")
                logger.info(f"   Registros: {len(ml_predictions)}")
                logger.info(f"   Métricas: {ml_predictions['metric'].nunique()}")
                logger.info(f"   Periodos: {ml_predictions['periodo'].max()}")
                
            except Exception as e:
                logger.error(f" Error en predicciones ML: {e}")
                raise
            
            # ============================================================================
            # PASO 2: VALIDACIÓN WALK-FORWARD (AJUSTADA PARA DATOS LIMITADOS)
            # ============================================================================
            logger.info("")
            logger.info("=" * 80)
            logger.info("PASO 2/5: Ejecutando validación walk-forward...")
            logger.info("=" * 80)
            
            try:
                # Validar predicciones
                validation_results = self.run_validation(
                    ml_predictions,
                    n_splits=3  # ← Reducido para datos anuales limitados
                )
                
                logger.info(f" Validación completada:")
                logger.info(f"   Periodos validados: {validation_results.get('validation_periods', 0)}")
                logger.info(f"   Mejor modelo: {validation_results.get('best_model', 'N/A')}")
                logger.info(f"   Ensemble MAE: {validation_results.get('ensemble_mae', 'N/A')}")
                logger.info(f"   Ensemble R²: {validation_results.get('ensemble_r2', 'N/A')}")
                
            except Exception as e:
                logger.error(f" Error en validación: {e}")
                validation_results = {}
            
            # ============================================================================
            # PASO 3: ANÁLISIS HÍBRIDO (LLM + ML + REGULATORIO)
            # ============================================================================
            logger.info("")
            logger.info("=" * 80)
            logger.info("PASO 3/5: Ejecutando análisis híbrido...")
            logger.info("=" * 80)
            
            try:
                # Pasar agent_results al análisis híbrido
                hybrid_results = await self.run_hybrid_analysis(
                    agent_results=agent_results  # ← CLAVE: Pasar agent_results
                )
                
                logger.info(f" Análisis híbrido completado:")
                
                # Procesar insights cualitativos
                llm_qualitative = hybrid_results.get('llm_qualitative', {})
                logger.info(f"   Categorías de insights: {len(llm_qualitative)}")
                
                for category, data in llm_qualitative.items():
                    if isinstance(data, dict) and 'findings' in data:
                        logger.info(f"      • {category}: {len(data['findings'])} findings")
                
                # Procesar recomendaciones
                recommendations = hybrid_results.get('recommendations', {})
                total_recs = sum(len(v) if isinstance(v, list) else 0 for v in recommendations.values())
                logger.info(f"   Recomendaciones integradas: {total_recs}")
                
                if isinstance(recommendations, dict):
                    logger.info(f"      • Strategic: {len(recommendations.get('strategic', []))}")
                    logger.info(f"      • Tactical: {len(recommendations.get('tactical', []))}")
                    logger.info(f"      • Risk Mitigation: {len(recommendations.get('risk_mitigation', []))}")
                
            except Exception as e:
                logger.error(f" Error en análisis híbrido: {e}")
                hybrid_results = {'llm_qualitative': {}, 'recommendations': {}}
            
            # ============================================================================
            # PASO 4: VALIDACIÓN AVANZADA (OPCIONAL)
            # ============================================================================
            logger.info("")
            logger.info("=" * 80)
            logger.info("PASO 4/5: Ejecutando validación avanzada (opcional)...")
            logger.info("=" * 80)
            
            advanced_validation = {}
            if run_advanced_validation:
                try:
                    advanced_validation = self.run_advanced_validation(ml_predictions)
                    logger.info(f" Validación avanzada completada")
                except Exception as e:
                    logger.warning(f" Error en validación avanzada (no crítico): {e}")
                    advanced_validation = {}
            else:
                logger.info("⊘ Validación avanzada omitida")
            
            # ============================================================================
            # PASO 5: CONSOLIDAR Y EXPORTAR RESULTADOS
            # ============================================================================
            logger.info("")
            logger.info("=" * 80)
            logger.info("PASO 5/5: Consolidando y exportando resultados...")
            logger.info("=" * 80)
            
            try:
                # Consolidar en estructura unificada
                self.consolidated_results = {
                    'timestamp': datetime.now().isoformat(),
                    'pipeline_status': 'completed',
                    'ml_predictions': ml_predictions.to_dict('records'),
                    'validation_results': validation_results,
                    'advanced_validation': advanced_validation,
                    'hybrid_analysis': hybrid_results,
                    'integrated_recommendations': hybrid_results.get('recommendations', {}),
                    'agent_results_processed': agent_results is not None,
                    'summary': {
                        'total_predictions': len(ml_predictions),
                        'total_recommendations': sum(
                            len(v) if isinstance(v, list) else 0 
                            for v in hybrid_results.get('recommendations', {}).values()
                        ),
                        'validation_periods': validation_results.get('validation_periods', 0),
                        'best_model': validation_results.get('best_model', 'ensemble'),
                        'ensemble_mae': validation_results.get('ensemble_mae', None),
                        'ensemble_r2': validation_results.get('ensemble_r2', None)
                    }
                }
                
                # Exportar a archivo
                output_path = self.consolidate_and_export(export_format='json')
                logger.info(f" Resultados exportados: {output_path}")
                
            except Exception as e:
                logger.error(f" Error consolidando resultados: {e}")
                raise
            
            # ============================================================================
            # RESUMEN FINAL
            # ============================================================================
            logger.info("")
            logger.info("=" * 80)
            logger.info(" PIPELINE COMPLETADO EXITOSAMENTE")
            logger.info("=" * 80)
            logger.info(f" Resumen final:")
            logger.info(f"   Predicciones generadas: {self.consolidated_results['summary']['total_predictions']}")
            logger.info(f"   Recomendaciones: {self.consolidated_results['summary']['total_recommendations']}")
            logger.info(f"   Períodos validados: {self.consolidated_results['summary']['validation_periods']}")
            logger.info(f"   Mejor modelo: {self.consolidated_results['summary']['best_model']}")
            logger.info(f"   Archivo: {output_path}")
            logger.info("=" * 80)
            
            return self.consolidated_results
            
        except Exception as e:
            logger.error("")
            logger.error("=" * 80)
            logger.error(f" ERROR CRÍTICO EN PIPELINE: {e}")
            logger.error("=" * 80)
            import traceback
            logger.error(traceback.format_exc())
            raise


# ============================================================================
# FUNCIONES DE EJECUCIÓN PRINCIPALES
# ============================================================================

async def run_garanti_bbva_analysis(
    bank_symbol: str = "GARAN.IS",
    jurisdiction: str = "TR",
    parent_bank: str = "BBVA",
    agent_results: Optional[Dict[str, Any]] = None,
    generate_new_predictions: bool = False,
    output_dir: str = "./data_outputs"
) -> Dict[str, Any]:
    """
    Función principal para ejecutar análisis de Garanti BBVA
    
    Args:
        bank_symbol: Símbolo del banco
        jurisdiction: Jurisdicción regulatoria
        parent_bank: Banco matriz
        agent_results: Resultados de agentes especializados
        generate_new_predictions: Si generar nuevas predicciones
        output_dir: Directorio de salida
        
    Returns:
        Diccionario con resultados consolidados
    """
    orchestrator = PredictorOrchestrator(
        bank_symbol=bank_symbol,
        jurisdiction=jurisdiction,
        parent_bank=parent_bank,
        data_dir=output_dir,
        output_dir=output_dir
    )
    
    results = await orchestrator.run_complete_pipeline(
        agent_results=agent_results,
        generate_new_predictions=generate_new_predictions
    )
    
    return results


def main():
    """Punto de entrada CLI para el orquestador predictor"""
    
    parser = argparse.ArgumentParser(
        description="Orquestador del Sistema Predictor Híbrido Multi-Agente"
    )
    
    parser.add_argument(
        '--bank-symbol',
        type=str,
        default='GARAN.IS',
        help='Símbolo del banco (default: GARAN.IS)'
    )
    
    parser.add_argument(
        '--jurisdiction',
        type=str,
        default='TR',
        help='Jurisdicción regulatoria (default: TR)'
    )
    
    parser.add_argument(
        '--parent-bank',
        type=str,
        default='BBVA',
        help='Banco matriz (default: BBVA)'
    )
    
    parser.add_argument(
        '--generate-new',
        action='store_true',
        help='Generar nuevas predicciones ML'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data_outputs',
        help='Directorio de salida (default: ./data_outputs)'
    )
    
    args = parser.parse_args()
    
    # Ejecutar pipeline
    asyncio.run(
        run_garanti_bbva_analysis(
            bank_symbol=args.bank_symbol,
            jurisdiction=args.jurisdiction,
            parent_bank=args.parent_bank,
            generate_new_predictions=args.generate_new,
            output_dir=args.output_dir
        )
    )


if __name__ == "__main__":
    main()
