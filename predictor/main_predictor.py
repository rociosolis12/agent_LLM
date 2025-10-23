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
    
    def run_ml_predictions(self, generate_new: bool = False) -> pd.DataFrame:
        """Carga predicciones ML existentes"""
        logger.info(" Ejecutando predicciones ML evolutivas...")
        
        # Ruta correcta a las predicciones
        csv_path = Path(self.output_dir) / "evolutionary_predictions.csv"
        
        # Cargar predicciones existentes
        if csv_path.exists():
            logger.info(" Cargando predicciones ML existentes...")
            ml_predictions = pd.read_csv(csv_path)
            logger.info(f" Predicciones ML cargadas: {len(ml_predictions)} registros")
        else:
            raise FileNotFoundError(f"No se encontró: {csv_path}")
        
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
            meta = self.consolidated_results['metadata']
            f.write(f"Fecha: {meta['execution_timestamp']}\n")
            f.write(f"Banco: {meta['bank_symbol']}\n")
            f.write(f"Jurisdicción: {meta['jurisdiction']}\n")
            f.write(f"Banco Matriz: {meta['parent_bank']}\n\n")
            
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
        generate_new_predictions: bool = False,
        run_advanced_validation: bool = True
    ) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo del sistema predictor
        
        Args:
            agent_results: Resultados de agentes financieros
            generate_new_predictions: Si generar nuevas predicciones
            run_advanced_validation: Si ejecutar validación avanzada
            
        Returns:
            Diccionario con todos los resultados consolidados
        """
        logger.info(" INICIANDO PIPELINE COMPLETO DEL SISTEMA PREDICTOR")
        logger.info("=" * 80)
        
        try:
            # 1. Predicciones ML
            ml_predictions = self.run_ml_predictions(
                generate_new=generate_new_predictions
            )
            
            # 2. Validación Walk-Forward
            validation_results = self.run_validation(ml_predictions)
            
            # 3. Análisis Híbrido
            hybrid_results = await self.run_hybrid_analysis(agent_results)
            
            # 4. Validación Avanzada (opcional)
            if run_advanced_validation:
                advanced_validation = self.run_advanced_validation(ml_predictions)
            
            # 5. Consolidar y exportar
            output_path = self.consolidate_and_export(export_format='json')
            
            return self.consolidated_results
            
        except Exception as e:
            logger.error(f" Error en pipeline: {e}")
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
