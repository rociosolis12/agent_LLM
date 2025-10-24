"""
main_system.py - Sistema Principal con REACT, Batch y Modo Interactivo

VERSIÓN COMPLETA con Integración del Extractor PDF y Agente Predictor
Incluye pipeline completo: PDF Extractor → Agentes Especializados → Predictor
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from datetime import datetime
import logging

# =============================
# CONFIGURACIÓN DE ENTORNO
# =============================

from dotenv import load_dotenv

# Cargar .env desde el directorio raíz del proyecto
project_root = Path(__file__).parent
env_path = project_root / ".env"
load_dotenv(env_path)

# Definir variables de configuración con valores por defecto seguros
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# Add agents directory to path
sys.path.append(str(Path(__file__).parent / "agents"))

# PIPELINE COMPLETO
from config import (
    DATA_INPUT_DIR, DATA_OUTPUT_DIR, REACT_STATES, PIPELINE_ORDER,
    PDF_EXTRACTOR_CONFIG, PREDICTOR_AGENT_CONFIG, FINANCIAL_AGENTS_CONFIG,
    EXECUTION_CONFIG, get_pdf_paths
)

from financial_coordinator import FinancialCoordinator
from extractor_pdf_agent import PDFExtractorAgent  

from predictor_agent import PredictorAgent

# ========== NUEVO: MÓDULO PREDICTOR HÍBRIDO ==========
try:
    from predictor.main_predictor import PredictorOrchestrator
    HYBRID_PREDICTOR_AVAILABLE = True
except ImportError as e:
    print(f" Módulo predictor híbrido no disponible: {e}")
    HYBRID_PREDICTOR_AVAILABLE = False

# Importar agentes especializados
from agents.balance_agent import BalanceREACTAgent
from agents.income_agent import IncomeREACTAgent
from agents. cashflows_agent import CashFlowsREACTAgent
from agents.equity_agent import EquityREACTAgent

# =============================
# CLASE PRINCIPAL 
# =============================

class FinancialExtractionSystem:
    """Sistema principal de extracción financiera con pipeline completo"""
    
    def __init__(self):
        self.logger = self._setup_logger()

        self.agents = {
            'balance': BalanceREACTAgent(),
            'income': IncomeREACTAgent(),
            'cashflows': CashFlowsREACTAgent(),
            'equity': EquityREACTAgent(),
        }
        
    
        self.pdf_extractor = PDFExtractorAgent()
        self.predictor = PredictorAgent()
        self.coordinator = FinancialCoordinator()

        if HYBRID_PREDICTOR_AVAILABLE:
            try:
                self.hybrid_predictor = PredictorOrchestrator(
                    bank_symbol="GARAN.IS",
                    jurisdiction="TR",
                    parent_bank="BBVA",
                    data_dir=str(DATA_OUTPUT_DIR),
                    output_dir=str(DATA_OUTPUT_DIR),
                    always_generate_new=True
                )
                self.logger.info(" Orquestador predictor híbrido inicializado")
            except Exception as e:
                self.logger.warning(f"Error inicializando predictor híbrido: {e}")
                self.hybrid_predictor = None
        else:
            self.hybrid_predictor = None
        
        self.results = {}
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Configurar logging del sistema"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    # Ejecutar pipeline completo
    async def run_complete_pipeline_with_hybrid_predictor(
        self, 
        question: str = None,
        generate_new_predictions: bool = True
    ) -> Dict:
        """
        Ejecutar pipeline completo con predictor híbrido avanzado
        
        Pipeline completo:
        ├── PASO 1: Extractor PDF
        │   └── Extrae estados financieros de PDFs
        ├── PASO 2: Agentes Especializados
        │   ├── BalanceAgent
        │   ├── IncomeAgent
        │   ├── CashFlowsAgent
        │   └── EquityAgent
        └── PASO 3: Predictor Híbrido 
            ├── 3.1: Generar NUEVAS predicciones ML (Prophet + XGBoost)
            ├── 3.2: Validación walk-forward
            ├── 3.3: Análisis híbrido (LLM + ML + Regulatory)
            └── 3.4: Consolidar y exportar resultados
        
        Args:
            question: Pregunta del usuario (opcional)
            generate_new_predictions: Si generar nuevas predicciones ML (default: True)
            
        Returns:
            Dict con resultados consolidados del pipeline completo
        """
        try:
            self.logger.info("="*80)
            self.logger.info(" INICIANDO PIPELINE COMPLETO CON PREDICTOR HÍBRIDO")
            self.logger.info("="*80)
            self.logger.info(" Modo: GENERACIÓN AUTOMÁTICA DE NUEVAS PREDICCIONES ML")
            
            pipeline_result = {
                "success": True,
                "pipeline_steps": [],
                "timestamp": datetime.now().isoformat(),
                "mode": "hybrid_predictor_advanced"
            }
            
            # ============================================================
            # PASO 1: EXTRACTOR PDF
            # ============================================================
            self.logger.info(" PASO 1/3: Ejecutando extractor PDF...")
            
            try:
                extraction_result = await self.pdf_extractor.extract_financial_statements()
                
                pipeline_result["pipeline_steps"].append({
                    "step": "pdf_extraction",
                    "step_number": 1,
                    "success": extraction_result["success"],
                    "details": extraction_result
                })
                
                if not extraction_result["success"]:
                    pipeline_result["success"] = False
                    pipeline_result["error"] = f"PDF extraction failed: {extraction_result.get('error', 'Unknown error')}"
                    self.logger.error(f" Error en extracción PDF: {pipeline_result['error']}")
                    return pipeline_result
                
                self.logger.info(f" PDF extraído: {extraction_result.get('pages_extracted', 0)} páginas procesadas")
                
            except Exception as e:
                self.logger.error(f" Error crítico en extracción PDF: {e}")
                pipeline_result["success"] = False
                pipeline_result["error"] = f"PDF extraction error: {str(e)}"
                return pipeline_result
            
            # ============================================================
            # PASO 2: AGENTES ESPECIALIZADOS
            # ============================================================
            self.logger.info(" PASO 2/3: Ejecutando agentes especializados...")
            
            try:
                if question:
                    self.logger.info(f"   Pregunta del usuario: {question}")
                    coordinator_result = await self.coordinator.process_question(question)
                else:
                    self.logger.info("   Ejecutando análisis general (sin pregunta específica)")
                    coordinator_result = await self.coordinator.process_request({
                        "type": "general_analysis",
                        "timestamp": datetime.now().isoformat()
                    })
                
                pipeline_result["pipeline_steps"].append({
                    "step": "specialized_agents",
                    "step_number": 2,
                    "success": coordinator_result.get("success", False),
                    "details": coordinator_result
                })
                
                agents_executed = len(coordinator_result.get("agents_results", {}))
                self.logger.info(f" Agentes especializados: {agents_executed} agentes ejecutados")
                
                if not coordinator_result.get("success"):
                    self.logger.warning(" Algunos agentes especializados fallaron, continuando con predictor...")
                
            except Exception as e:
                self.logger.error(f" Error en agentes especializados: {e}")
                pipeline_result["pipeline_steps"].append({
                    "step": "specialized_agents",
                    "step_number": 2,
                    "success": False,
                    "error": str(e)
                })
            
            # ============================================================
            # PASO 3: PREDICTOR HÍBRIDO AVANZADO
            # ============================================================
            self.logger.info(" PASO 3/3: Ejecutando predictor híbrido avanzado...")
            
            if self.hybrid_predictor and HYBRID_PREDICTOR_AVAILABLE:
                
                hybrid_predictor_results = None

                try:
                    self.logger.info("    Componente: EvolutionaryPredictorAgent (Prophet + XGBoost)")
                    self.logger.info("    Componente: WalkForwardValidator")
                    self.logger.info("    Componente: HybridPredictorAgent (LLM + ML)")
                    self.logger.info("    Componente: RegulatoryConfigAgent")
                    self.logger.info("    Generando NUEVAS predicciones ML...")
                    
                    # Preparar resultados de agentes para el predictor
                    agent_results = {
                        "agents_results": coordinator_result.get("agents_results", {}),
                        "structured_data": coordinator_result.get("structured_for_predictor", {}),
                        "pdf_extraction": extraction_result,
                        "question": question,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Ejecutar pipeline completo del predictor híbrido
                    # SIEMPRE con generate_new_predictions=True
                    hybrid_predictor_results = await self.hybrid_predictor.run_complete_pipeline(
                        agent_results=agent_results,
                        generate_new_predictions=True,  
                        run_advanced_validation=True
                    )
                    
                    pipeline_result["pipeline_steps"].append({
                        "step": "hybrid_predictor",
                        "step_number": 3,
                        "success": True,
                        "details": hybrid_predictor_results,
                        "new_predictions_generated": True,
                        "components": {
                            "ml_predictions": True,
                            "walk_forward_validation": True,
                            "hybrid_analysis": True,
                            "regulatory_config": True
                        }
                    })
                    
                    # Mostrar estadísticas del predictor
                    ml_pred_count = len(hybrid_predictor_results.get('ml_predictions', []))
                    validation_count = len(hybrid_predictor_results.get('validation_results', {}))
                    recommendations_count = len(hybrid_predictor_results.get('recommendations', {}))
                    
                    self.logger.info(" Predictor híbrido completado exitosamente:")
                    self.logger.info(f"   • Predicciones ML: {ml_pred_count}")
                    self.logger.info(f"   • Métricas validadas: {validation_count}")
                    self.logger.info(f"   • Recomendaciones: {recommendations_count}")
                    
                except Exception as e:
                    self.logger.error(f" Error en predictor híbrido: {e}")
                    import traceback
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    
                    pipeline_result["pipeline_steps"].append({
                        "step": "hybrid_predictor",
                        "step_number": 3,
                        "success": False,
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    })
                    
                    # No fallar todo el pipeline, solo marcar este paso como fallido
                    self.logger.warning(" Pipeline continuará con resultados parciales")
            else:
                # Predictor híbrido no disponible
                self.logger.error(" PREDICTOR HÍBRIDO NO DISPONIBLE")
                self.logger.error("   Verifica que los módulos estén instalados:")
                self.logger.error("   • main_predictor.py")
                self.logger.error("   • update_predictor_agent.py")
                self.logger.error("   • validation_module.py")
                self.logger.error("   • regulatory_config_agent.py")
                
                pipeline_result["pipeline_steps"].append({
                    "step": "hybrid_predictor",
                    "step_number": 3,
                    "success": False,
                    "error": "Hybrid predictor not available",
                    "reason": "HYBRID_PREDICTOR_AVAILABLE = False"
                })
                
                pipeline_result["success"] = False
                pipeline_result["error"] = "Hybrid predictor module not available"
            
            # ============================================================
            # COMPILAR RESULTADO FINAL
            # ============================================================
            self.logger.info(" Compilando resultados finales del pipeline...")
            
            pipeline_result.update({
                "pdf_extraction": extraction_result,
                "financial_analysis": coordinator_result,
                "hybrid_predictions": hybrid_predictor_results if self.hybrid_predictor else None,
                "total_steps_completed": len([s for s in pipeline_result["pipeline_steps"] if s["success"]]),
                "total_steps": len(pipeline_result["pipeline_steps"]),
                "execution_summary": self._create_execution_summary(pipeline_result)
            })
            
            self.logger.info("="*80)
            self.logger.info(" PIPELINE COMPLETO CON PREDICTOR HÍBRIDO FINALIZADO")
            self.logger.info(f"   Pasos completados: {pipeline_result['total_steps_completed']}/{pipeline_result['total_steps']}")
            self.logger.info("="*80)
            
            return pipeline_result
            
        except Exception as e:
            self.logger.error("="*80)
            self.logger.error(f"ERROR CRÍTICO EN PIPELINE COMPLETO: {str(e)}")
            self.logger.error("="*80)
            import traceback
            self.logger.error(traceback.format_exc())
            
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "pipeline_steps": pipeline_result.get("pipeline_steps", []),
                "traceback": traceback.format_exc()
            }

    def _count_generated_files(self) -> int:
        """Cuenta archivos generados en data_outputs"""
        try:
            output_dir = Path(DATA_OUTPUT_DIR)
            if output_dir.exists():
                return len(list(output_dir.glob("*.*")))
            return 0
        except Exception as e:
            self.logger.warning(f"No se pudo contar archivos: {e}")
            return 0


    def _create_execution_summary(self, pipeline_result: Dict) -> Dict:
        """
        Crear resumen ejecutivo mejorado incluyendo métricas del predictor híbrido
        
        Args:
            pipeline_result: Diccionario con resultados del pipeline
            
        Returns:
            Dict con resumen ejecutivo consolidado
        """
        successful_steps = sum(1 for step in pipeline_result["pipeline_steps"] if step.get("success"))
        total_steps = len(pipeline_result["pipeline_steps"])
        
        summary = {
            "pipeline_completion": f"{successful_steps}/{total_steps}",
            "success_rate": (successful_steps / total_steps) * 100 if total_steps > 0 else 0,
            "pdf_pages_processed": pipeline_result.get("pdf_extraction", {}).get("pages_extracted", 0),
            "agents_executed": len(pipeline_result.get("financial_analysis", {}).get("agents_results", {})),
            "files_generated": self._count_generated_files(),
            "mode": pipeline_result.get("mode", "standard"),
            "timestamp": pipeline_result.get("timestamp", datetime.now().isoformat())
        }
        
        # Agregar métricas del predictor híbrido si está disponible
        hybrid_preds = pipeline_result.get("hybrid_predictions", {})
        if hybrid_preds:
            ml_predictions = hybrid_preds.get("ml_predictions", [])
            validation_results = hybrid_preds.get("validation_results", {})
            recommendations = hybrid_preds.get("recommendations", {})
            
            summary.update({
                "ml_predictions_count": len(ml_predictions) if isinstance(ml_predictions, list) else 0,
                "validation_metrics_count": len(validation_results) if isinstance(validation_results, dict) else 0,
                "recommendations_count": len(recommendations) if isinstance(recommendations, dict) else 0,
                "hybrid_predictor_enabled": True
            })
        else:
            summary.update({
                "ml_predictions_count": 0,
                "validation_metrics_count": 0,
                "recommendations_count": 0,
                "hybrid_predictor_enabled": False
            })
        
        return summary

    # =============================
    # MÉTODOS EXISTENTES ACTUALIZADOS
    # =============================

    def sanity_check_pdfs(self, input_dir: Path) -> List[Path]:
        """Perform sanity check on PDFs in input directory"""
        print(" Performing sanity check on PDF files...")
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
        
        pdf_files = list(input_dir.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {input_dir}")
        
        valid_pdfs = []
        for pdf_file in pdf_files:
            try:
                import fitz
                doc = fitz.open(str(pdf_file))
                page_count = len(doc)
                doc.close()
                
                if page_count > 0:
                    valid_pdfs.append(pdf_file)
                    print(f" {pdf_file.name}: {page_count} pages")
                else:
                    print(f" {pdf_file.name}: Empty PDF")
            except Exception as e:
                print(f" {pdf_file.name}: Error reading - {str(e)}")
                continue
        
        if not valid_pdfs:
            raise ValueError("No valid PDF files found")
        
        print(f" Found {len(valid_pdfs)} valid PDF files")
        return valid_pdfs

    async def run_extraction_for_pdf(self, pdf_path: Path, agent_types: List[str] = None, question: str = None) -> Dict:
        """Run extraction for a specific PDF with optional question - ACTUALIZADO para usar pipeline"""
        if agent_types is None:
            agent_types = list(self.agents.keys())
        
        print(f"\n Starting extraction for: {pdf_path.name}")
        if question:
            print(f" Question: {question}")
        
        # PIPELINE COMPLETO en lugar de procesamiento individual
        pipeline_result = await self.run_complete_pipeline(question)
        
        if pipeline_result["success"]:
            print(f" Pipeline completo ejecutado para {pdf_path.name}")
            
            # Mostrar resumen de ejecución
            summary = pipeline_result["execution_summary"]
            print(f" Resumen: {summary['pipeline_completion']} pasos completados")
            print(f" Páginas PDF procesadas: {summary['pdf_pages_processed']}")
            print(f" Agentes ejecutados: {summary['agents_executed']}")
            if summary['predictions_generated'] > 0:
                print(f" Predicciones generadas: {summary['predictions_generated']}")
            
        else:
            print(f" Error en pipeline para {pdf_path.name}: {pipeline_result.get('error', 'Unknown error')}")
        
        return pipeline_result

    async def run_batch_extraction(self, input_dir: Path = None, agent_types: List[str] = None):
        """Run extraction for all PDFs in input directory - ACTUALIZADO"""
        if input_dir is None:
            input_dir = DATA_INPUT_DIR
        
        print(" Starting batch financial extraction process with COMPLETE PIPELINE")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {DATA_OUTPUT_DIR}")
        print(f"Pipeline steps: {' → '.join(PIPELINE_ORDER)}")
        
        # Sanity check
        valid_pdfs = self.sanity_check_pdfs(input_dir)
        
        # Process each PDF with complete pipeline
        for pdf_path in valid_pdfs:
            self.results[pdf_path.name] = await self.run_extraction_for_pdf(pdf_path, agent_types)
        
        # Generate enhanced summary report
        self._generate_enhanced_summary_report()

    def _generate_enhanced_summary_report(self):
        """Generate enhanced summary report with pipeline details"""
        print("\n" + "="*70)
        print(" COMPLETE PIPELINE EXTRACTION SUMMARY REPORT")
        print("="*70)
        
        total_pdfs = len(self.results)
        successful_pipelines = 0
        total_steps_completed = 0
        total_predictions = 0
        
        for pdf_name, result in self.results.items():
            print(f"\n {pdf_name}:")
            
            if result.get("success", False):
                successful_pipelines += 1
                summary = result.get("execution_summary", {})
                
                print(f"   Pipeline: {summary.get('pipeline_completion', 'N/A')}")
                print(f"   Success Rate: {summary.get('success_rate', 0):.1f}%")
                print(f"   Pages Processed: {summary.get('pdf_pages_processed', 0)}")
                print(f"   Agents Executed: {summary.get('agents_executed', 0)}")
                
                predictions = summary.get('predictions_generated', 0)
                if predictions > 0:
                    print(f"   Predictions: {predictions}")
                    total_predictions += predictions
                
                total_steps_completed += summary.get('pipeline_completion', '0/0').split('/')[0]
            else:
                print(f"   Pipeline Failed: {result.get('error', 'Unknown error')}")
        
        print(f"\n OVERALL PIPELINE STATISTICS:")
        print(f"  • Total PDFs processed: {total_pdfs}")
        print(f"  • Successful pipelines: {successful_pipelines}")
        print(f"  • Pipeline success rate: {(successful_pipelines/total_pdfs*100) if total_pdfs > 0 else 0:.1f}%")
        print(f"  • Total predictions generated: {total_predictions}")
        
        # List generated files
        output_files = list(DATA_OUTPUT_DIR.glob("*.csv"))
        if output_files:
            print(f"\n Generated files ({len(output_files)}):")
            for file in output_files:
                print(f"  • {file.name}")

# ============================================================================
# MODO INTERACTIVO CON PREDICTOR HÍBRIDO
# ============================================================================

async def interactive_mode_hybrid(system: FinancialExtractionSystem, use_hybrid: bool = True):
    """
    Modo interactivo optimizado para predictor híbrido
    Permite hacer múltiples preguntas sin reiniciar el sistema
    SIEMPRE GENERA NUEVAS PREDICCIONES ML en cada ejecución
    
    Args:
        system: Instancia de FinancialExtractionSystem
        use_hybrid: Si usar predictor híbrido (default: True)
    """
    
    # Banner de bienvenida
    print("\n" + "="*80)
    print("  SISTEMA MULTI-AGENTE FINANCIERO - MODO INTERACTIVO")
    if use_hybrid:
        print("  PREDICTOR HÍBRIDO ACTIVADO")
        print("  Generación automática de NUEVAS predicciones ML en cada consulta")
    else:
        print(" Modo básico (sin predictor híbrido)")
    print("="*80)
    
    # Verificar disponibilidad del predictor híbrido
    if use_hybrid and (not system.hybrid_predictor or not HYBRID_PREDICTOR_AVAILABLE):
        print("\n ADVERTENCIA: Predictor híbrido no disponible")
        print("   Verifica que los siguientes módulos existan:")
        print("   • main_predictor.py")
        print("   • update_predictor_agent.py")
        print("   • validation_module.py")
        print("   • hybrid_predictor_agent.py")
        print("   • regulatory_config_agent.py")
        print("\n   El sistema funcionará en modo básico\n")
        use_hybrid = False
    
    # Mostrar comandos especiales
    print("\n  COMANDOS ESPECIALES:")
    print(" • 'quit', 'exit', 'salir' → Cierra el sistema")
    print(" • 'help' → Muestra ayuda detallada")
    print(" • 'status' → Muestra estado del sistema y componentes")
    print(" • 'clear' → Limpia la pantalla")
    print(" • Cualquier otra cosa → Se interpreta como pregunta financiera")
    print("="*80)
    
    print("\n Sistema listo. Escribe tu pregunta o comando.\n")
    
    # Contador de preguntas
    question_count = 0
    
    # Loop interactivo infinito
    while True:
        try:
            # Solicitar pregunta al usuario
            if use_hybrid:
                prompt = f"\n Tu pregunta [HYBRID] [{question_count}]: "
            else:
                prompt = f"\n Tu pregunta [{question_count}]: "
            
            question = input(prompt).strip()
            
            # Ignorar entradas vacías
            if not question:
                continue
            
            # ========== COMANDOS ESPECIALES ==========
            
            # COMANDO: quit/exit/salir
            if question.lower() in ['quit', 'exit', 'salir']:
                print("\n" + "="*80)
                print(" Cerrando sistema. ¡Hasta luego!")
                print("="*80)
                break
            
            # COMANDO: help
            elif question.lower() == 'help':
                print("\n" + "="*80)
                print(" AYUDA DEL SISTEMA MULTI-AGENTE FINANCIERO")
                print("="*80)
                print("\n TIPOS DE PREGUNTAS QUE PUEDES HACER:")
                print("-" * 80)
                print("• Balance general y activos:")
                print("  - ¿Cuál es el total de activos?")
                print("  - ¿Cuánto efectivo tiene la empresa?")
                print("  - ¿Cuál es la estructura del balance?")
                print("\n• Estado de resultados e ingresos:")
                print("  - ¿Cuáles son los ingresos totales?")
                print("  - ¿Cuál es el margen de beneficio?")
                print("  - ¿Cómo han evolucionado las ventas?")
                print("\n• Flujos de caja:")
                print("  - ¿Cuál es el flujo de caja operativo?")
                print("  - ¿Hay flujo de caja positivo?")
                print("  - ¿Cuánto cash genera la empresa?")
                print("\n• Patrimonio y capital:")
                print("  - ¿Cuál es el patrimonio neto?")
                print("  - ¿Cómo está distribuido el capital?")
                print("  - ¿Hay dividendos?")
                
                if use_hybrid:
                    print("\n• Predicciones y proyecciones (PREDICTOR HÍBRIDO):")
                    print("  - ¿Cuál será el ROA proyectado?")
                    print("  - ¿Cómo evolucionará la solvencia?")
                    print("  - ¿Qué predicciones hay para el próximo trimestre?")
                
                print("\n" + "-" * 80)
                if use_hybrid:
                    print(" PIPELINE EJECUTADO POR CADA PREGUNTA:")
                    print("-" * 80)
                    print("  1. Extracción de datos PDF")
                    print("  2. Análisis de agentes especializados")
                    print("  3.  Generación de NUEVAS predicciones ML (Prophet + XGBoost)")
                    print("  4. Validación walk-forward temporal")
                    print("  5. Análisis híbrido (LLM + ML + Regulatorio)")
                    print("  6. Exportación de resultados consolidados")
                else:
                    print(" PIPELINE EJECUTADO POR CADA PREGUNTA:")
                    print("-" * 80)
                    print("  1. Extracción de datos PDF")
                    print("  2. Análisis de agentes especializados")
                    print("  3. Predictor básico")
                print("="*80)
                continue
            
            # COMANDO: status
            elif question.lower() == 'status':
                print("\n" + "="*80)
                print(" ESTADO DEL SISTEMA")
                print("="*80)
                
                print("\n COMPONENTES:")
                print("-" * 80)
                print(f" Predictor híbrido: {'ACTIVO' if use_hybrid and system.hybrid_predictor else 'INACTIVO'}")
                print(f" Generación automática ML: {'ACTIVO' if use_hybrid else 'INACTIVO'}")
                print(f" Agentes especializados: {len(system.agents)}")
                print(f"   - Balance Agent")
                print(f"   - Income Agent")
                print(f"   - CashFlows Agent")
                print(f"   - Equity Agent")
                print(f" Extractor PDF: DISPONIBLE")
                print(f" Coordinador: DISPONIBLE")
                
                if use_hybrid and system.hybrid_predictor:
                    print(f" Validador walk-forward: DISPONIBLE")
                    print(f" Agente regulatorio: DISPONIBLE")
                
                print("\n DIRECTORIOS:")
                print("-" * 80)
                print(f"   Entrada: {DATA_INPUT_DIR}")
                print(f"   Salida: {DATA_OUTPUT_DIR}")
                
                print("\n ESTADÍSTICAS:")
                print("-" * 80)
                print(f"   Preguntas procesadas en esta sesión: {question_count}")
                
                print("="*80)
                continue
            
            # COMANDO: clear
            elif question.lower() == 'clear':
                import os
                os.system('cls' if os.name == 'nt' else 'clear')
                print("\n Pantalla limpiada. Sistema listo.\n")
                continue
            
            # ========== PROCESAR PREGUNTA FINANCIERA ==========
            
            question_count += 1
            
            print("\n" + "="*80)
            print(f" PROCESANDO PREGUNTA {question_count}...")
            print("="*80)
            
            if use_hybrid:
                print(" Ejecutando pipeline completo con predictor híbrido...")
                print(" Generando NUEVAS predicciones ML automáticamente...\n")
                
                # Ejecutar pipeline con predictor híbrido
                result = await system.run_complete_pipeline_with_hybrid_predictor(
                    question=question
                )
            else:
                print(" Ejecutando pipeline estándar...\n")
                
                # Ejecutar pipeline estándar (sin predictor híbrido)
                result = await system.run_complete_pipeline(question)
            
            # Mostrar resultados
            print_pipeline_results(result)
            
            # Mostrar información de archivos generados
            if result.get("success"):
                print(f"\n RESULTADOS GUARDADOS EN: {DATA_OUTPUT_DIR}")
                print("-" * 80)
                print("   Archivos disponibles:")
                if use_hybrid:
                    print("   • consolidated_results_YYYYMMDD_HHMMSS.json")
                    print("   • evolutionary_predictions.csv ( NUEVAS predicciones)")
                    print("   • hybrid_analysis_YYYYMMDD_HHMMSS.json")
                    print("   • executive_summary_YYYYMMDD_HHMMSS.txt")
                else:
                    print("   • financial_analysis_results.json")
                print("-" * 80)
            
            print("\n Escribe otra pregunta o 'quit' para salir.\n")
            
        except KeyboardInterrupt:
            # Manejar Ctrl+C sin cerrar abruptamente
            print("\n\n" + "="*80)
            print(" Interrumpido por usuario (Ctrl+C)")
            print("="*80)
            
            confirm = input("\n¿Deseas salir del sistema? (s/n): ").strip().lower()
            
            if confirm in ['s', 'si', 'yes', 'y']:
                print("\n Cerrando sistema...")
                break
            else:
                print("\n Continuando en modo interactivo...\n")
                continue
        
        except Exception as e:
            # Manejar errores sin cerrar el sistema
            print("\n" + "="*80)
            print(f" ERROR AL PROCESAR PREGUNTA")
            print("="*80)
            print(f"Error: {str(e)}")
            
            # Mostrar traceback detallado
            import traceback
            print("\n DETALLES DEL ERROR:")
            print("-" * 80)
            traceback.print_exc()
            print("-" * 80)
            
            print("\n CONSEJOS:")
            print("   • Verifica los logs en: predictor_pipeline.log")
            print("   • Revisa que los archivos PDF estén disponibles")
            print("   • Comprueba la conectividad de las APIs")
            print("   • Intenta reformular tu pregunta")
            
            print("\n El sistema sigue activo. Puedes hacer otra pregunta.\n")
            continue
    
    # Mensaje de despedida final
    print("\n" + "="*80)
    print(f" RESUMEN DE LA SESIÓN")
    print("="*80)
    print(f"   Preguntas procesadas: {question_count}")
    print(f"   Directorio de salida: {DATA_OUTPUT_DIR}")
    print("="*80)
    print(" Gracias por usar el Sistema Multi-Agente Financiero")
    print("="*80 + "\n")


# ============================================================================
# FUNCIÓN HELPER PARA MOSTRAR RESULTADOS
# ============================================================================

def print_pipeline_results(result: Dict):
    """
    Mostrar resultados del pipeline de forma legible y estructurada
    
    Args:
        result: Diccionario con resultados del pipeline
    """
    print("\n" + "="*80)
    print(" RESULTADOS DEL PIPELINE")
    print("="*80)
    
    if result.get("success"):
        summary = result.get("execution_summary", {})
        
        # Información general
        print(f"\n Estado: EXITOSO")
        print(f" Pipeline: {summary.get('pipeline_completion', 'N/A')}")
        print(f" Tasa de éxito: {summary.get('success_rate', 0):.1f}%")
        print(f" Páginas PDF procesadas: {summary.get('pdf_pages_processed', 0)}")
        print(f" Agentes ejecutados: {summary.get('agents_executed', 0)}")
        
        # Métricas del predictor híbrido (si está disponible)
        if summary.get('hybrid_predictor_enabled') and summary.get('ml_predictions_count', 0) > 0:
            print("\n" + "-" * 80)
            print(" PREDICTOR HÍBRIDO:")
            print("-" * 80)
            print(f" NUEVAS predicciones ML generadas: {summary['ml_predictions_count']}")
            print(f" Métricas validadas: {summary.get('validation_metrics_count', 0)}")
            print(f" Recomendaciones generadas: {summary.get('recommendations_count', 0)}")
        
        # Archivos generados
        print("\n" + "-" * 80)
        print(f" Archivos generados totales: {summary.get('files_generated', 0)}")
        print("-" * 80)
        
        # Detalles de pasos ejecutados
        pipeline_steps = result.get('pipeline_steps', [])
        if pipeline_steps:
            print("\n PASOS EJECUTADOS:")
            print("-" * 80)
            for step in pipeline_steps:
                status_icon = "✅" if step.get('success') else "❌"
                step_name = step.get('step', 'unknown').replace('_', ' ').title()
                print(f"   {status_icon} {step_name}")
                
                # Mostrar info adicional de predictor híbrido
                if step.get('step') == 'hybrid_predictor' and step.get('new_predictions_generated'):
                    print(f"       Nuevas predicciones ML generadas")
                    components = step.get('components', {})
                    if components.get('ml_predictions'):
                        print(f"      ✓ Prophet + XGBoost")
                    if components.get('walk_forward_validation'):
                        print(f"      ✓ Validación temporal")
                    if components.get('regulatory_config'):
                        print(f"      ✓ Configuración regulatoria")
            print("-" * 80)
        
        # Timestamp
        print(f"\n Timestamp: {summary.get('timestamp', 'N/A')}")
        
    else:
        # Error en el pipeline
        print(f"\n Estado: ERROR")
        print(f" Error: {result.get('error', 'Unknown error')}")
        
        # Mostrar pasos que fallaron
        pipeline_steps = result.get('pipeline_steps', [])
        if pipeline_steps:
            print("\n PASOS EJECUTADOS:")
            print("-" * 80)
            for step in pipeline_steps:
                status_icon = "✅" if step.get('success') else "❌"
                step_name = step.get('step', 'unknown').replace('_', ' ').title()
                print(f"   {status_icon} {step_name}")
                if not step.get('success') and step.get('error'):
                    print(f"      Error: {step.get('error')}")
            print("-" * 80)
    
    print("="*80)


def show_enhanced_help():
    """Mostrar ayuda mejorada con información del pipeline"""
    print("\n EJEMPLOS DE PREGUNTAS - PIPELINE COMPLETO")
    print("="*60)
    
    print("\n ANÁLISIS GENERAL (ejecuta pipeline completo):")
    print("  • ¿Cuál es la situación financiera general?")
    print("  • Analiza todos los estados financieros")
    print("  • ¿Cuáles son las principales métricas financieras?")
    
    print("\n PREGUNTAS ESPECÍFICAS POR AGENTE:")
    print("  Balance: ¿Cuál es el total de activos?")
    print("  Income: ¿Cuál fue el beneficio neto del año?")  
    print("  Cashflows: ¿Cuánto efectivo generaron las operaciones?")
    print("  Equity: ¿Cómo cambió el patrimonio?")
    
    if PREDICTOR_AGENT_CONFIG['enabled']:
        print("\n PREGUNTAS PREDICTIVAS:")
        print("  • ¿Cuál será la tendencia futura de ingresos?")
        print("  • ¿Qué riesgos financieros se anticipan?")
        print("  • ¿Cuál es la proyección de crecimiento?")
        print("  • ¿Cómo evolucionará la rentabilidad?")
    
    print("\n COMANDOS DE CONTROL:")
    print("  • 'extract' - Solo ejecutar extractor PDF")
    print("  • 'pipeline' - Ver estado del pipeline")
    print("  • 'predict on/off' - Controlar predicciones")

async def show_system_status(system):
    """Mostrar estado completo del sistema"""
    print("\n ESTADO DEL SISTEMA MULTI-AGENTE COMPLETO")
    print("="*50)
    
    # Estado de PDFs
    pdf_paths = get_pdf_paths()
    input_exists = Path(pdf_paths['input_pdf']).exists()
    output_exists = Path(pdf_paths['output_pdf']).exists()
    
    print(f" PDF fuente: {'✅' if input_exists else '❌'} {pdf_paths['input_pdf']}")
    print(f" PDF extraído: {'✅' if output_exists else '❌'} {pdf_paths['output_pdf']}")
    
    # Estado de agentes
    print(f"\n AGENTES DISPONIBLES:")
    print(f"   Extractor PDF: ✅ {system.pdf_extractor.__class__.__name__}")
    print(f"   Agentes especializados: {len(system.agents)} disponibles")
    print(f"   Predictor: {'✅' if PREDICTOR_AGENT_CONFIG['enabled'] else '❌'} {system.predictor.__class__.__name__}")
    
    # Pipeline status
    print(f"\n PIPELINE CONFIGURATION:")
    print(f"   Orden: {' → '.join(PIPELINE_ORDER)}")
    print(f"   Ejecución paralela: {'✅' if EXECUTION_CONFIG['parallel_execution']['enabled'] else '❌'}")
    print(f"   Predicciones: {'✅' if PREDICTOR_AGENT_CONFIG['enabled'] else '❌'}")
    
    # Archivos disponibles
    output_files = list(DATA_OUTPUT_DIR.glob("*.csv"))
    print(f"\n ARCHIVOS GENERADOS ({len(output_files)}):")
    for file in output_files[:3]:
        print(f"  • {file.name}")
    if len(output_files) > 3:
        print(f"  • ... y {len(output_files) - 3} más")

def show_agents_info(system):
    """Mostrar información detallada de agentes"""
    print("\n INFORMACIÓN DETALLADA DE AGENTES")
    print("="*40)
    
    print(f" PDF EXTRACTOR AGENT:")
    print(f"  Estado:  Activo")
    print(f"  Función: Extraer páginas 54-60 de estados financieros")
    print(f"  Entrada: {PDF_EXTRACTOR_CONFIG['input_path']}")
    print(f"  Salida: {PDF_EXTRACTOR_CONFIG['output_path']}")
    
    print(f"\n AGENTES ESPECIALIZADOS ({len(system.agents)}):")
    for agent_name, agent_class in system.agents.items():
        print(f"  {agent_name.upper()}: ✅ {agent_class.__name__}")
    
    if PREDICTOR_AGENT_CONFIG['enabled']:
        print(f"\n PREDICTOR AGENT:")
        print(f"  Estado:  Activo")
        print(f"  Horizonte: {PREDICTOR_AGENT_CONFIG['prediction_horizon']} meses")
        print(f"  Tipos predicción: {len(PREDICTOR_AGENT_CONFIG['prediction_types'])}")

def show_pipeline_status():
    """Mostrar estado detallado del pipeline"""
    print("\n ESTADO DEL PIPELINE COMPLETO")
    print("="*35)
    
    for i, step in enumerate(PIPELINE_ORDER, 1):
        if step == 'pdf_extractor':
            status = "✅" if PDF_EXTRACTOR_CONFIG['agent_enabled'] else "❌"
            print(f"{i}.  {step.upper()}: {status}")
        elif step in ['balance_agent', 'income_agent', 'equity_agent', 'cashflow_agent']:
            print(f"{i}.  {step.upper()}: ✅")
        elif step == 'predictor_agent':
            status = "✅" if PREDICTOR_AGENT_CONFIG['enabled'] else "❌"
            print(f"{i}.  {step.upper()}: {status}")
    
    print(f"\n CONFIGURACIÓN:")
    print(f"  Pasos totales: {len(PIPELINE_ORDER)}")
    print(f"  Ejecución paralela: {'Sí' if EXECUTION_CONFIG['parallel_execution']['enabled'] else 'No'}")
    print(f"  Timeout por agente: {EXECUTION_CONFIG['parallel_execution']['timeout_per_agent']}s")

# =============================
# FUNCIONES AUXILIARES Y WRAPPERS
# =============================

async def batch_mode_wrapper(args):
    """Wrapper para el modo batch con pipeline completo"""
    system = FinancialExtractionSystem()
    
    try:
        if args.validate_only:
            # Run validation only
            input_dir = Path(args.input_dir) if args.input_dir else DATA_INPUT_DIR
            system.sanity_check_pdfs(input_dir)
            print(" Validation completed successfully")
            
        elif args.pdf:
            # Process specific PDF with complete pipeline
            pdf_path = Path(args.pdf)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            result = await system.run_extraction_for_pdf(pdf_path, args.agents, args.question)
            print(f" Complete pipeline processing completed for {pdf_path.name}")
            
        else:
            # Run batch extraction with complete pipeline
            input_dir = Path(args.input_dir) if args.input_dir else DATA_INPUT_DIR
            await system.run_batch_extraction(input_dir, args.agents)
            print(" Batch extraction with complete pipeline completed")
            
    except Exception as e:
        print(f" System error: {str(e)}")
        sys.exit(1)
# ============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ============================================================================

def main():
    """
    Punto de entrada principal del sistema multi-agente financiero
    Soporta múltiples modos: interactivo, batch, pregunta única
    """
    
    # Configuración de argumentos CLI
    parser = argparse.ArgumentParser(
        description="Sistema Multi-Agente de Análisis Financiero con Predictor Híbrido",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  
  # Modo interactivo con predictor híbrido (RECOMENDADO)
  python main_system.py --interactive
  
  # Pregunta única con predictor híbrido
  python main_system.py --question "¿Cuál es el ROA proyectado?"
  
  # Modo batch (procesar todos los PDFs)
  python main_system.py --batch
  
  # Modo básico sin predictor híbrido
  python main_system.py --no-hybrid --question "¿Cuál es el total de activos?"
  
  # Solo extracción PDF
  python main_system.py --extract-only
        """
    )
    
    # ===== MODOS DE OPERACIÓN =====
    mode_group = parser.add_argument_group('Modos de operación')
    mode_group.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Modo interactivo (permite múltiples preguntas)'
    )
    mode_group.add_argument(
        '--batch', '-b',
        action='store_true',
        help='Procesamiento batch de todos los PDFs'
    )
    mode_group.add_argument(
        '--question', '-q',
        type=str,
        help='Pregunta específica (modo no interactivo)'
    )
    mode_group.add_argument(
        '--extract-only',
        action='store_true',
        help='Solo ejecutar extractor PDF (sin análisis)'
    )
    
    # ===== CONFIGURACIÓN DEL PREDICTOR =====
    predictor_group = parser.add_argument_group('Configuración del predictor')
    predictor_group.add_argument(
        '--no-hybrid',
        action='store_true',
        help='Desactivar predictor híbrido (usar predictor básico)'
    )
    predictor_group.add_argument(
        '--no-predict',
        action='store_true',
        help='Deshabilitar predicciones completamente'
    )
    
    # ===== ARGUMENTOS OPCIONALES =====
    optional_group = parser.add_argument_group('Opciones adicionales')
    optional_group.add_argument(
        '--pdf',
        type=str,
        help='Ruta específica al PDF a analizar'
    )
    optional_group.add_argument(
        '--output-dir',
        type=str,
        default=str(DATA_OUTPUT_DIR),
        help=f'Directorio de salida (default: {DATA_OUTPUT_DIR})'
    )
    optional_group.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Mostrar información detallada del proceso'
    )
    
    args = parser.parse_args()
    
    # ===== BANNER INICIAL =====
    print("\n" + "="*80)
    print("  SISTEMA MULTI-AGENTE DE ANÁLISIS FINANCIERO")
    print("="*80)
    print(f" Versión: 4.0 - Pipeline Completo")
    print(f" Configuración: Groq {GROQ_MODEL} + Azure {AZURE_OPENAI_DEPLOYMENT}")
    print(f" Pipeline: {' → '.join(PIPELINE_ORDER)}")
    print("="*80)
    
    # ===== DETERMINAR SI USAR PREDICTOR HÍBRIDO =====
    use_hybrid = not args.no_hybrid and HYBRID_PREDICTOR_AVAILABLE
    
    if not use_hybrid and not args.no_hybrid:
        print("\n ADVERTENCIA: Predictor híbrido no disponible")
        print("   El sistema funcionará en modo básico")
    elif args.no_hybrid:
        print("\n ℹPredictor híbrido desactivado manualmente")
    elif use_hybrid:
        print("\n Predictor híbrido ACTIVADO")
        print("   Generación automática de nuevas predicciones ML")
    
    print("="*80 + "\n")
    
    try:
        # ===== CREAR INSTANCIA DEL SISTEMA =====
        system = FinancialExtractionSystem()
        
        # ===== MODO 1: SOLO EXTRACCIÓN PDF =====
        if args.extract_only:
            print(" Modo: Solo extracción PDF\n")
            result = asyncio.run(system.pdf_extractor.extract_financial_statements())
            
            if result["success"]:
                print("\nPDF extraído exitosamente:")
                print(f"   Páginas procesadas: {result.get('total_pages_extracted', 0)}")
                print(f"   Archivo generado: {result.get('output_file', 'N/A')}")
            else:
                print(f"\n Error en extracción: {result.get('error', 'Unknown error')}")
            
            return
        
        # ===== MODO 2: BATCH =====
        elif args.batch:
            print(" Modo: Procesamiento batch\n")
            result = asyncio.run(system.run_batch_extraction())
            return
        
        # ===== MODO 3: INTERACTIVO =====
        elif args.interactive:
            print(" Modo: Interactivo")
            if use_hybrid:
                print("Con predictor híbrido avanzado\n")
            else:
                print(" Con predictor básico\n")
            
            asyncio.run(interactive_mode_hybrid(system, use_hybrid=use_hybrid))
            return
        
        # ===== MODO 4: PREGUNTA ÚNICA =====
        elif args.question:
            print(f" Modo: Pregunta única")
            print(f" Pregunta: {args.question}\n")
            
            if use_hybrid:
                print(" Ejecutando pipeline con predictor híbrido...\n")
                result = asyncio.run(
                    system.run_complete_pipeline_with_hybrid_predictor(
                        question=args.question
                    )
                )
            else:
                print(" Ejecutando pipeline estándar...\n")
                result = asyncio.run(
                    system.run_complete_pipeline(args.question)
                )
            
            # Mostrar resultados
            print_pipeline_results(result)
            
            # Mostrar respuesta principal si existe
            if result.get("success"):
                financial_analysis = result.get("financial_analysis", {})
                if financial_analysis.get("answer"):
                    print("\n" + "="*80)
                    print(" RESPUESTA")
                    print("="*80)
                    print(financial_analysis["answer"])
                    print("="*80)
            
            return
        
        # ===== MODO 5: POR DEFECTO (INTERACTIVO) =====
        else:
            print(" Modo por defecto: Interactivo")
            print("   (Usa --help para ver otras opciones)\n")
            
            if use_hybrid:
                print("🔬 Predictor híbrido ACTIVADO\n")
            
            asyncio.run(interactive_mode_hybrid(system, use_hybrid=use_hybrid))
            return
    
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print(" Proceso interrumpido por el usuario")
        print("="*80)
        sys.exit(0)
    
    except Exception as e:
        print("\n" + "="*80)
        print(" ERROR CRÍTICO")
        print("="*80)
        print(f"Error: {str(e)}")
        
        if args.verbose:
            import traceback
            print("\n TRACEBACK COMPLETO:")
            print("-" * 80)
            traceback.print_exc()
            print("-" * 80)
        
        print("\n CONSEJOS:")
        print("   • Verifica que los archivos PDF estén disponibles")
        print("   • Revisa los logs del sistema")
        print("   • Usa --verbose para más detalles")
        print("   • Contacta al administrador si el error persiste")
        print("="*80)
        
        sys.exit(1)


if __name__ == "__main__":
    main()
