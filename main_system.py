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

# Importar agentes especializados
from balance_agent import BalanceREACTAgent
from income_agent import IncomeREACTAgent
from cashflows_agent import CashFlowsREACTAgent
from equity_agent import EquityREACTAgent

# =============================
# CLASE PRINCIPAL 
# =============================

class FinancialExtractionSystem:
    """Sistema principal de extracción financiera con pipeline completo"""
    
    def __init__(self):
        self.agents = {
            'balance': BalanceREACTAgent,
            'income': IncomeREACTAgent,
            'cashflows': CashFlowsREACTAgent,
            'equity': EquityREACTAgent,
        }
        
    
        self.pdf_extractor = PDFExtractorAgent()
        self.predictor = PredictorAgent()
        self.coordinator = FinancialCoordinator()
        
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
    async def run_complete_pipeline(self, question: str = None) -> Dict:
        """
        Ejecutar pipeline completo: PDF Extractor → Agentes Especializados → Predictor
        """
        try:
            self.logger.info(" Iniciando pipeline completo multi-agente")
            
            pipeline_result = {
                "success": True,
                "pipeline_steps": [],
                "timestamp": datetime.now().isoformat()
            }
            
            # PASO 1: EXTRACTOR PDF
            self.logger.info(" Ejecutando extractor PDF...")
            extraction_result = await self.pdf_extractor.extract_financial_statements()
            
            pipeline_result["pipeline_steps"].append({
                "step": "pdf_extraction",
                "success": extraction_result["success"],
                "details": extraction_result
            })
            
            if not extraction_result["success"]:
                pipeline_result["success"] = False
                pipeline_result["error"] = f"PDF extraction failed: {extraction_result['error']}"
                return pipeline_result
            
            # PASO 2: AGENTES ESPECIALIZADOS
            self.logger.info(" Ejecutando agentes especializados...")
            
            if question:
                coordinator_result = await self.coordinator.process_question(question)
            else:
                # Ejecutar análisis general
                coordinator_result = await self.coordinator.process_request({
                    "type": "general_analysis",
                    "timestamp": datetime.now().isoformat()
                })
            
            pipeline_result["pipeline_steps"].append({
                "step": "specialized_agents",
                "success": coordinator_result["success"],
                "details": coordinator_result
            })
            
            if not coordinator_result["success"]:
                self.logger.warning("⚠️ Algunos agentes especializados fallaron, continuando...")
            
            # PASO 3: AGENTE PREDICTOR (si está habilitado)
            if PREDICTOR_AGENT_CONFIG['enabled']:
                self.logger.info(" Ejecutando agente predictor...")
                
                predictor_input = {
                    "financial_data": {
                        "agents_results": coordinator_result.get("agents_results", {}),
                        "structured_for_predictor": coordinator_result.get("structured_for_predictor", {})
                    },
                    "config": PREDICTOR_AGENT_CONFIG,
                    "request_metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "question": question,
                        "data_source": "extracted_pdf_statements"
                    }
                }
                
                predictor_result = await self.predictor.generate_predictions(predictor_input)
                
                pipeline_result["pipeline_steps"].append({
                    "step": "predictor",
                    "success": predictor_result["success"],
                    "details": predictor_result
                })
            else:
                pipeline_result["pipeline_steps"].append({
                    "step": "predictor",
                    "success": True,
                    "skipped": True,
                    "reason": "Predictor disabled in config"
                })
            
            # COMPILAR RESULTADO FINAL
            pipeline_result.update({
                "pdf_extraction": extraction_result,
                "financial_analysis": coordinator_result,
                "predictions": predictor_result if PREDICTOR_AGENT_CONFIG['enabled'] else None,
                "total_steps_completed": len([s for s in pipeline_result["pipeline_steps"] if s["success"]]),
                "execution_summary": self._create_execution_summary(pipeline_result)
            })
            
            self.logger.info(" Pipeline completo ejecutado exitosamente")
            return pipeline_result
            
        except Exception as e:
            self.logger.error(f" Error en pipeline completo: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "pipeline_steps": pipeline_result.get("pipeline_steps", [])
            }
    
    def _create_execution_summary(self, pipeline_result: Dict) -> Dict:
        """Crear resumen ejecutivo del pipeline"""
        successful_steps = sum(1 for step in pipeline_result["pipeline_steps"] if step["success"])
        total_steps = len(pipeline_result["pipeline_steps"])
        
        return {
            "pipeline_completion": f"{successful_steps}/{total_steps}",
            "success_rate": (successful_steps / total_steps) * 100 if total_steps > 0 else 0,
            "pdf_pages_processed": pipeline_result.get("pdf_extraction", {}).get("total_pages_extracted", 0),
            "agents_executed": len(pipeline_result.get("financial_analysis", {}).get("agents_results", {})),
            "predictions_generated": len(pipeline_result.get("predictions", {}).get("predictions", {})) if pipeline_result.get("predictions") else 0,
            "files_generated": self._count_generated_files(),
            "execution_time": "calculated_in_production"  # Se calcularía en tiempo real
        }
    
    def _count_generated_files(self) -> int:
        """Contar archivos generados en directorio de salida"""
        try:
            output_files = list(DATA_OUTPUT_DIR.glob("*.csv"))
            return len(output_files)
        except:
            return 0

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

# =============================
# MODO INTERACTIVO MEJORADO CON PIPELINE COMPLETO
# =============================

async def interactive_mode():
    """Modo interactivo mejorado con pipeline completo"""
    print("\n" + "="*70)
    print(" SISTEMA MULTI-AGENTE FINANCIERO - MODO INTERACTIVO COMPLETO")
    print("="*70)
    print(" Comandos especiales:")
    print("  • 'quit', 'exit', 'salir' - Salir del sistema")
    print("  • 'help', 'ayuda' - Mostrar ejemplos de preguntas")
    print("  • 'status' - Estado del sistema y agentes")
    print("  • 'agents' - Lista de agentes disponibles")
    print("  • 'pipeline' - Mostrar estado del pipeline")
    print("  • 'extract' - Ejecutar solo extractor PDF")
    print("  • 'predict on/off' - Activar/desactivar predicciones")
    print("="*70)

    # Inicializar sistema
    system = FinancialExtractionSystem()
    
    # Verificar estado inicial
    pdf_paths = get_pdf_paths()
    input_pdf_exists = Path(pdf_paths['input_pdf']).exists()
    
    print(" SISTEMA LISTO - Pipeline completo disponible")
    print(f" PDF fuente: {' Disponible' if input_pdf_exists else ' No encontrado'}")
    print(f" Predicciones: {' Habilitadas' if PREDICTOR_AGENT_CONFIG['enabled'] else ' Deshabilitadas'}")
    
    predictions_enabled = PREDICTOR_AGENT_CONFIG['enabled']
    
    while True:
        try:
            print("\n" + "-"*50)
            prompt = " Tu pregunta (pipeline completo): " if predictions_enabled else " Tu pregunta: "
            question = input(prompt).strip()
            
            if not question:
                continue
                
            # Comandos especiales
            if question.lower() in ['quit', 'exit', 'salir']:
                print(" ¡Hasta luego! Gracias por usar el sistema.")
                break
                
            elif question.lower() in ['help', 'ayuda', '?']:
                show_enhanced_help()
                continue
                
            elif question.lower() in ['status', 'estado']:
                await show_system_status(system)
                continue
                
            elif question.lower() in ['agents', 'agentes']:
                show_agents_info(system)
                continue
                
            elif question.lower() == 'pipeline':
                show_pipeline_status()
                continue
                
            elif question.lower() == 'extract':
                print(" Ejecutando solo extractor PDF...")
                result = await system.pdf_extractor.extract_financial_statements()
                if result["success"]:
                    print(f" PDF extraído: {result['pages_extracted']} páginas")
                else:
                    print(f" Error: {result['error']}")
                continue
                
            elif question.lower().startswith('predict'):
                if 'on' in question.lower():
                    predictions_enabled = True
                    print(" Predicciones activadas")
                elif 'off' in question.lower():
                    predictions_enabled = False
                    print(" Solo análisis histórico activado")
                else:
                    print(f" Estado predicciones: {'ON' if predictions_enabled else 'OFF'}")
                continue

            # Procesar pregunta con pipeline completo
            print(" Ejecutando pipeline completo...")
            
            pipeline_result = await system.run_complete_pipeline(question)
            
            if pipeline_result["success"]:
                print(" PIPELINE COMPLETADO EXITOSAMENTE")
                
                # Mostrar resumen
                summary = pipeline_result["execution_summary"]
                print(f" Pipeline: {summary['pipeline_completion']} pasos")
                print(f" Páginas procesadas: {summary['pdf_pages_processed']}")
                print(f" Agentes ejecutados: {summary['agents_executed']}")
                
                # Mostrar respuesta de coordinador
                financial_analysis = pipeline_result.get("financial_analysis", {})
                if financial_analysis.get("answer"):
                    print(f"\n RESPUESTA:")
                    print(financial_analysis["answer"])
                
                # Mostrar predicciones si están disponibles
                predictions = pipeline_result.get("predictions", {})
                if predictions and predictions.get("success") and predictions_enabled:
                    print(f"\n PREDICCIONES GENERADAS:")
                    pred_count = len(predictions.get("predictions", {}))
                    print(f" Total predicciones: {pred_count}")
                    
                    # Mostrar algunas predicciones clave
                    preds = predictions.get("predictions", {})
                    if "revenue_growth" in preds:
                        growth = preds["revenue_growth"]
                        print(f" Crecimiento ingresos: {growth.get('predicted_growth_rate', 'N/A')}")
                    
                    if "risk_assessment" in preds:
                        risk = preds["risk_assessment"]
                        print(f" Nivel de riesgo: {risk.get('overall_risk_level', 'N/A')}")
                
                # Archivos generados
                if summary.get('files_generated', 0) > 0:
                    print(f" Archivos generados: {summary['files_generated']}")
                    
            else:
                print(f" ERROR EN PIPELINE: {pipeline_result.get('error', 'Error desconocido')}")
                
                # Mostrar qué pasos fallaron
                failed_steps = [s["step"] for s in pipeline_result.get("pipeline_steps", []) if not s["success"]]
                if failed_steps:
                    print(f" Pasos fallidos: {', '.join(failed_steps)}")
                    
        except KeyboardInterrupt:
            print("\n\n Interrumpido por el usuario. ¡Hasta luego!")
            break
        except Exception as e:
            print(f" Error inesperado: {e}")

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
            print(f"{i}. 📄 {step.upper()}: {status}")
        elif step in ['balance_agent', 'income_agent', 'equity_agent', 'cashflow_agent']:
            print(f"{i}. 🤖 {step.upper()}: ✅")
        elif step == 'predictor_agent':
            status = "✅" if PREDICTOR_AGENT_CONFIG['enabled'] else "❌"
            print(f"{i}. 🔮 {step.upper()}: {status}")
    
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

# =============================
# FUNCIÓN MAIN COMPLETA
# =============================

async def main():
    """Función main con soporte para pipeline completo"""
    
    # ===== CONFIGURACIÓN PREDEFINIDA =====
    DEFAULT_CONFIG = {
        "pdf": str(Path(PDF_EXTRACTOR_CONFIG['output_path']) / PDF_EXTRACTOR_CONFIG['output_filename']),
        "out": str(DATA_OUTPUT_DIR),
        "anchor_page": 55,
        "anchor_titles": ["statement of financial position", "garantibank international"],
        "max_steps": 20
    }

    parser = argparse.ArgumentParser(
        description="Sistema Multi-Agente Financiero - Pipeline Completo con Extractor PDF y Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

MODO INTERACTIVO (recomendado):
  python main_system.py --interactive

PIPELINE COMPLETO:
  python main_system.py --question "¿Cuál es la situación financiera?"
  python main_system.py --question "Analiza todos los estados financieros"

PREGUNTAS ESPECÍFICAS:
  python main_system.py --question "¿Cuál es el total de activos?"
  python main_system.py --question "¿Cuál fue el beneficio neto?" --agent balance

PREGUNTAS PREDICTIVAS:
  python main_system.py --question "¿Cuál será la tendencia futura?"
  python main_system.py --question "¿Qué riesgos se anticipan?"

MODO BATCH CON PIPELINE:
  python main_system.py --batch
  python main_system.py --pdf documento.pdf --agents balance,income

CARACTERÍSTICAS NUEVAS:
- Pipeline completo: PDF Extractor → Agentes Especializados → Predictor
- Extracción automática de estados financieros (páginas 54-60)
- Análisis predictivo integrado
- Ejecución paralela de agentes especializados
- Monitoreo completo del pipeline
- Modo interactivo mejorado con comandos de control
        """
    )

    # ===== ARGUMENTOS PRINCIPALES =====
    parser.add_argument("--question", type=str, help="Pregunta específica para el análisis financiero")
    parser.add_argument("--pdf", default=DEFAULT_CONFIG["pdf"], help=f"Ruta al PDF a analizar (por defecto: PDF extraído)")
    parser.add_argument("--agent", choices=['balance', 'income', 'cashflows', 'equity'], help="Forzar uso de agente específico")
    parser.add_argument("--agents", help="Lista de agentes a ejecutar (separados por comas)")

    # ===== ARGUMENTOS DE PIPELINE =====
    parser.add_argument("--pipeline", action='store_true', help="Ejecutar pipeline completo (PDF → Agentes → Predictor)")
    parser.add_argument("--extract-only", action='store_true', help="Solo ejecutar extractor PDF")
    parser.add_argument("--no-predict", action='store_true', help="Deshabilitar predicciones para esta ejecución")

    # ===== MODOS DE OPERACIÓN =====
    parser.add_argument("--interactive", action='store_true', help="Modo interactivo mejorado con pipeline completo")
    parser.add_argument("--batch", action='store_true', help="Modo batch con pipeline completo")

    # ===== ARGUMENTOS OPCIONALES =====
    parser.add_argument("--out", default=DEFAULT_CONFIG["out"], help=f"Directorio de salida (por defecto: {DEFAULT_CONFIG['out']})")
    parser.add_argument("--input_dir", help="Directorio de entrada para modo batch")
    parser.add_argument("--validate_only", action='store_true', help="Solo validar PDFs sin procesarlos")
    parser.add_argument("--verbose", action='store_true', help="Mostrar información detallada del proceso")

    args = parser.parse_args()

    # ===== CONFIGURAR RUTAS =====
    pdf_path = Path(args.pdf)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===== MOSTRAR CONFIGURACIÓN =====
    print(f" Sistema Multi-Agente Financiero v4.0 - Pipeline Completo")
    print(f" PDF: {pdf_path}")
    print(f" Salida: {output_dir}")
    print(f" Configuración: Groq {GROQ_MODEL} + Azure {AZURE_OPENAI_DEPLOYMENT}")
    print(f" Pipeline: {' → '.join(PIPELINE_ORDER)}")
    print("="*60)

    try:
        # Crear sistema
        system = FinancialExtractionSystem()
        
        # ===== DETERMINAR MODO DE OPERACIÓN =====
        if args.interactive:
            # Modo interactivo mejorado con pipeline completo
            await interactive_mode()
            
        elif args.extract_only:
            # Solo ejecutar extractor PDF
            print(" Ejecutando solo extractor PDF...")
            result = await system.pdf_extractor.extract_financial_statements()
            
            if result["success"]:
                print(f" PDF extraído exitosamente:")
                print(f"   Páginas procesadas: {result['total_pages_extracted']}")
                print(f"   Archivo generado: {result['output_file']}")
            else:
                print(f" Error en extracción: {result['error']}")
                
        elif args.question:
            # Procesar pregunta con pipeline completo
            print(f" Pregunta: {args.question}")
            print(" Ejecutando pipeline completo...")
            
            pipeline_result = await system.run_complete_pipeline(args.question)
            
            if pipeline_result["success"]:
                print(" PIPELINE COMPLETADO EXITOSAMENTE")
                
                # Mostrar respuesta principal
                financial_analysis = pipeline_result.get("financial_analysis", {})
                if financial_analysis.get("answer"):
                    print(f"\n RESPUESTA:")
                    print(financial_analysis["answer"])
                
                # Mostrar resumen de ejecución
                summary = pipeline_result["execution_summary"]
                print(f"\n RESUMEN DE EJECUCIÓN:")
                print(f"  Pipeline: {summary['pipeline_completion']}")
                print(f"  Páginas procesadas: {summary['pdf_pages_processed']}")
                print(f"  Agentes ejecutados: {summary['agents_executed']}")
                
                # Mostrar predicciones si están disponibles
                if not args.no_predict:
                    predictions = pipeline_result.get("predictions", {})
                    if predictions and predictions.get("success"):
                        pred_count = len(predictions.get("predictions", {}))
                        print(f"   Predicciones generadas: {pred_count}")
                
            else:
                print(f" ERROR EN PIPELINE: {pipeline_result.get('error', 'Error desconocido')}")
                
        elif args.batch:
            # Modo batch con pipeline completo
            agents_list = args.agents.split(',') if args.agents else None
            await batch_mode_wrapper(args)
            
        else:
            # Por defecto: ejecutar pipeline completo
            print(" Ejecutando pipeline completo por defecto...")
            pipeline_result = await system.run_complete_pipeline()
            
            if pipeline_result["success"]:
                print(" Pipeline completo ejecutado exitosamente")
                summary = pipeline_result["execution_summary"]
                print(f" Resumen: {summary['pipeline_completion']} completado")
            else:
                print(f" Error en pipeline: {pipeline_result.get('error')}")

    except Exception as e:
        print(f" Error durante la ejecución: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    print(f"\n Sistema completado exitosamente!")
    return 0

if __name__ == "__main__":
    # Ejecutar main con asyncio para soportar pipeline async
    import sys
    sys.exit(asyncio.run(main()))
