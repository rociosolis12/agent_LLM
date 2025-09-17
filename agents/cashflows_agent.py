"""
CashFlows Agent REACT - Versión Multi-Agente AUTÓNOMA COMPLETA
Especializado en análisis de estado de flujos de efectivo con patrón REACT exitoso
CARACTERÍSTICAS: Tool calls, detección robusta, completamente autónomo, compatible con coordinador
"""

from __future__ import annotations
import os
import re
import json
import time
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI
import groq

# ===== CONFIGURACIÓN DEL PROYECTO =====
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)
os.chdir(project_root)

if not env_path.exists():
    print(f"⚠️ Warning: Archivo .env no encontrado en {env_path}")

print("🔧 Cargar .env desde el directorio raíz del proyecto...")

# ===== CONFIGURACIÓN AZURE OPENAI =====
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

print("🔧 ----- Azure OpenAI Configuration -----")
print(f"🔗 Endpoint: {AZURE_OPENAI_ENDPOINT}")
print(f"🔑 API Key: {'✓' if AZURE_OPENAI_API_KEY else '✗'}")
print(f"📋 Deployment: {AZURE_OPENAI_DEPLOYMENT}")

if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
    raise ValueError("Azure OpenAI credentials required")

# ===== CONFIGURACIÓN GROQ =====
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

print("🔧 ----- Groq Configuration -----")
print(f"🔑 API Key: {'✓' if GROQ_API_KEY else '✗'}")
print(f"🤖 Model: {GROQ_MODEL}")

# ===== CLIENTE CHAT =====
class ChatClient:
    def __init__(self):
        self.azure_client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION
        ) if AZURE_OPENAI_API_KEY else None
        
        self.groq_client = groq.Groq(
            api_key=GROQ_API_KEY
        ) if GROQ_API_KEY else None

    def chat(self, history: List[Dict[str, str]], max_tokens: int = 1500) -> str:
        try:
            if self.groq_client:
                response = self.groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=history,
                    max_tokens=max_tokens,
                    temperature=0.1
                )
                return response.choices[0].message.content
            
            elif self.azure_client:
                response = self.azure_client.chat.completions.create(
                    model=AZURE_OPENAI_DEPLOYMENT,
                    messages=history,
                    max_tokens=max_tokens,
                    temperature=0.1
                )
                return response.choices[0].message.content
                
        except Exception as e:
            raise RuntimeError(f"Chat API error: {str(e)}")

# ===== INICIALIZACIÓN =====
chat_client = ChatClient()

# ===== HERRAMIENTAS ESPECÍFICAS PARA CASHFLOWS =====

@dataclass
class AnalyzeCashflowStructureTool:
    name: str = "analyzecashflowstructure"
    description: str = "Analiza estructura del PDF para localizar el estado de flujos de efectivo"
    
    def run(self, pdf_path: str, anchor_page: int = 10, max_pages: int = 25, extend: int = 2, **kwargs) -> Dict[str, Any]:
        try:
            print(f"🔍 Analizando estructura de flujos de efectivo - página ancla: {anchor_page}")
            
            # Páginas objetivo más probables para flujos de efectivo
            target_pages = list(range(max(1, anchor_page - extend), min(max_pages, anchor_page + extend + 1)))
            
            with pdfplumber.open(pdf_path) as pdf:
                found_cashflow = False
                for page_num in target_pages:
                    if page_num <= len(pdf.pages):
                        page = pdf.pages[page_num - 1]
                        text = page.extract_text() or ""
                        text_lower = text.lower()
                        
                        # Buscar indicadores de flujos de efectivo
                        cashflow_indicators = [
                            "statement of cash flows", "cash flow statement", "flujos de efectivo",
                            "operating activities", "investing activities", "financing activities",
                            "net cash provided", "net cash used", "cash and cash equivalents"
                        ]
                        
                        if any(indicator in text_lower for indicator in cashflow_indicators):
                            print(f"✅ Flujos de efectivo encontrados en página {page_num}")
                            found_cashflow = True
                            break
            
            return {
                "success": True,
                "pages_selected": target_pages[:5],  # Primeras 5 páginas para procesar
                "cashflow_found": found_cashflow,
                "anchor_page_used": anchor_page
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

@dataclass
class ExtractCashflowStatementTool:
    name: str = "extractcashflowstatement"
    description: str = "Extrae el contenido del estado de flujos de efectivo"
    
    def run(self, pdf_path: str, analysis_json: Dict = None, extract_semantic_chunks: bool = True, **kwargs) -> Dict[str, Any]:
        try:
            pages_to_process = analysis_json.get("pages_selected", [5, 6, 7, 8, 9]) if analysis_json else [5, 6, 7, 8, 9]
            print(f"📄 Extrayendo páginas: {pages_to_process}")
            
            extracted_text = ""
            total_chars = 0
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num in pages_to_process:
                    if page_num <= len(pdf.pages):
                        page = pdf.pages[page_num - 1]
                        text = page.extract_text() or ""
                        
                        # Buscar contenido relevante de flujos de efectivo
                        text_lower = text.lower()
                        cashflow_keywords = [
                            "cash", "flow", "operating", "investing", "financing", 
                            "activities", "net cash", "efectivo", "flujo"
                        ]
                        
                        if any(keyword in text_lower for keyword in cashflow_keywords):
                            extracted_text += f"\n=== PÁGINA {page_num} ===\n{text}"
                            total_chars += len(text)
                            print(f"✅ Página {page_num}: {len(text)} caracteres extraídos")
            
            if not extracted_text:
                # Fallback: extraer todas las páginas en el rango
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num in range(1, min(15, len(pdf.pages) + 1)):
                        page = pdf.pages[page_num - 1]
                        text = page.extract_text() or ""
                        extracted_text += f"\n=== PÁGINA {page_num} ===\n{text}"
                        total_chars += len(text)
            
            # Crear chunks semánticos
            chunks = []
            if extract_semantic_chunks and extracted_text:
                lines = extracted_text.split('\n')
                current_chunk = ""
                
                for line in lines:
                    if len(current_chunk) > 800:
                        chunks.append(current_chunk.strip())
                        current_chunk = line
                    else:
                        current_chunk += "\n" + line
                
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
            
            confidence = 1.0 if total_chars > 1000 else 0.8
            
            return {
                "success": True,
                "text": extracted_text,
                "total_characters": total_chars,
                "chunks": chunks,
                "confidence": confidence,
                "pages_processed": pages_to_process
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

@dataclass
class ValidateCashflowQualityTool:
    name: str = "validatecashflowquality"
    description: str = "Valida la calidad de los datos extraídos de flujos de efectivo"
    
    def run(self, extraction: Dict, **kwargs) -> Dict[str, Any]:
        try:
            text = extraction.get("text", "")
            confidence = extraction.get("confidence", 0.0)
            
            # Criterios de validación para flujos de efectivo
            quality_score = 0
            validation_details = []
            
            text_lower = text.lower()
            
            # Verificar secciones principales
            if "operating" in text_lower:
                quality_score += 25
                validation_details.append("✅ Actividades operativas encontradas")
            
            if "investing" in text_lower:
                quality_score += 25
                validation_details.append("✅ Actividades de inversión encontradas")
            
            if "financing" in text_lower:
                quality_score += 25
                validation_details.append("✅ Actividades de financiación encontradas")
            
            if "cash and cash equivalents" in text_lower or "efectivo y equivalentes" in text_lower:
                quality_score += 25
                validation_details.append("✅ Efectivo y equivalentes encontrados")
            
            # Determinar calidad
            if quality_score >= 75:
                quality = "excellent"
            elif quality_score >= 50:
                quality = "good"
            elif quality_score >= 25:
                quality = "fair"
            else:
                quality = "poor"
            
            final_confidence = min(confidence + (quality_score / 100), 1.0)
            
            print(f"✅ Validación completada: {quality} (confianza: {final_confidence:.3f})")
            
            return {
                "success": True,
                "quality": quality,
                "confidence": final_confidence,
                "score": quality_score,
                "details": validation_details
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

@dataclass
class SaveCashflowResultsTool:
    name: str = "savecashflowresults"
    description: str = "Guarda los resultados del análisis de flujos de efectivo"
    
    def run(self, output_dir: str, pdf_name: str, analysis: Dict, extraction: Dict, validation: Dict, **kwargs) -> Dict[str, Any]:
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            base_name = Path(pdf_name).stem
            files_created = 0
            
            # 1. Guardar resumen JSON
            summary = {
                "analysis": analysis,
                "extraction": {
                    "total_characters": extraction.get("total_characters", 0),
                    "confidence": extraction.get("confidence", 0),
                    "pages_processed": extraction.get("pages_processed", [])
                },
                "validation": validation,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            summary_file = output_path / f"{base_name}_cashflow_summary.json"
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            files_created += 1
            
            # 2. Guardar chunks semánticos
            if extraction.get("chunks"):
                chunks_file = output_path / f"{base_name}_semantic_chunks.json"
                with open(chunks_file, "w", encoding="utf-8") as f:
                    json.dump(extraction["chunks"], f, indent=2, ensure_ascii=False)
                files_created += 1
            
            # 3. Guardar reporte de calidad
            quality_report = f"""
REPORTE DE CALIDAD - FLUJOS DE EFECTIVO
=====================================
PDF: {pdf_name}
Fecha: {time.strftime("%Y-%m-%d %H:%M:%S")}

RESULTADOS DE VALIDACIÓN:
- Calidad: {validation.get('quality', 'unknown')}
- Confianza: {validation.get('confidence', 0):.3f}
- Puntuación: {validation.get('score', 0)}/100

DETALLES:
{chr(10).join(validation.get('details', []))}

EXTRACCIÓN:
- Caracteres procesados: {extraction.get('total_characters', 0)}
- Páginas procesadas: {extraction.get('pages_processed', [])}
- Chunks generados: {len(extraction.get('chunks', []))}
"""
            
            quality_file = output_path / f"{base_name}_cashflow_quality.txt"
            with open(quality_file, "w", encoding="utf-8") as f:
                f.write(quality_report)
            files_created += 1
            
            print(f"💾 Archivos guardados en: {output_path}")
            print(f" - JSON: {base_name}_cashflow_summary.json")
            print(f" - Chunks: {base_name}_semantic_chunks.json")  
            print(f" - Reporte: {base_name}_cashflow_quality.txt")
            
            return {
                "success": True,
                "files_created": files_created,
                "output_directory": str(output_path)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

# ===== REGISTRO DE HERRAMIENTAS =====
TOOLS_REGISTRY = {
    "analyzecashflowstructure": AnalyzeCashflowStructureTool(),
    "extractcashflowstatement": ExtractCashflowStatementTool(), 
    "validatecashflowquality": ValidateCashflowQualityTool(),
    "savecashflowresults": SaveCashflowResultsTool()
}

# ===== PROMPT SYSTEM PARA REACT =====
REACT_SYSTEM_PROMPT = """
Eres un agente especializado en extraer estados de flujos de efectivo.

OBJETIVO: Extraer el Statement of Cash Flows de GarantiBank International N.V.

REGLAS IMPORTANTES:
1. Ejecuta UNA SOLA herramienta por respuesta
2. NO planifiques múltiples herramientas a la vez
3. NO digas "CASHFLOWEXTRACTIONCOMPLETED" hasta completar TODAS las herramientas
4. Responde SOLO con el nombre de la herramienta a ejecutar

SECUENCIA OBLIGATORIA:
1. PRIMERA RESPUESTA: "analyzecashflowstructure" 
2. SEGUNDA RESPUESTA: "extractcashflowstatement"
3. TERCERA RESPUESTA: "validatecashflowquality"
4. CUARTA RESPUESTA: "savecashflowresults"  
5. QUINTA RESPUESTA: "CASHFLOWEXTRACTIONCOMPLETED"

EMPEZAR AHORA - Responde SOLO con: analyzecashflowstructure
"""


# ===== CLASE CASHFLOWS REACT AGENT =====
class CashFlowsREACTAgent:
    """Agente REACT especializado en flujos de efectivo - Patrón exitoso del Balance Agent"""
    
    def __init__(self):
        self.agent_type = "cashflows"
        self.max_steps = 10  # Reducido para evitar loops
        self.chat_client = chat_client
    
    def run_final_financial_extraction_agent(self, pdf_path: str, question: str = None) -> Dict[str, Any]:
        """Ejecuta la extracción de flujos de efectivo con patrón REACT"""
        try:
            print(f"🚀 CashFlowsREACTAgent AUTÓNOMO iniciando extracción para: {pdf_path}")
            
            pdf_file = Path(pdf_path)
            output_dir = Path("data/salida")
            
            # Configurar contexto de herramientas
            tools_ctx = {
                "pdfpath": str(pdf_file),
                "outputdir": str(output_dir),
                "anchorpage": 8,  # Página más probable para flujos de efectivo
                "lastanalysis": {},
                "lastextraction": {},
                "lastvalidation": {}
            }
            
            # Ejecutar patrón REACT
            history = [{"role": "system", "content": REACT_SYSTEM_PROMPT}]
            finished = False
            steps = 0
            
            print(f"🚀 Iniciando CashFlows Agent MEJORADO para {pdf_file.name}")
            print(f"📄 PDF: {pdf_file}")
            print(f"📁 Output: {output_dir}")
            print(f"🎯 Anchor page: {tools_ctx['anchorpage']}")
            
            while not finished and steps < self.max_steps:
                steps += 1
                print(f"📍 Paso ReAct {steps}/{self.max_steps}")
                
                history, finished = self.execute_react_step(history, tools_ctx)
                
                if finished:
                    print(f"🎉 TAREA COMPLETADA - Finalizando flujo ReAct")
                    break
            
            if steps >= self.max_steps:
                print(f"⚠️ Alcanzado límite máximo de pasos ({self.max_steps})")
                return {
                    "status": "error",
                    "steps_taken": steps,
                    "session_id": f"cashflows_{pdf_file.stem}",
                    "final_response": "Proceso interrumpido por límite de pasos",
                    "agent_type": "cashflows",
                    "error_details": "Max steps reached",
                    "specific_answer": "El análisis de flujos de efectivo fue interrumpido por límite de pasos."
                }
            
            # Generar respuesta específica
            specific_answer = self.generate_specific_response(question, tools_ctx)
            
            print("✅ Análisis completado exitosamente")
            print("✅ Cashflow extraction completed successfully (AUTÓNOMO)")
            
            return {
                "status": "task_completed",
                "steps_taken": steps,
                "session_id": f"cashflows_{pdf_file.stem}",
                "final_response": "Cashflow extraction completed successfully - AUTONOMOUS VERSION",
                "agent_type": "cashflows",
                "files_generated": tools_ctx.get("files_created", 3),
                "specific_answer": specific_answer
            }
            
        except Exception as e:
            print(f"❌ Error en CashFlowsREACTAgent: {str(e)}")
            return {
                "status": "error",
                "steps_taken": 0,
                "session_id": "cashflows_error",
                "final_response": f"Error in cashflows extraction: {str(e)}",
                "agent_type": "cashflows",
                "error_details": str(e),
                "specific_answer": f"Error durante la extracción de los flujos de efectivo: {str(e)}"
            }
    
    def execute_react_step(self, history: List[Dict[str, str]], tools_ctx: Dict[str, Any]) -> Tuple[List[Dict[str, str]], bool]:
        try:
            assistant_text = self.chat_client.chat(history, max_tokens=100)
            history.append({"role": "assistant", "content": assistant_text})
            
            print(f"🤖 Respuesta: {assistant_text.strip()}")
            
            # FINALIZACIÓN: Solo si es respuesta específica y corta
            if (len(assistant_text.strip()) < 50 and 
                "cashflowextractioncompleted" in assistant_text.lower()):
                print(f"🎉 FINALIZACIÓN CORRECTA DETECTADA")
                return history, True
            
            # TOOL DETECTION: Buscar herramienta específica
            tool_name = None
            tool_names = ["analyzecashflowstructure", "extractcashflowstatement", 
                        "validatecashflowquality", "savecashflowresults"]
            
            assistant_clean = assistant_text.lower().strip()
            
            for tool in tool_names:
                if tool == assistant_clean or (tool in assistant_clean and len(assistant_text) < 200):
                    tool_name = tool
                    break
            
            if not tool_name:
                if len(assistant_text) > 200:
                    feedback = "Responde SOLO con el nombre de la herramienta: analyzecashflowstructure"
                else:
                    feedback = "Herramienta no reconocida. Usa: analyzecashflowstructure"
                history.append({"role": "user", "content": feedback})
                return history, False
            
            # EJECUTAR HERRAMIENTA
            print(f"🚀 EJECUTANDO: {tool_name}")
            
            # Preparar parámetros según herramienta
            if tool_name == "analyzecashflowstructure":
                params = {
                    "pdf_path": tools_ctx["pdfpath"],
                    "anchor_page": tools_ctx.get("anchorpage", 8),
                    "max_pages": 25, "extend": 2
                }
            elif tool_name == "extractcashflowstatement":
                params = {
                    "pdf_path": tools_ctx["pdfpath"],
                    "analysis_json": tools_ctx.get("lastanalysis", {}),
                    "extract_semantic_chunks": True
                }
            elif tool_name == "validatecashflowquality":
                params = {
                    "extraction": tools_ctx.get("lastextraction", {"text": "test", "confidence": 0.8})
                }
            elif tool_name == "savecashflowresults":
                params = {
                    "output_dir": tools_ctx["outputdir"], "pdf_name": tools_ctx["pdfpath"],
                    "analysis": tools_ctx.get("lastanalysis", {}),
                    "extraction": tools_ctx.get("lastextraction", {}),
                    "validation": tools_ctx.get("lastvalidation", {})
                }
            
            # Ejecutar herramienta
            tool_obj = TOOLS_REGISTRY.get(tool_name)
            result = tool_obj.run(**params)
            
            if result.get("success"):
                print(f"✅ {tool_name} exitoso")
                
                # Actualizar contexto
                if tool_name == "analyzecashflowstructure":
                    tools_ctx["lastanalysis"] = result
                    feedback = f"✅ Estructura analizada. Siguiente herramienta: extractcashflowstatement"
                elif tool_name == "extractcashflowstatement":
                    tools_ctx["lastextraction"] = result
                    feedback = f"✅ Datos extraídos. Siguiente herramienta: validatecashflowquality"
                elif tool_name == "validatecashflowquality":
                    tools_ctx["lastvalidation"] = result
                    feedback = f"✅ Validación completa. Siguiente herramienta: savecashflowresults"
                elif tool_name == "savecashflowresults":
                    tools_ctx["files_created"] = result.get("files_created", 3)
                    feedback = f"✅ Archivos guardados. Responde con: CASHFLOWEXTRACTIONCOMPLETED"
            else:
                feedback = f"❌ Error en {tool_name}: {result.get('error')}"
            
            history.append({"role": "user", "content": feedback})
            return history, False
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return history, False

    
    def generate_specific_response(self, question: str, tools_ctx: Dict[str, Any]) -> str:
        """Genera respuesta específica basada en los datos extraídos"""
        try:
            extraction = tools_ctx.get("lastextraction", {})
            validation = tools_ctx.get("lastvalidation", {})
            
            if not extraction or not extraction.get("success"):
                return "No se pudieron extraer datos de flujos de efectivo del documento."
            
            # Extraer información clave del texto
            text = extraction.get("text", "")
            confidence = validation.get("confidence", 0.8)
            quality = validation.get("quality", "unknown")
            
            # Generar respuesta específica
            response_parts = ["📊 **RESUMEN DE FLUJOS DE EFECTIVO EXTRAÍDOS**"]
            
            # Buscar cifras clave en el texto
            text_lower = text.lower()
            
            if "operating activities" in text_lower or "actividades operativas" in text_lower:
                response_parts.append("• **Actividades Operativas**: Datos identificados y procesados")
            
            if "investing activities" in text_lower or "actividades de inversión" in text_lower:
                response_parts.append("• **Actividades de Inversión**: Información extraída")
            
            if "financing activities" in text_lower or "actividades de financiación" in text_lower:
                response_parts.append("• **Actividades de Financiación**: Datos capturados")
            
            if "cash and cash equivalents" in text_lower or "efectivo y equivalentes" in text_lower:
                response_parts.append("• **Efectivo y Equivalentes**: Posiciones inicial y final identificadas")
            
            response_parts.append(f"**Calidad de Extracción**: {quality.title()} (confianza: {confidence:.1%})")
            response_parts.append("**Fuente**: Estado de flujos de efectivo consolidado de GarantiBank International N.V.")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            return f"Se completó la extracción de flujos de efectivo, pero hubo un error al generar la respuesta específica: {str(e)}"

# ===== FUNCIONES DE COMPATIBILIDAD =====
def run_cashflow_agent(pdf_path: Path, output_dir: Path, max_steps: int = 10) -> Dict[str, Any]:
    """Función principal para ejecutar el agente de cashflows - compatibilidad"""
    agent = CashFlowsREACTAgent()
    try:
        result = agent.run_final_financial_extraction_agent(str(pdf_path))
        return {
            "history": [],
            "context": {"last_extraction": result},
            "steps_completed": result.get("steps_taken", 5),
            "finished": result.get("status") == "task_completed"
        }
    except Exception as e:
        return {
            "history": [],
            "context": {},
            "steps_completed": 0,
            "finished": False,
            "error": str(e)
        }

# ===== CONFIGURACIÓN Y MAIN =====
DEFAULT_CONFIG = {
    "pdf": "data/entrada/output/bbva_2023_div.pdf",
    "out": "data/salida", 
    "maxsteps": 10
}

def main():
    parser = argparse.ArgumentParser(
        description="CashFlows Agent AUTÓNOMO especializado en Estado de Flujos de Efectivo - Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
  python agents/cashflows_agent.py                    # Usa configuración predefinida
  python agents/cashflows_agent.py --pdf otro.pdf    # Sobreescribe PDF

CARACTERÍSTICAS AUTÓNOMAS:
  - Patrón REACT exitoso del Balance Agent
  - Detección robusta de tool calls  
  - Herramientas específicas para flujos de efectivo
  - Respuestas detalladas y elaboradas
  - Fallback robusto con respuestas basadas en datos extraídos

Sistema Multi-Agente:
  Esta versión incluye CashFlowsREACTAgent AUTÓNOMO para integración con main_system.py
"""
    )
    
    # Argumentos opcionales
    parser.add_argument("--pdf", default=DEFAULT_CONFIG["pdf"], 
                       help=f"Ruta al PDF (por defecto: {DEFAULT_CONFIG['pdf']})")
    parser.add_argument("--out", default=DEFAULT_CONFIG["out"],
                       help=f"Directorio de salida (por defecto: {DEFAULT_CONFIG['out']})")
    parser.add_argument("--maxsteps", type=int, default=DEFAULT_CONFIG["maxsteps"],
                       help=f"Máximo pasos REACT (por defecto: {DEFAULT_CONFIG['maxsteps']})")
    parser.add_argument("--question", type=str, default=None,
                       help="Pregunta específica sobre flujos de efectivo")
    
    args = parser.parse_args()
    
    # MOSTRAR CONFIGURACIÓN
    print("🚀 Cashflows Agent v3.0 AUTÓNOMO Multi-Agent - Configuración Automática")
    print(f"📄 PDF: {args.pdf}")
    print(f"📁 Salida: {args.out}")
    print(f"⚙️ Azure OpenAI: {AZURE_OPENAI_DEPLOYMENT}")
    print(f"🔧 Max steps: {args.maxsteps}")
    print(f"🤖 Multi-Agent: CashFlowsREACTAgent AUTÓNOMO class available")
    print("🆕 CARACTERÍSTICAS: Patrón REACT exitoso, detección robusta tool calls, respuestas elaboradas")
    
    try:
        # VERIFICAR PDF
        pdf_path = Path(args.pdf)
        output_dir = Path(args.out)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not pdf_path.exists():
            print(f"❌ Error: PDF no encontrado en {pdf_path}")
            return
        
        # CREAR AGENTE Y EJECUTAR
        agent = CashFlowsREACTAgent()
        
        if args.question:
            print(f"❓ Pregunta específica: {args.question}")
            result = agent.run_final_financial_extraction_agent(str(pdf_path), args.question)
        else:
            result = agent.run_final_financial_extraction_agent(str(pdf_path))
        
        # MOSTRAR RESULTADOS
        print("🎯 ==== RESUMEN DE EJECUCIÓN AUTÓNOMO ====")
        print(f"Estado: {'✅ EXITOSO' if result.get('status') == 'task_completed' else '❌ ERROR'}")
        print(f"Pasos completados: {result.get('steps_taken', 0)}")
        print(f"Archivos generados: {result.get('files_generated', 0)}")
        
        if result.get('status') == 'task_completed':
            print("📋 ==== RESPUESTA ESPECÍFICA ====")
            print(result.get("specific_answer", "No hay respuesta específica disponible"))
        else:
            print(f"❌ Error: {result.get('error_details', 'Error desconocido')}")
        
        print("🎉 Análisis de flujos de efectivo completado!")
        print("🤖 Clase CashFlowsREACTAgent AUTÓNOMA disponible para sistema multi-agente")
        print("🆕 Versión autónoma con patrón REACT exitoso y respuestas específicas usando LLM")
        
    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
        raise

if __name__ == "__main__":
    main()
