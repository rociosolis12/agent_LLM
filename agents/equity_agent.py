"""
Equity Agent REACT - Versión Multi-Agente AUTÓNOMA COMPLETA
Especializado en análisis de estado de cambios en patrimonio con patrón REACT exitoso
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
    print(f" Warning: Archivo .env no encontrado en {env_path}")

print(" Cargar .env desde el directorio raíz del proyecto...")

# ===== CONFIGURACIÓN AZURE OPENAI =====
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_EMBEDDING_MODEL = os.getenv("AZURE_EMBEDDING_MODEL", "text-embedding-3-small")


print(" ----- Azure OpenAI Configuration -----")
print(f" Endpoint: {AZURE_OPENAI_ENDPOINT}")
print(f" API Key: {'✓' if AZURE_OPENAI_API_KEY else '✗'}")
print(f" Deployment: {AZURE_OPENAI_DEPLOYMENT}")

if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
    raise ValueError("Azure OpenAI credentials required")

# ===== CONFIGURACIÓN GROQ =====
#GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


# ===== CLIENTE CHAT =====
class ChatClient:
    """Cliente de chat usando Azure OpenAI """
    
    def __init__(self):
        # Usar Azure OpenAI en lugar de Groq
        self.client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION
        )
        self.model = AZURE_OPENAI_DEPLOYMENT  # gpt-4o
        
    def chat(self, messages: List[Dict], **kwargs) -> str:
        """
        Genera respuesta de chat usando Azure OpenAI
        
        Args:
            messages: Lista de mensajes [{"role": "system/user/assistant", "content": "..."}]
            **kwargs: Parámetros adicionales (temperature, max_tokens, etc.)
            
        Returns:
            Contenido de la respuesta del asistente
        """
        try:
            # Extraer parámetros con valores por defecto
            temperature = kwargs.get('temperature', 0.0)
            max_tokens = kwargs.get('max_tokens', 2000)
            
            # Llamar a Azure OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extraer contenido de la respuesta
            return response.choices[0].message.content
            
        except Exception as e:
            raise RuntimeError(f"Chat API error: {e}")


# ===== CLIENTE EMBEDDINGS AZURE =====
class AzureEmbeddingClient:
    """Cliente para generar embeddings usando Azure OpenAI"""
    
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION
        )
        self.model = AZURE_EMBEDDING_MODEL
        
    def get_text_embedding(self, text: str, max_length: int = 8000) -> Optional[np.ndarray]:
        """
        Genera embeddings usando Azure OpenAI
        
        Args:
            text: Texto para generar embedding
            max_length: Longitud máxima del texto
            
        Returns:
            Vector de embedding normalizado o None si hay error
        """
        try:
            # Truncar texto si excede el límite
            text = text[:max_length] if len(text) > max_length else text
            
            # Generar embedding con Azure OpenAI
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            
            # Extraer el vector de embedding
            embedding = np.array(response.data[0].embedding)
            
            # Normalizar el embedding para cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            print(f"❌ Error generating Azure embedding: {e}")
            return None
    
    def find_similar_sections(
        self, 
        query_text: str, 
        text_chunks: List[str], 
        top_k: int = 5
    ) -> List[tuple]:
        """
        Encuentra secciones más relevantes usando embeddings
        
        Args:
            query_text: Texto de consulta
            text_chunks: Lista de chunks de texto
            top_k: Número de resultados a devolver
            
        Returns:
            Lista de tuplas (índice, similaridad, texto)
        """
        query_embedding = self.get_text_embedding(query_text)
        if query_embedding is None:
            return []
        
        similarities = []
        for i, chunk in enumerate(text_chunks):
            chunk_embedding = self.get_text_embedding(chunk)
            if chunk_embedding is not None:
                similarity = cosine_similarity(
                    [query_embedding], 
                    [chunk_embedding]
                )[0][0]
                similarities.append((i, float(similarity), chunk))
        
        # Ordenar por similaridad descendente
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

# ===== INICIALIZACIÓN =====
chat_client = ChatClient()
embedding_client = AzureEmbeddingClient() 

# ===== HERRAMIENTAS ESPECÍFICAS PARA EQUITY =====

@dataclass
class AnalyzeEquityStructureTool:
    name: str = "analyzeequitystructure"
    description: str = "Analiza estructura del PDF para localizar el estado de cambios en patrimonio"
    
    def run(self, pdf_path: str, anchor_page: int = 8, max_pages: int = 25, extend: int = 2, **kwargs) -> Dict[str, Any]:
        try:
            print(f" Analizando estructura de cambios en patrimonio - página ancla: {anchor_page}")
            
            # Páginas objetivo más probables para cambios en patrimonio
            target_pages = list(range(max(1, anchor_page - extend), min(max_pages, anchor_page + extend + 1)))
            
            with pdfplumber.open(pdf_path) as pdf:
                found_equity = False
                for page_num in target_pages:
                    if page_num <= len(pdf.pages):
                        page = pdf.pages[page_num - 1]
                        text = page.extract_text() or ""
                        text_lower = text.lower()
                        
                        # Buscar indicadores de cambios en patrimonio
                        equity_indicators = [
                            "statement of changes in equity", "changes in equity", "equity statement",
                            "consolidated statement of changes in equity", "shareholders' equity changes",
                            "share capital", "retained earnings", "reserves", "patrimonio", "capital social"
                        ]
                        
                        if any(indicator in text_lower for indicator in equity_indicators):
                            print(f" Cambios en patrimonio encontrados en página {page_num}")
                            found_equity = True
                            break
            
            return {
                "success": True,
                "pages_selected": target_pages[:5],  # Primeras 5 páginas para procesar
                "equity_found": found_equity,
                "anchor_page_used": anchor_page
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

@dataclass
@dataclass
class ExtractEquityStatementTool:
    name: str = "extractequitystatement"
    description: str = "Extrae el contenido del estado de cambios en patrimonio con búsqueda semántica"
    
    def run(self, pdf_path: str, analysis_json: Dict = None, extract_semantic_chunks: bool = True, **kwargs) -> Dict[str, Any]:
        try:
            pages_to_process = analysis_json.get("pages_selected", [6, 7, 8, 9, 10]) if analysis_json else [6, 7, 8, 9, 10]
            print(f"📄 Extrayendo páginas: {pages_to_process}")
            
            extracted_text = ""
            total_chars = 0
            all_text_chunks = []  # NUEVO: Para búsqueda semántica
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num in pages_to_process:
                    if page_num <= len(pdf.pages):
                        page = pdf.pages[page_num - 1]
                        text = page.extract_text() or ""
                        
                        # Buscar contenido relevante de cambios en patrimonio
                        text_lower = text.lower()
                        equity_keywords = [
                            "share capital", "retained earnings", "reserves", "equity",
                            "capital social", "reservas", "patrimonio", "dividends",
                            "comprehensive income", "other equity", "total equity"
                        ]
                        
                        if any(keyword in text_lower for keyword in equity_keywords):
                            extracted_text += f"\n=== PÁGINA {page_num} ===\n{text}"
                            total_chars += len(text)
                            
                            # NUEVO: Dividir en chunks para embeddings
                            page_chunks = self._split_text_into_chunks(text, chunk_size=1000)
                            all_text_chunks.extend([
                                {'page': page_num, 'chunk': chunk}
                                for chunk in page_chunks
                            ])
                            
                            print(f"✅ Página {page_num}: {len(text)} caracteres extraídos")
            
            if not extracted_text:
                # Fallback: extraer todas las páginas en el rango
                print("⚠️ Aplicando fallback: extrayendo páginas 1-15")
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num in range(1, min(15, len(pdf.pages) + 1)):
                        page = pdf.pages[page_num - 1]
                        text = page.extract_text() or ""
                        extracted_text += f"\n=== PÁGINA {page_num} ===\n{text}"
                        total_chars += len(text)
                        
                        page_chunks = self._split_text_into_chunks(text, chunk_size=1000)
                        all_text_chunks.extend([
                            {'page': page_num, 'chunk': chunk}
                            for chunk in page_chunks
                        ])
            
            # ===== NUEVA SECCIÓN: BÚSQUEDA SEMÁNTICA CON EMBEDDINGS =====
            semantic_results = {}
            if all_text_chunks and extract_semantic_chunks:
                print(f"🔍 Aplicando búsqueda semántica con embeddings ({len(all_text_chunks)} chunks)...")
                
                chunk_texts = [item['chunk'] for item in all_text_chunks]
                
                # Queries semánticas específicas para cambios en patrimonio
                semantic_queries = {
                    'share_capital': "capital social acciones ordinarias y movimientos de capital",
                    'retained_earnings': "resultados acumulados beneficios retenidos y reservas",
                    'dividends': "dividendos pagados distribución de beneficios a accionistas",
                    'reserves': "reservas legales reservas estatutarias y otras reservas",
                    'comprehensive_income': "resultado global resultado integral otro resultado integral",
                    'total_equity': "patrimonio total patrimonio neto consolidado equity total"
                }
                
                for category, query in semantic_queries.items():
                    try:
                        similar_sections = embedding_client.find_similar_sections(
                            query, 
                            chunk_texts, 
                            top_k=3
                        )
                        
                        relevant_chunks = []
                        for idx, score, chunk_text in similar_sections:
                            if score > 0.65:  # Umbral de similaridad
                                chunk_info = all_text_chunks[idx]
                                relevant_chunks.append({
                                    'score': score,
                                    'text': chunk_text,
                                    'page': chunk_info['page']
                                })
                                print(f"  ✅ {category} (score {score:.2f}, page {chunk_info['page']}): {chunk_text[:60]}...")
                        
                        if relevant_chunks:
                            semantic_results[category] = relevant_chunks
                    
                    except Exception as e:
                        print(f"  ⚠️ Error en búsqueda semántica para {category}: {e}")
                
                # Enriquecer texto con resultados semánticos
                if semantic_results:
                    print(f"📋 Categorías encontradas con embeddings: {len(semantic_results)}")
                    enriched_text = "\n\n=== SECCIONES RELEVANTES (BÚSQUEDA SEMÁNTICA) ===\n"
                    
                    for category, chunks in semantic_results.items():
                        enriched_text += f"\n--- {category.upper()} ---\n"
                        for chunk_data in chunks[:2]:  # Top 2 por categoría
                            enriched_text += f"[Página {chunk_data['page']}, Score: {chunk_data['score']:.2f}]\n"
                            enriched_text += chunk_data['text'][:500] + "...\n\n"
                    
                    # Prefijar al texto original
                    extracted_text = enriched_text + "\n\n=== TEXTO COMPLETO EXTRAÍDO ===\n" + extracted_text[:5000]
            
            # Crear chunks básicos (compatibilidad)
            chunks = []
            if extracted_text:
                chunks = self._split_text_into_chunks(extracted_text, chunk_size=800)
            
            confidence = 1.0 if total_chars > 1000 else 0.8
            
            # NUEVO: Ajustar confianza con embeddings
            if semantic_results:
                confidence = min(1.0, confidence + 0.1)
            
            return {
                "success": True,
                "text": extracted_text,
                "total_characters": total_chars,
                "chunks": chunks,
                "confidence": confidence,
                "pages_processed": pages_to_process,
                "semantic_results": semantic_results,  # NUEVO
                "semantic_categories_found": len(semantic_results),  # NUEVO
                "total_chunks_analyzed": len(all_text_chunks)  # NUEVO
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Divide texto en chunks para procesamiento con embeddings"""
        if not text:
            return []
        
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1
            if current_length + word_length > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(word)
            current_length += word_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks


@dataclass
class ValidateEquityQualityTool:
    name: str = "validateequityquality"
    description: str = "Valida la calidad de los datos extraídos de cambios en patrimonio"
    
    def run(self, extraction: Dict, **kwargs) -> Dict[str, Any]:
        try:
            text = extraction.get("text", "")
            confidence = extraction.get("confidence", 0.0)
            
            # Criterios de validación para cambios en patrimonio
            quality_score = 0
            validation_details = []
            
            text_lower = text.lower()
            
            # Verificar componentes principales
            if "share capital" in text_lower or "capital social" in text_lower:
                quality_score += 25
                validation_details.append(" Capital social encontrado")
            
            if "retained earnings" in text_lower or "resultados acumulados" in text_lower:
                quality_score += 25
                validation_details.append(" Resultados acumulados encontrados")
            
            if "reserves" in text_lower or "reservas" in text_lower:
                quality_score += 25
                validation_details.append(" Reservas encontradas")
            
            if "total equity" in text_lower or "patrimonio total" in text_lower:
                quality_score += 25
                validation_details.append(" Patrimonio total encontrado")
            
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
            
            print(f" Validación completada: {quality} (confianza: {final_confidence:.3f})")
            
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
class SaveEquityResultsTool:
    name: str = "saveequityresults"
    description: str = "Guarda los resultados del análisis de cambios en patrimonio"
    
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
            
            summary_file = output_path / f"{base_name}_equity_summary.json"
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
REPORTE DE CALIDAD - CAMBIOS EN PATRIMONIO
==========================================
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
            
            quality_file = output_path / f"{base_name}_equity_quality.txt"
            with open(quality_file, "w", encoding="utf-8") as f:
                f.write(quality_report)
            files_created += 1
            
            print(f" Archivos guardados en: {output_path}")
            print(f" - JSON: {base_name}_equity_summary.json")
            print(f" - Chunks: {base_name}_semantic_chunks.json")  
            print(f" - Reporte: {base_name}_equity_quality.txt")
            
            return {
                "success": True,
                "files_created": files_created,
                "output_directory": str(output_path)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

# ===== REGISTRO DE HERRAMIENTAS =====
TOOLS_REGISTRY = {
    "analyzeequitystructure": AnalyzeEquityStructureTool(),
    "extractequitystatement": ExtractEquityStatementTool(), 
    "validateequityquality": ValidateEquityQualityTool(),
    "saveequityresults": SaveEquityResultsTool()
}

# ===== PROMPT SYSTEM PARA REACT =====
REACT_SYSTEM_PROMPT = """
Eres un agente especializado en extraer estados de cambios en patrimonio.

OBJETIVO: Extraer el Statement of Changes in Equity de GarantiBank International N.V.

REGLAS IMPORTANTES:
1. Ejecuta UNA SOLA herramienta por respuesta
2. NO planifiques múltiples herramientas a la vez
3. NO digas "EQUITYEXTRACTIONCOMPLETED" hasta completar TODAS las herramientas
4. Responde SOLO con el nombre de la herramienta a ejecutar

SECUENCIA OBLIGATORIA:
1. PRIMERA RESPUESTA: "analyzeequitystructure"
2. SEGUNDA RESPUESTA: "extractequitystatement"
3. TERCERA RESPUESTA: "validateequityquality"
4. CUARTA RESPUESTA: "saveequityresults"  
5. QUINTA RESPUESTA: "EQUITYEXTRACTIONCOMPLETED"

EMPEZAR AHORA - Responde SOLO con: analyzeequitystructure
"""

# ===== CLASE EQUITY REACT AGENT =====
class EquityREACTAgent:
    """Agente REACT especializado en cambios en patrimonio - Patrón exitoso del Balance Agent"""
    
    def __init__(self):
        self.agent_type = "equity"
        self.max_steps = 10  # Reducido para evitar loops
        self.chat_client = chat_client
    
    def run_final_financial_extraction_agent(self, pdf_path: str, question: str = None) -> Dict[str, Any]:
        """Ejecuta la extracción de cambios en patrimonio con patrón REACT"""
        try:
            print(f" EquityREACTAgent AUTÓNOMO iniciando extracción para: {pdf_path}")
            
            pdf_file = Path(pdf_path)
            output_dir = Path("data/salida")
            
            # Configurar contexto de herramientas
            tools_ctx = {
                "pdfpath": str(pdf_file),
                "outputdir": str(output_dir),
                "anchorpage": 8,  # Página más probable para cambios en patrimonio
                "lastanalysis": {},
                "lastextraction": {},
                "lastvalidation": {}
            }
            
            # Ejecutar patrón REACT
            history = [{"role": "system", "content": REACT_SYSTEM_PROMPT}]
            finished = False
            steps = 0
            
            print(f" Iniciando Equity Agent MEJORADO para {pdf_file.name}")
            print(f" PDF: {pdf_file}")
            print(f" Output: {output_dir}")
            print(f" Anchor page: {tools_ctx['anchorpage']}")
            
            while not finished and steps < self.max_steps:
                steps += 1
                print(f" Paso ReAct {steps}/{self.max_steps}")
                
                history, finished = self.execute_react_step(history, tools_ctx)
                
                if finished:
                    print(f" TAREA COMPLETADA - Finalizando flujo ReAct")
                    break
            
            if steps >= self.max_steps:
                print(f" Alcanzado límite máximo de pasos ({self.max_steps})")
                return {
                    "status": "error",
                    "steps_taken": steps,
                    "session_id": f"equity_{pdf_file.stem}",
                    "final_response": "Proceso interrumpido por límite de pasos",
                    "agent_type": "equity",
                    "error_details": "Max steps reached",
                    "specific_answer": "El análisis de cambios en patrimonio fue interrumpido por límite de pasos."
                }
            
            # Generar respuesta específica
            specific_answer = self.generate_specific_response(question, tools_ctx)
            
            print(" Análisis completado exitosamente")
            print(" Equity extraction completed successfully (AUTÓNOMO)")
            
            return {
                "status": "task_completed",
                "steps_taken": steps,
                "session_id": f"equity_{pdf_file.stem}",
                "final_response": "Equity extraction completed successfully - AUTONOMOUS VERSION",
                "agent_type": "equity",
                "files_generated": tools_ctx.get("files_created", 3),
                "specific_answer": specific_answer
            }
            
        except Exception as e:
            print(f" Error en EquityREACTAgent: {str(e)}")
            return {
                "status": "error",
                "steps_taken": 0,
                "session_id": "equity_error",
                "final_response": f"Error in equity extraction: {str(e)}",
                "agent_type": "equity",
                "error_details": str(e),
                "specific_answer": f"Error durante la extracción de los cambios en patrimonio: {str(e)}"
            }
    
    def execute_react_step(self, history: List[Dict[str, str]], tools_ctx: Dict[str, Any]) -> Tuple[List[Dict[str, str]], bool]:
        """Ejecuta un paso del patrón REACT - MISMA LÓGICA EXITOSA DEL BALANCE/CASHFLOWS AGENT"""
        try:
            assistant_text = self.chat_client.chat(history, max_tokens=100)
            history.append({"role": "assistant", "content": assistant_text})
            
            print(f" Respuesta: {assistant_text.strip()}")
            
            # FINALIZACIÓN: Solo si es respuesta específica y corta
            if (len(assistant_text.strip()) < 50 and 
                "equityextractioncompleted" in assistant_text.lower()):
                print(f" FINALIZACIÓN CORRECTA DETECTADA")
                return history, True
            
            # TOOL DETECTION: Buscar herramienta específica
            tool_name = None
            tool_names = ["analyzeequitystructure", "extractequitystatement", 
                         "validateequityquality", "saveequityresults"]
            
            assistant_clean = assistant_text.lower().strip()
            
            for tool in tool_names:
                if tool == assistant_clean or (tool in assistant_clean and len(assistant_text) < 200):
                    tool_name = tool
                    break
            
            if not tool_name:
                if len(assistant_text) > 200:
                    feedback = "Responde SOLO con el nombre de la herramienta: analyzeequitystructure"
                else:
                    feedback = "Herramienta no reconocida. Usa: analyzeequitystructure"
                history.append({"role": "user", "content": feedback})
                return history, False
            
            # EJECUTAR HERRAMIENTA
            print(f" EJECUTANDO: {tool_name}")
            
            # Preparar parámetros según herramienta
            if tool_name == "analyzeequitystructure":
                params = {
                    "pdf_path": tools_ctx["pdfpath"],
                    "anchor_page": tools_ctx.get("anchorpage", 8),
                    "max_pages": 25, "extend": 2
                }
            elif tool_name == "extractequitystatement":
                params = {
                    "pdf_path": tools_ctx["pdfpath"],
                    "analysis_json": tools_ctx.get("lastanalysis", {}),
                    "extract_semantic_chunks": True
                }
            elif tool_name == "validateequityquality":
                params = {
                    "extraction": tools_ctx.get("lastextraction", {"text": "test", "confidence": 0.8})
                }
            elif tool_name == "saveequityresults":
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
                print(f" {tool_name} exitoso")
                
                # Actualizar contexto
                if tool_name == "analyzeequitystructure":
                    tools_ctx["lastanalysis"] = result
                    feedback = f" Estructura analizada. Siguiente herramienta: extractequitystatement"
                elif tool_name == "extractequitystatement":
                    tools_ctx["lastextraction"] = result
                    feedback = f" Datos extraídos. Siguiente herramienta: validateequityquality"
                elif tool_name == "validateequityquality":
                    tools_ctx["lastvalidation"] = result
                    feedback = f" Validación completa. Siguiente herramienta: saveequityresults"
                elif tool_name == "saveequityresults":
                    tools_ctx["files_created"] = result.get("files_created", 3)
                    feedback = f" Archivos guardados. Responde con: EQUITYEXTRACTIONCOMPLETED"
            else:
                feedback = f" Error en {tool_name}: {result.get('error')}"
            
            history.append({"role": "user", "content": feedback})
            return history, False
            
        except Exception as e:
            print(f" Error: {str(e)}")
            return history, False
    
    def generate_specific_response(self, question: str, tools_ctx: Dict[str, Any]) -> str:
        """Genera respuesta específica usando LLM para análisis inteligente basado en datos reales"""
        try:
            extraction = tools_ctx.get("lastextraction", {})
            validation = tools_ctx.get("lastvalidation", {})
            
            if not extraction or not extraction.get("success"):
                return "No se pudieron extraer datos de cambios en patrimonio del documento."
            
            # Extraer el contenido real del PDF
            text = extraction.get("text", "")
            confidence = validation.get("confidence", 0.8)
            quality = validation.get("quality", "unknown")
            
            if not text or len(text.strip()) < 100:
                return "El contenido extraído de cambios en patrimonio es insuficiente para realizar un análisis detallado."
            
            # **ANÁLISIS INTELIGENTE CON LLM**
            analysis_prompt = f"""
Actúa como un analista financiero experto especializado en estados de cambios en patrimonio.

Analiza el siguiente contenido extraído del estado de cambios en patrimonio de GarantiBank International N.V. y proporciona un análisis detallado, profesional y específico:

CONTENIDO EXTRAÍDO:
{text[:4000]}  # Limitamos a 4000 caracteres para el análisis

INSTRUCCIONES:
1. Identifica y analiza las CIFRAS ESPECÍFICAS encontradas en el texto
2. Examina los componentes: Capital Social, Reservas, Resultados Acumulados, Patrimonio Total
3. Proporciona interpretación profesional de los movimientos patrimoniales
4. Identifica distribuciones de dividendos, aumentos de capital, y otros cambios
5. Incluye comparaciones año anterior si están disponibles
6. Genera conclusiones estratégicas específicas sobre la gestión patrimonial

FORMATO REQUERIDO:
- Análisis basado ÚNICAMENTE en los datos reales encontrados
- Incluye cifras específicas cuando las encuentres
- Interpretación profesional de cada componente patrimonial
- Conclusiones sobre la solidez patrimonial y política de dividendos
- Longitud: 400-600 palabras

No inventes cifras que no aparezcan en el texto. Si una sección no tiene datos específicos, indica que requiere análisis adicional.
"""

            try:
                # Usar el LLM para generar análisis inteligente
                analysis_response = self.chat_client.chat([
                    {"role": "system", "content": "Eres un analista financiero experto en análisis patrimonial bancario."},
                    {"role": "user", "content": analysis_prompt}
                ], max_tokens=1500)
                
                # Combinar el análisis del LLM con información técnica
                response_parts = [
                    " **ANÁLISIS DETALLADO DE CAMBIOS EN PATRIMONIO - GarantiBank International N.V.**",
                    "=" * 80,
                    "",
                    analysis_response,
                    "",
                    "### 📋 **INFORMACIÓN TÉCNICA DEL ANÁLISIS**",
                    f"• **Calidad de extracción**: {quality.title()} con {confidence:.1%} de confianza",
                    f"• **Caracteres analizados**: {len(text):,} caracteres del documento original",
                    f"• **Páginas procesadas**: {len(extraction.get('pages_processed', []))} páginas del estado financiero",
                    "• **Metodología**: Análisis automático con IA especializada en patrimonio bancario",
                    "• **Fuente**: Estado de cambios en patrimonio consolidado de GarantiBank International N.V.",
                    "",
                    "=" * 80,
                    " *Análisis generado por sistema de IA especializada en análisis patrimonial*"
                ]
                
                return "\n".join(response_parts)
                
            except Exception as llm_error:
                print(f"Error en análisis LLM: {str(llm_error)}")
                # Fallback: análisis básico si el LLM falla
                return self.generate_fallback_analysis(text, confidence, quality, extraction)
                
        except Exception as e:
            return f"Error al generar análisis específico de cambios en patrimonio: {str(e)}"

    def generate_fallback_analysis(self, text: str, confidence: float, quality: str, extraction: Dict) -> str:
        """Análisis de respaldo basado en extracción de datos específicos del texto"""
        
        response_parts = []
        response_parts.append(" **ANÁLISIS DE CAMBIOS EN PATRIMONIO - GarantiBank International N.V.**")
        response_parts.append("=" * 70)
        
        text_lower = text.lower()
        
        # Extraer cifras específicas del texto usando regex
        import re
        
        # Buscar cifras en miles de euros
        money_patterns = [
            r'(\d{1,3}(?:,\d{3})*)\s*(?:thousand|miles?\s+de\s+euros?|\b€\b)',
            r'€\s*(\d{1,3}(?:,\d{3})*)',
            r'(\d{1,3}(?:,\d{3})*)\s*(?:k|thousand)',
        ]
        
        found_amounts = []
        for pattern in money_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            found_amounts.extend(matches)
        
        # Análisis por componentes patrimoniales
        response_parts.append("\n### 🏛️ **CAPITAL SOCIAL**")
        if "share capital" in text_lower or "capital social" in text_lower:
            response_parts.append("• Movimientos en capital social identificados en el período")
            if found_amounts:
                response_parts.append(f"• Se detectaron cifras específicas relacionadas con capital")
        else:
            response_parts.append("• El capital social requiere análisis específico adicional")
        
        response_parts.append("\n### 💰 **RESERVAS Y RESULTADOS ACUMULADOS**")
        if "reserves" in text_lower or "retained earnings" in text_lower:
            response_parts.append("• Movimientos en reservas y resultados acumulados detectados")
            if "dividend" in text_lower:
                response_parts.append("• Distribución de dividendos identificada durante el período")
        else:
            response_parts.append("• Las reservas y resultados acumulados requieren análisis detallado")
        
        response_parts.append("\n###  **PATRIMONIO TOTAL**")
        if "total equity" in text_lower or "patrimonio total" in text_lower:
            response_parts.append("• Evolución del patrimonio total consolidado identificada")
            if found_amounts:
                response_parts.append(f"• Cifras patrimoniales específicas detectadas: {len(found_amounts)} importes")
        else:
            response_parts.append("• El patrimonio total requiere análisis cuantitativo específico")
        
        response_parts.append("\n###  **CONCLUSIONES BASADAS EN DATOS EXTRAÍDOS**")
        response_parts.append(f"• **Calidad del análisis**: {quality.title()} con {confidence:.1%} de confianza")
        response_parts.append(f"• **Contenido procesado**: {len(text)} caracteres de estado patrimonial")
        
        if len(found_amounts) > 0:
            response_parts.append(f"• **Cifras específicas**: Se identificaron {len(found_amounts)} importes monetarios")
        else:
            response_parts.append("• **Recomendación**: Se requiere acceso a cifras numéricas específicas para análisis cuantitativo completo")
        
        response_parts.append("\n• **Metodología**: Análisis automatizado basado en contenido extraído del PDF")
        response_parts.append("• **Fuente**: Estado de cambios en patrimonio de GarantiBank International N.V.")
        
        return "\n".join(response_parts)

# ===== FUNCIONES DE COMPATIBILIDAD =====
def run_equity_agent(pdf_path: Path, output_dir: Path, max_steps: int = 10) -> Dict[str, Any]:
    """Función principal para ejecutar el agente de equity - compatibilidad"""
    agent = EquityREACTAgent()
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
        description="Equity Agent AUTÓNOMO especializado en Estado de Cambios en Patrimonio - Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
  python agents/equity_agent.py                    # Usa configuración predefinida
  python agents/equity_agent.py --pdf otro.pdf    # Sobreescribe PDF

CARACTERÍSTICAS AUTÓNOMAS:
  - Patrón REACT exitoso 
  - Detección robusta de tool calls  
  - Herramientas específicas para cambios en patrimonio
  - Análisis inteligente con LLM para respuestas detalladas
  - Fallback robusto con respuestas basadas en datos extraídos

Sistema Multi-Agente:
  Esta versión incluye EquityREACTAgent AUTÓNOMO para integración con main_system.py
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
                       help="Pregunta específica sobre cambios en patrimonio")
    
    args = parser.parse_args()
    
    # MOSTRAR CONFIGURACIÓN
    print(" Equity Agent v3.0 AUTÓNOMO Multi-Agent - Configuración Automática")
    print(f" PDF: {args.pdf}")
    print(f" Salida: {args.out}")
    print(f" Azure OpenAI: {AZURE_OPENAI_DEPLOYMENT}")
    print(f" Max steps: {args.maxsteps}")
    print(f" Multi-Agent: EquityREACTAgent AUTÓNOMO class available")
    print(" CARACTERÍSTICAS: Patrón REACT exitoso, análisis inteligente con LLM, respuestas elaboradas")
    
    try:
        # VERIFICAR PDF
        pdf_path = Path(args.pdf)
        output_dir = Path(args.out)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not pdf_path.exists():
            print(f" Error: PDF no encontrado en {pdf_path}")
            return
        
        # CREAR AGENTE Y EJECUTAR
        agent = EquityREACTAgent()
        
        if args.question:
            print(f" Pregunta específica: {args.question}")
            result = agent.run_final_financial_extraction_agent(str(pdf_path), args.question)
        else:
            result = agent.run_final_financial_extraction_agent(str(pdf_path))
        
        # MOSTRAR RESULTADOS
        print(" ==== RESUMEN DE EJECUCIÓN AUTÓNOMO ====")
        print(f"Estado: {' EXITOSO' if result.get('status') == 'task_completed' else '❌ ERROR'}")
        print(f"Pasos completados: {result.get('steps_taken', 0)}")
        print(f"Archivos generados: {result.get('files_generated', 0)}")
        
        if result.get('status') == 'task_completed':
            print(" ==== RESPUESTA ESPECÍFICA ====")
            print(result.get("specific_answer", "No hay respuesta específica disponible"))
        else:
            print(f" Error: {result.get('error_details', 'Error desconocido')}")
        
        print(" Análisis de cambios en patrimonio completado!")
        print(" Clase EquityREACTAgent AUTÓNOMA disponible para sistema multi-agente")
        print(" Versión autónoma con patrón REACT exitoso, análisis inteligente con LLM")
        
    except Exception as e:
        print(f" Error durante la ejecución: {e}")
        raise

if __name__ == "__main__":
    main()
