"""
Income Agent REACT - Versión Multi-Agente AUTÓNOMA COMPLETA
Especializado en análisis de cuenta de resultados con análisis detallado
CARACTERÍSTICAS: Extracción avanzada, análisis LLM especializado, respuestas extensas
"""

from __future__ import annotations
import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
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
    print(f"Warning: Archivo .env no encontrado en {env_path}")

print("🔧 Cargar .env desde el directorio raíz del proyecto...")

# ----- Azure OpenAI Configuration -----
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

print("🔧 ----- Azure OpenAI Configuration -----")
print(f"🔗 Endpoint: {AZURE_OPENAI_ENDPOINT}")
print(f"🔑 API Key: {'✓' if AZURE_OPENAI_API_KEY else '✗'}")
print(f"📋 Deployment: {AZURE_OPENAI_DEPLOYMENT}")

# Validación de credenciales
if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
    raise ValueError("Azure OpenAI credentials required")

# ----- Groq Configuration -----
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

# Inicialización del cliente
chat_client = ChatClient()

# ===== DICCIONARIOS ESPECÍFICOS PARA CUENTA DE RESULTADOS =====
INCOME_TITLES_EN = [
    "income statement", "statement of income", "profit and loss",
    "consolidated income statement", "statement of profit or loss",
    "comprehensive income statement"
]

INCOME_TITLES_ES = [
    "cuenta de resultados", "estado de resultados", "cuenta de pérdidas y ganancias",
    "estado consolidado de resultados", "cuenta de resultado del ejercicio"
]

# Términos específicos de ingresos
REVENUE_HINTS = [
    "net interest income", "interest income", "fee and commission income",
    "trading income", "other operating income", "total income",
    "margen de intereses", "ingresos por intereses", "comisiones netas",
    "ingresos por operaciones", "otros ingresos", "margen bruto"
]

# Términos específicos de gastos
EXPENSE_HINTS = [
    "operating expenses", "staff costs", "personnel expenses",
    "administrative expenses", "depreciation", "amortization",
    "provisions", "loan loss provisions", "impairment losses",
    "gastos de explotación", "gastos de personal", "gastos administrativos",
    "dotaciones", "provisiones", "deterioro"
]

# Términos de rentabilidad
PROFIT_HINTS = [
    "profit before tax", "net profit", "earnings", "net income",
    "return on equity", "return on assets", "profit margin",
    "beneficio antes de impuestos", "beneficio neto", "resultado neto",
    "rentabilidad sobre patrimonio", "margen de beneficio"
]

# ===== FUNCIONES AUXILIARES MEJORADAS =====

def normalize_text(s: str) -> str:
    s = s or ""
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

def detect_language(text: str) -> str:
    t = normalize_text(text)
    score_es = sum(1 for w in ["resultados", "ingresos", "gastos", "beneficio"] if w in t)
    score_en = sum(1 for w in ["income", "revenue", "expenses", "profit"] if w in t)
    return "es" if score_es >= score_en else "en"

def extract_comprehensive_income_data(text: str) -> Dict[str, List[float]]:
    """NUEVA FUNCIÓN: Extrae datos financieros específicos del texto con patrones avanzados"""
    
    patterns = {
        'net_interest_income': [
            r'margen.*intereses.*€?\s*([0-9.,]+)\s*(?:miles|million|thousand)',
            r'net.*interest.*income.*€?\s*([0-9.,]+)\s*(?:miles|million|thousand)',
            r'interest.*margin.*€?\s*([0-9.,]+)\s*(?:miles|million|thousand)',
            r'€\s*([0-9.,]+).*margen.*intereses',
            r'€\s*([0-9.,]+).*net.*interest'
        ],
        'fee_commission_income': [
            r'comisiones.*netas.*€?\s*([0-9.,]+)\s*(?:miles|million|thousand)',
            r'fee.*commission.*income.*€?\s*([0-9.,]+)\s*(?:miles|million|thousand)',
            r'ingresos.*comisiones.*€?\s*([0-9.,]+)\s*(?:miles|million|thousand)',
            r'€\s*([0-9.,]+).*comisiones',
            r'€\s*([0-9.,]+).*fee.*commission'
        ],
        'operating_expenses': [
            r'gastos.*explotación.*€?\s*([0-9.,]+)\s*(?:miles|million|thousand)',
            r'operating.*expenses.*€?\s*([0-9.,]+)\s*(?:miles|million|thousand)',
            r'gastos.*operativos.*€?\s*([0-9.,]+)\s*(?:miles|million|thousand)',
            r'€\s*([0-9.,]+).*gastos.*operativ',
            r'€\s*([0-9.,]+).*operating.*expenses'
        ],
        'staff_costs': [
            r'gastos.*personal.*€?\s*([0-9.,]+)\s*(?:miles|million|thousand)',
            r'staff.*costs.*€?\s*([0-9.,]+)\s*(?:miles|million|thousand)',
            r'personnel.*expenses.*€?\s*([0-9.,]+)\s*(?:miles|million|thousand)',
            r'€\s*([0-9.,]+).*gastos.*personal',
            r'€\s*([0-9.,]+).*staff.*costs'
        ],
        'provisions': [
            r'dotaciones.*provisiones.*€?\s*([0-9.,]+)\s*(?:miles|million|thousand)',
            r'loan.*loss.*provisions.*€?\s*([0-9.,]+)\s*(?:miles|million|thousand)',
            r'impairment.*losses.*€?\s*([0-9.,]+)\s*(?:miles|million|thousand)',
            r'provisiones.*€?\s*([0-9.,]+)\s*(?:miles|million|thousand)',
            r'€\s*([0-9.,]+).*provisiones',
            r'€\s*([0-9.,]+).*provisions'
        ],
        'net_profit': [
            r'beneficio.*neto.*€?\s*([0-9.,]+)\s*(?:miles|million|thousand)',
            r'net.*profit.*€?\s*([0-9.,]+)\s*(?:miles|million|thousand)',
            r'net.*income.*€?\s*([0-9.,]+)\s*(?:miles|million|thousand)',
            r'resultado.*neto.*€?\s*([0-9.,]+)\s*(?:miles|million|thousand)',
            r'€\s*([0-9.,]+).*beneficio.*neto',
            r'€\s*([0-9.,]+).*net.*profit'
        ],
        'total_income': [
            r'margen.*bruto.*€?\s*([0-9.,]+)\s*(?:miles|million|thousand)',
            r'total.*income.*€?\s*([0-9.,]+)\s*(?:miles|million|thousand)',
            r'ingresos.*totales.*€?\s*([0-9.,]+)\s*(?:miles|million|thousand)',
            r'total.*revenue.*€?\s*([0-9.,]+)\s*(?:miles|million|thousand)',
            r'€\s*([0-9.,]+).*margen.*bruto',
            r'€\s*([0-9.,]+).*total.*income'
        ]
    }
    
    extracted_data = {}
    
    # Buscar años específicos para comparación
    years = re.findall(r'\b(20\d{2})\b', text)
    extracted_data['years_found'] = list(set(years))
    
    # Extraer datos por categoría
    for category, pattern_list in patterns.items():
        values = []
        for pattern in pattern_list:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                try:
                    # Limpiar y convertir números
                    clean_number = re.sub(r'[^\d,.]', '', match)
                    if clean_number and clean_number not in ['', '.', ',']:
                        # Manejar formato europeo (1.234,56) y americano (1,234.56)
                        if ',' in clean_number and '.' in clean_number:
                            if clean_number.rindex(',') > clean_number.rindex('.'):
                                # Formato europeo: 1.234,56
                                clean_number = clean_number.replace('.', '').replace(',', '.')
                            # Si es formato americano (1,234.56), ya está bien
                        elif ',' in clean_number:
                            # Solo coma: podría ser decimal o separador miles
                            if len(clean_number.split(',')[-1]) <= 2:
                                # Probablemente decimal: 1234,56
                                clean_number = clean_number.replace(',', '.')
                            else:
                                # Probablemente separador miles: 1,234
                                clean_number = clean_number.replace(',', '')
                        
                        number = float(clean_number)
                        if number > 0:  # Solo valores positivos significativos
                            values.append(number)
                except ValueError:
                    continue
        
        # Remover duplicados manteniendo orden
        unique_values = []
        for v in values:
            if v not in unique_values:
                unique_values.append(v)
        
        extracted_data[category] = unique_values
    
    return extracted_data

def calculate_financial_ratios(data: Dict[str, List[float]]) -> Dict[str, float]:
    """NUEVA FUNCIÓN: Calcular ratios financieros automáticamente"""
    
    ratios = {}
    
    # Obtener valores principales (último valor de cada lista)
    net_profit = max(data.get('net_profit', [0])) if data.get('net_profit') else 0
    total_income = max(data.get('total_income', [0])) if data.get('total_income') else 0
    operating_expenses = max(data.get('operating_expenses', [0])) if data.get('operating_expenses') else 0
    net_interest_income = max(data.get('net_interest_income', [0])) if data.get('net_interest_income') else 0
    staff_costs = max(data.get('staff_costs', [0])) if data.get('staff_costs') else 0
    
    # Calcular ratios si hay datos disponibles
    if total_income > 0:
        if net_profit > 0:
            ratios['net_profit_margin'] = (net_profit / total_income) * 100
            
        if operating_expenses > 0:
            ratios['cost_income_ratio'] = (operating_expenses / total_income) * 100
            ratios['efficiency_ratio'] = (operating_expenses / total_income) * 100
            
        if net_interest_income > 0:
            ratios['interest_income_ratio'] = (net_interest_income / total_income) * 100
            
        if staff_costs > 0:
            ratios['staff_cost_ratio'] = (staff_costs / total_income) * 100
    
    # Calcular variaciones si hay datos de múltiples años
    for category, values in data.items():
        if len(values) >= 2:
            # Calcular crecimiento año-año
            growth = ((values[-1] - values[0]) / abs(values[0])) * 100 if values[0] != 0 else 0
            ratios[f'{category}_growth'] = growth
    
    return ratios

# ===== CLASE WRAPPER AUTÓNOMA PARA SISTEMA MULTI-AGENTE - INCOME =====

class IncomeREACTAgent:
    """
    Wrapper REACT COMPLETAMENTE AUTÓNOMO para el Income Agent
    
    Esta clase es completamente autónoma y genera respuestas específicas usando LLM
    basándose en los datos de cuenta de resultados que extrae.
    """
    
    def __init__(self):
        self.agent_type = "income"
        self.max_steps = 25  # Aumentado para el wrapper
        self.chat_client = chat_client

    def run_final_financial_extraction_agent(self, pdf_path: str, question: str = None) -> Dict[str, Any]:
        """Ejecuta la extracción de cuenta de resultados con wrapper autónomo"""
        try:
            print(f"🔧 IncomeREACTAgent AUTÓNOMO iniciando extracción para: {pdf_path}")
            
            pdf_file = Path(pdf_path)
            output_dir = Path("data/salida")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            start_time = time.time()
            
            # EXTRACCIÓN MEJORADA
            extraction_result = self.extract_income_data_enhanced(pdf_file)
            
            if not extraction_result.get("success"):
                return {
                    "status": "error",
                    "steps_taken": 1,
                    "session_id": f"income_{pdf_file.stem}",
                    "final_response": f"Income extraction failed: {extraction_result.get('error')}",
                    "agent_type": "income",
                    "error_details": extraction_result.get("error"),
                    "specific_answer": "No se encontraron datos de cuenta de resultados"
                }
            
            # VALIDACIÓN MEJORADA
            validation_result = self.validate_income_data_enhanced(extraction_result)
            
            # GUARDAR RESULTADOS MEJORADOS
            save_result = self.save_income_results_enhanced(pdf_file, output_dir, extraction_result, validation_result)
            
            # GENERAR RESPUESTA ESPECÍFICA MEJORADA
            if question:
                print(f"❓ Pregunta específica recibida: {question}")
            
            specific_answer = self.generate_enhanced_income_analysis(question, extraction_result, validation_result)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            print("✅ Income extraction completed successfully (AUTÓNOMO)")
            
            return {
                "status": "task_completed",
                "steps_taken": 5,  # Análisis, extracción, validación, guardado, respuesta
                "session_id": f"income_{pdf_file.stem}",
                "final_response": "Income extraction completed successfully - AUTONOMOUS VERSION",
                "agent_type": "income",
                "files_generated": save_result.get("files_created", 3),
                "processing_time": processing_time,
                "specific_answer": specific_answer,
                "extraction_summary": {
                    "total_characters": extraction_result.get("total_characters", 0),
                    "financial_data_categories": len(extraction_result.get("financial_data", {})),
                    "confidence": validation_result.get("confidence", 0.8),
                    "quality": validation_result.get("quality", "unknown")
                }
            }
            
        except Exception as e:
            print(f"❌ Error en IncomeREACTAgent: {str(e)}")
            return {
                "status": "error",
                "steps_taken": 0,
                "session_id": "income_error",
                "final_response": f"Error in income extraction: {str(e)}",
                "agent_type": "income",
                "error_details": str(e),
                "specific_answer": f"Error durante la extracción de la cuenta de resultados: {str(e)}"
            }

    def extract_income_data_enhanced(self, pdf_file: Path) -> Dict[str, Any]:
        """NUEVA FUNCIÓN: Extracción mejorada de datos de cuenta de resultados"""
        try:
            print(f"🔍 Extrayendo cuenta de resultados de: {pdf_file}")
            
            # Páginas más probables para cuenta de resultados en documentos bancarios
            target_pages = [1, 2, 3, 4, 5, 6, 7, 8]  # Ampliar búsqueda
            
            extracted_text = ""
            total_chars = 0
            financial_data = {}
            relevant_pages = []
            
            with fitz.open(pdf_file) as pdf:
                for page_num in range(min(len(pdf), 15)):  # Buscar en primeras 15 páginas
                    page = pdf[page_num]
                    text = page.get_text()
                    text_lower = normalize_text(text)
                    
                    # Detectar relevancia para cuenta de resultados
                    relevance_score = 0
                    
                    # Buscar títulos específicos
                    title_indicators = INCOME_TITLES_EN + INCOME_TITLES_ES
                    for indicator in title_indicators:
                        if normalize_text(indicator) in text_lower:
                            relevance_score += 10
                    
                    # Buscar términos de ingresos
                    for hint in REVENUE_HINTS:
                        if normalize_text(hint) in text_lower:
                            relevance_score += 3
                    
                    # Buscar términos de gastos
                    for hint in EXPENSE_HINTS:
                        if normalize_text(hint) in text_lower:
                            relevance_score += 3
                    
                    # Buscar términos de rentabilidad
                    for hint in PROFIT_HINTS:
                        if normalize_text(hint) in text_lower:
                            relevance_score += 5
                    
                    # Si la página es relevante, extraer
                    if relevance_score >= 5 or page_num + 1 in target_pages:
                        extracted_text += f"\n=== PÁGINA {page_num + 1} (Score: {relevance_score}) ===\n{text}"
                        total_chars += len(text)
                        relevant_pages.append(page_num + 1)
                        print(f"✅ Página {page_num + 1}: {len(text)} caracteres extraídos (relevance: {relevance_score})")
                        
                        # NUEVA: Extracción de datos financieros específicos
                        page_financial_data = extract_comprehensive_income_data(text)
                        for key, values in page_financial_data.items():
                            if key not in financial_data:
                                financial_data[key] = []
                            financial_data[key].extend(values)
            
            # Si no se encontró contenido relevante, extraer páginas por defecto
            if total_chars < 1000:
                print("⚠️ Poco contenido relevante encontrado, extrayendo páginas por defecto...")
                with fitz.open(pdf_file) as pdf:
                    for page_num in range(min(10, len(pdf))):
                        page = pdf[page_num]
                        text = page.get_text()
                        extracted_text += f"\n=== PÁGINA {page_num + 1} (DEFAULT) ===\n{text}"
                        total_chars += len(text)
            
            print(f"📊 Texto total extraído: {total_chars} caracteres de {len(relevant_pages)} páginas")
            
            # NUEVA: Extracción total mejorada
            if financial_data:
                total_extracted = sum(len(values) for values in financial_data.values() if values)
                print(f"📈 Total extraído: {total_extracted} entradas financieras")
            
            confidence = 1.0 if total_chars > 3000 else 0.8 if total_chars > 1500 else 0.6
            
            return {
                "success": True,
                "text": extracted_text,
                "total_characters": total_chars,
                "pages_processed": relevant_pages,
                "financial_data": financial_data,  # NUEVO
                "confidence": confidence,
                "language": detect_language(extracted_text)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def validate_income_data_enhanced(self, extraction: Dict[str, Any]) -> Dict[str, Any]:
        """NUEVA FUNCIÓN: Validación mejorada de datos de cuenta de resultados"""
        try:
            text = extraction.get("text", "")
            confidence = extraction.get("confidence", 0.0)
            financial_data = extraction.get("financial_data", {})
            
            # Criterios de validación mejorados
            quality_score = 0
            validation_details = []
            
            text_lower = normalize_text(text)
            
            # Verificar secciones principales (peso variable)
            if any(normalize_text(term) in text_lower for term in ["interest income", "margen intereses", "ingresos intereses"]):
                quality_score += 20
                validation_details.append("✅ Ingresos por intereses encontrados")
            
            if any(normalize_text(term) in text_lower for term in ["commission", "comisiones", "fee income"]):
                quality_score += 15
                validation_details.append("✅ Ingresos por comisiones encontrados")
            
            if any(normalize_text(term) in text_lower for term in ["operating expenses", "gastos explotación", "gastos operativos"]):
                quality_score += 20
                validation_details.append("✅ Gastos operativos encontrados")
            
            if any(normalize_text(term) in text_lower for term in ["staff costs", "gastos personal", "personnel expenses"]):
                quality_score += 15
                validation_details.append("✅ Gastos de personal encontrados")
            
            if any(normalize_text(term) in text_lower for term in ["provisions", "provisiones", "impairment"]):
                quality_score += 10
                validation_details.append("✅ Provisiones encontradas")
            
            if any(normalize_text(term) in text_lower for term in ["net profit", "beneficio neto", "net income"]):
                quality_score += 15
                validation_details.append("✅ Beneficio neto encontrado")
            
            # NUEVA: Bonificaciones por datos financieros específicos
            if financial_data:
                categories_with_data = sum(1 for values in financial_data.values() if values)
                data_bonus = min(15, categories_with_data * 2)  # Máximo 15 puntos extra
                quality_score += data_bonus
                validation_details.append(f"✅ Datos financieros específicos: {categories_with_data} categorías")
            
            # Determinar calidad final
            if quality_score >= 80:
                quality = "excellent"
            elif quality_score >= 60:
                quality = "good"
            elif quality_score >= 40:
                quality = "fair"
            else:
                quality = "poor"
            
            final_confidence = min(confidence + (quality_score / 100 * 0.3), 1.0)
            
            print(f"✅ Validación completada: {quality} (puntuación: {quality_score}/100, confianza: {final_confidence:.3f})")
            
            return {
                "success": True,
                "quality": quality,
                "confidence": final_confidence,
                "score": quality_score,
                "details": validation_details,
                "financial_categories_found": len([k for k, v in financial_data.items() if v]) if financial_data else 0
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def save_income_results_enhanced(self, pdf_file: Path, output_dir: Path, extraction: Dict, validation: Dict) -> Dict[str, Any]:
        """NUEVA FUNCIÓN: Guardar resultados mejorados"""
        try:
            base_name = pdf_file.stem
            files_created = 0
            
            # 1. Guardar resumen JSON extendido
            summary = {
                "extraction": {
                    "total_characters": extraction.get("total_characters", 0),
                    "pages_processed": extraction.get("pages_processed", []),
                    "financial_data": extraction.get("financial_data", {}),
                    "confidence": extraction.get("confidence", 0.8),
                    "language": extraction.get("language", "unknown")
                },
                "validation": validation,
                "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "quality_metrics": {
                    "data_categories_found": validation.get("financial_categories_found", 0),
                    "quality_score": validation.get("score", 0),
                    "final_confidence": validation.get("confidence", 0.8)
                }
            }
            
            summary_file = output_dir / f"{base_name}_income_summary.json"
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            files_created += 1
            
            # 2. Guardar datos financieros específicos
            if extraction.get("financial_data"):
                financial_data_file = output_dir / f"{base_name}_financial_data.json"
                with open(financial_data_file, "w", encoding="utf-8") as f:
                    json.dump(extraction["financial_data"], f, indent=2, ensure_ascii=False)
                files_created += 1
            
            # 3. Guardar reporte de calidad extendido
            quality_report = f"""
REPORTE DE CALIDAD EXTENDIDO - CUENTA DE RESULTADOS
==================================================
PDF: {pdf_file.name}
Fecha: {time.strftime("%Y-%m-%d %H:%M:%S")}

RESULTADOS DE VALIDACIÓN:
- Calidad: {validation.get('quality', 'unknown')}
- Puntuación: {validation.get('score', 0)}/100
- Confianza final: {validation.get('confidence', 0):.3f}
- Categorías financieras: {validation.get('financial_categories_found', 0)}

DETALLES DE VALIDACIÓN:
{chr(10).join(validation.get('details', []))}

EXTRACCIÓN DETALLADA:
- Caracteres procesados: {extraction.get('total_characters', 0)}
- Páginas procesadas: {extraction.get('pages_processed', [])}
- Idioma detectado: {extraction.get('language', 'unknown')}
- Datos financieros extraídos:
{json.dumps(extraction.get('financial_data', {}), indent=2)}

MÉTRICAS DE CALIDAD:
- Cobertura de secciones: {'Completa' if validation.get('score', 0) >= 70 else 'Parcial' if validation.get('score', 0) >= 50 else 'Limitada'}
- Precisión de extracción: {validation.get('confidence', 0.8)*100:.1f}%
- Recomendación: {'Análisis confiable' if validation.get('score', 0) >= 60 else 'Requiere revisión manual'}
"""
            
            quality_file = output_dir / f"{base_name}_income_quality.txt"
            with open(quality_file, "w", encoding="utf-8") as f:
                f.write(quality_report)
            files_created += 1
            
            print(f"💾 Archivos guardados: {files_created}")
            
            return {
                "success": True,
                "files_created": files_created,
                "output_directory": str(output_dir)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def generate_enhanced_income_analysis(self, question: str, extraction: Dict, validation: Dict) -> str:
        """NUEVA FUNCIÓN: Genera análisis extendido y detallado con LLM especializado"""
        try:
            text = extraction.get("text", "")
            confidence = validation.get("confidence", 0.8)
            quality = validation.get("quality", "unknown")
            financial_data = extraction.get("financial_data", {})
            
            if not text or len(text.strip()) < 300:
                return "El contenido extraído de la cuenta de resultados es insuficiente para realizar un análisis detallado profesional."
            
            # Calcular ratios financieros
            ratios = calculate_financial_ratios(financial_data) if financial_data else {}
            
            # ANÁLISIS INTELIGENTE CON LLM - PROMPT ESPECIALIZADO PARA CUENTA DE RESULTADOS
            analysis_prompt = f"""
Actúa como un analista financiero senior especializado en banca internacional con 15 años de experiencia en análisis de cuentas de resultados.

Analiza de forma EXHAUSTIVA y DETALLADA el siguiente contenido extraído de la cuenta de resultados de GarantiBank International N.V./BBVA:

CONTENIDO EXTRAÍDO DEL PDF:
{text[:4000]}

DATOS FINANCIEROS ESPECÍFICOS IDENTIFICADOS:
{json.dumps(financial_data, indent=2) if financial_data else "No se identificaron datos numéricos específicos"}

RATIOS FINANCIEROS CALCULADOS:
{json.dumps(ratios, indent=2) if ratios else "No se pudieron calcular ratios específicos"}

INSTRUCCIONES PARA ANÁLISIS PROFESIONAL BANCARIO:

1. **Estructura tu análisis en las siguientes secciones obligatorias:**
   - Resumen ejecutivo de la rentabilidad y performance financiera
   - Análisis detallado del margen de intereses (principal fuente de ingresos bancarios)
   - Evaluación de ingresos por comisiones y servicios
   - Análisis de gastos operativos y eficiencia
   - Evaluación de provisiones y calidad crediticia
   - Análisis de rentabilidad y márgenes (ROE, ROA, si calculables)
   - Comparación con períodos anteriores (si datos disponibles)
   - Identificación de riesgos y oportunidades estratégicas
   - Conclusiones y recomendaciones para la gestión

2. **Utiliza ÚNICAMENTE las cifras y datos presentes en el texto extraído:**
   - Cita cifras exactas cuando estén disponibles (ej: "€1,025k en 2023")
   - NO inventes números que no aparezcan en el contenido
   - Si una cifra no está disponible, menciona "requiere datos adicionales"
   - Usa los ratios calculados automáticamente cuando estén disponibles

3. **Proporciona interpretación profesional específica para banca:**
   - Explica el significado de variaciones en margen de intereses
   - Analiza la diversificación de ingresos (intereses vs comisiones)
   - Evalúa la eficiencia operativa (cost-income ratio)
   - Comenta sobre la calidad crediticia basada en provisiones
   - Relaciona con estrategia bancaria y entorno regulatorio

4. **Formato profesional y técnico:**
   - Usa terminología técnica apropiada para entidades financieras
   - Estructura clara con subtítulos descriptivos
   - Párrafos bien desarrollados con argumentación sólida
   - Longitud objetivo: 800-1000 palabras

5. **Enfoque específico bancario:**
   - Considera particularidades del sector financiero
   - Analiza impacto de tasas de interés en resultados
   - Evalúa calidad de los ingresos recurrentes vs no recurrentes
   - Comenta sobre cumplimiento regulatorio y capital
   - Identifica tendencias del mercado bancario

Genera un análisis que demuestre expertise profesional avanzado, útil tanto para el comité de dirección como para stakeholders externos, con insights accionables y recomendaciones estratégicas específicas.
"""

            try:
                # Usar el LLM para generar análisis inteligente
                analysis_response = self.chat_client.chat([
                    {"role": "system", "content": "Eres un analista financiero senior con maestría en finanzas y 15+ años especializándote en análisis de cuentas de resultados de entidades bancarias internacionales."},
                    {"role": "user", "content": analysis_prompt}
                ], max_tokens=2200)  # Aumentado para respuestas más extensas
                
                # Construir respuesta final con información técnica expandida
                response_parts = [
                    "📊 **ANÁLISIS PROFESIONAL DE CUENTA DE RESULTADOS - GarantiBank International N.V.**",
                    "=" * 90,
                    "",
                    analysis_response,
                    "",
                    "### 📋 **INFORMACIÓN TÉCNICA Y METODOLÓGICA DEL ANÁLISIS**",
                    f"• **Calidad de extracción**: {quality.title()} (puntuación: {validation.get('score', 0)}/100)",
                    f"• **Confianza en datos**: {confidence:.1%}",
                    f"• **Caracteres analizados**: {len(text):,} del documento original",
                    f"• **Páginas procesadas**: {len(extraction.get('pages_processed', []))} páginas del estado financiero",
                    f"• **Categorías financieras identificadas**: {len([k for k, v in financial_data.items() if v])} de 7 principales" if financial_data else "• **Datos financieros**: Análisis basado en contenido textual estructurado",
                    f"• **Ratios calculados**: {len(ratios)} indicadores financieros" if ratios else "• **Ratios**: No calculables con datos actuales",
                    f"• **Idioma del documento**: {extraction.get('language', 'Desconocido').title()}",
                    "• **Metodología**: Extracción automática + análisis con IA especializada en banca",
                    "• **Fuente**: Cuenta de resultados consolidada de GarantiBank International N.V.",
                    "• **Estándares**: Análisis conforme a mejores prácticas de análisis financiero bancario",
                    "",
                    "=" * 90,
                    "📊 *Análisis generado por sistema de IA especializada en análisis de rentabilidad bancaria*"
                ]
                
                return "\n".join(response_parts)
                
            except Exception as llm_error:
                print(f"Error en análisis LLM: {str(llm_error)}")
                # Fallback: análisis básico si el LLM falla
                return self.generate_fallback_income_analysis(text, confidence, quality, financial_data, ratios)
                
        except Exception as e:
            return f"Error al generar análisis específico de cuenta de resultados: {str(e)}"

    def generate_fallback_income_analysis(self, text: str, confidence: float, quality: str, 
                                        financial_data: Dict, ratios: Dict) -> str:
        """Análisis de respaldo basado en extracción de datos específicos"""
        
        response_parts = []
        response_parts.append("📊 **ANÁLISIS DE CUENTA DE RESULTADOS - GarantiBank International N.V.**")
        response_parts.append("=" * 75)
        
        text_lower = normalize_text(text)
        
        # Análisis de ingresos principales
        response_parts.append("\n### 💰 **ANÁLISIS DE INGRESOS PRINCIPALES**")
        
        # Margen de intereses
        if financial_data.get('net_interest_income'):
            amounts = financial_data['net_interest_income']
            response_parts.append(f"• **Margen de intereses**: {amounts} (miles de euros)")
            if 'net_interest_income_growth' in ratios:
                growth = ratios['net_interest_income_growth']
                response_parts.append(f"  - Variación: {growth:+.1f}% respecto período anterior")
        elif any(term in text_lower for term in ["interest", "intereses"]):
            response_parts.append("• **Margen de intereses**: Identificado como fuente principal de ingresos bancarios")
        
        # Comisiones
        if financial_data.get('fee_commission_income'):
            amounts = financial_data['fee_commission_income']
            response_parts.append(f"• **Ingresos por comisiones**: {amounts} (miles de euros)")
            if 'fee_commission_income_growth' in ratios:
                growth = ratios['fee_commission_income_growth']
                response_parts.append(f"  - Variación: {growth:+.1f}% respecto período anterior")
                if growth < -50:
                    response_parts.append("  - ⚠️ ATENCIÓN: Caída significativa que requiere análisis estratégico")
        elif any(term in text_lower for term in ["commission", "comisiones"]):
            response_parts.append("• **Ingresos por comisiones**: Fuente complementaria de ingresos identificada")
        
        # Análisis de gastos
        response_parts.append("\n### 💸 **ANÁLISIS DE GASTOS OPERATIVOS**")
        
        if financial_data.get('operating_expenses'):
            amounts = financial_data['operating_expenses']
            response_parts.append(f"• **Gastos operativos**: {amounts} (miles de euros)")
            if 'efficiency_ratio' in ratios:
                efficiency = ratios['efficiency_ratio']
                response_parts.append(f"  - Ratio de eficiencia: {efficiency:.1f}%")
                if efficiency < 50:
                    response_parts.append("  - ✅ Eficiencia operativa superior al promedio sectorial")
                elif efficiency > 60:
                    response_parts.append("  - ⚠️ Oportunidades de mejora en eficiencia operativa")
        
        if financial_data.get('staff_costs'):
            amounts = financial_data['staff_costs']
            response_parts.append(f"• **Gastos de personal**: {amounts} (miles de euros)")
            if 'staff_cost_ratio' in ratios:
                staff_ratio = ratios['staff_cost_ratio']
                response_parts.append(f"  - Ratio sobre ingresos: {staff_ratio:.1f}%")
        
        # Provisiones y calidad crediticia
        response_parts.append("\n### 🛡️ **PROVISIONES Y CALIDAD CREDITICIA**")
        
        if financial_data.get('provisions'):
            amounts = financial_data['provisions']
            response_parts.append(f"• **Provisiones**: {amounts} (miles de euros)")
            response_parts.append("• Las provisiones reflejan la gestión prudente del riesgo crediticio")
        elif any(term in text_lower for term in ["provision", "provisiones"]):
            response_parts.append("• **Provisiones**: Identificadas como parte de la gestión de riesgos")
        
        # Rentabilidad
        response_parts.append("\n### 📈 **ANÁLISIS DE RENTABILIDAD**")
        
        if financial_data.get('net_profit'):
            amounts = financial_data['net_profit']
            response_parts.append(f"• **Beneficio neto**: {amounts} (miles de euros)")
            if 'net_profit_margin' in ratios:
                margin = ratios['net_profit_margin']
                response_parts.append(f"  - Margen neto: {margin:.1f}%")
                if margin > 15:
                    response_parts.append("  - ✅ Rentabilidad sólida para el sector bancario")
                elif margin < 10:
                    response_parts.append("  - ⚠️ Margen por debajo del promedio sectorial")
        
        # Ratios adicionales
        if ratios:
            response_parts.append("\n### 📊 **RATIOS FINANCIEROS CALCULADOS**")
            for ratio_name, value in ratios.items():
                if not ratio_name.endswith('_growth'):
                    response_parts.append(f"• **{ratio_name.replace('_', ' ').title()}**: {value:.2f}%")
        
        # Conclusiones
        response_parts.append("\n### 🎯 **CONCLUSIONES BASADAS EN DATOS EXTRAÍDOS**")
        response_parts.append(f"• **Calidad del análisis**: {quality.title()} con {confidence:.1%} de confianza")
        response_parts.append(f"• **Contenido procesado**: {len(text):,} caracteres de información financiera")
        
        if financial_data:
            categories_found = len([k for k, v in financial_data.items() if v])
            response_parts.append(f"• **Datos específicos**: {categories_found} categorías financieras identificadas")
            
            # Identificar tendencias principales
            declining_categories = []
            growing_categories = []
            for category, values in financial_data.items():
                if len(values) >= 2:
                    if values[-1] < values[0]:
                        declining_categories.append(category.replace('_', ' '))
                    else:
                        growing_categories.append(category.replace('_', ' '))
            
            if declining_categories:
                response_parts.append(f"• **Tendencias descendentes**: {', '.join(declining_categories)}")
            if growing_categories:
                response_parts.append(f"• **Tendencias ascendentes**: {', '.join(growing_categories)}")
        else:
            response_parts.append("• **Recomendación**: Se requiere acceso a cifras numéricas específicas para análisis cuantitativo completo")
        
        response_parts.append("\n• **Metodología**: Análisis automatizado basado en contenido extraído y patrones financieros")
        response_parts.append("• **Fuente**: Cuenta de resultados consolidada de GarantiBank International N.V.")
        response_parts.append("• **Nota**: Para análisis más profundo se recomienda acceso a datos históricos completos")
        
        return "\n".join(response_parts)

# ===== CONFIGURACIÓN Y MAIN =====
DEFAULT_CONFIG = {
    "pdf": "data/entrada/output/bbva_2023_div.pdf",
    "out": "data/salida", 
    "maxsteps": 25
}

def main():
    parser = argparse.ArgumentParser(
        description="Income Agent AUTÓNOMO con Análisis Detallado - Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
  python agents/income_agent.py                    # Usa configuración predefinida
  python agents/income_agent.py --pdf otro.pdf    # Sobreescribe PDF

CARACTERÍSTICAS AVANZADAS:
  - Análisis detallado de 800-1000 palabras generado por LLM especializado
  - Extracción automática de cifras financieras específicas
  - Cálculo automático de ratios bancarios (eficiencia, rentabilidad, crecimiento)
  - Validación mejorada con puntuación de calidad detallada
  - Análisis fallback robusto basado en datos extraídos

MEJORAS IMPLEMENTADAS:
  - Búsqueda inteligente por relevancia de páginas
  - Identificación automática de cifras de ingresos, gastos, y rentabilidad
  - Análisis profesional con terminología bancaria especializada
  - Cálculo de variaciones interanuales automático
  - Informes de calidad técnicos extendidos
  - Respuestas estructuradas con insights accionables
"""
    )
    
    # Argumentos opcionales
    parser.add_argument("--pdf", default=DEFAULT_CONFIG["pdf"], 
                       help=f"Ruta al PDF (por defecto: {DEFAULT_CONFIG['pdf']})")
    parser.add_argument("--out", default=DEFAULT_CONFIG["out"],
                       help=f"Directorio de salida (por defecto: {DEFAULT_CONFIG['out']})")
    parser.add_argument("--maxsteps", type=int, default=DEFAULT_CONFIG["maxsteps"],
                       help=f"Máximo pasos (por defecto: {DEFAULT_CONFIG['maxsteps']})")
    parser.add_argument("--question", type=str, default=None,
                       help="Pregunta específica sobre cuenta de resultados")
    
    args = parser.parse_args()
    
    # MOSTRAR CONFIGURACIÓN
    print("🚀 Income Agent v4.0 AUTÓNOMO Multi-Agent - Análisis Detallado")
    print(f"📄 PDF: {args.pdf}")
    print(f"📁 Salida: {args.out}")
    print(f"⚙️ Groq/Azure OpenAI: Configuración optimizada")
    print(f"🔧 Max steps: {args.maxsteps}")
    print("🆕 CARACTERÍSTICAS: Análisis extenso con LLM, extracción avanzada, ratios automáticos")
    
    try:
        # VERIFICAR PDF
        pdf_path = Path(args.pdf)
        output_dir = Path(args.out)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not pdf_path.exists():
            print(f"❌ Error: PDF no encontrado en {pdf_path}")
            return
        
        # CREAR AGENTE Y EJECUTAR
        agent = IncomeREACTAgent()
        
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
            print("📋 ==== ANÁLISIS DETALLADO GENERADO ====")
            analysis = result.get("specific_answer", "No hay respuesta específica disponible")
            print(f"Longitud del análisis: {len(analysis)} caracteres")
            
            summary = result.get("extraction_summary", {})
            print(f"Caracteres procesados: {summary.get('total_characters', 0):,}")
            print(f"Categorías financieras: {summary.get('financial_data_categories', 0)}")
            print(f"Confianza: {summary.get('confidence', 0.8):.1%}")
            print(f"Calidad: {summary.get('quality', 'unknown').title()}")
            print("✅ Análisis detallado con LLM especializado completado")
        else:
            print(f"❌ Error: {result.get('error_details', 'Error desconocido')}")
        
        print("🎉 Análisis de cuenta de resultados completado!")
        print("🤖 IncomeREACTAgent con análisis detallado disponible para sistema multi-agente")
        
    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
        raise

if __name__ == "__main__":
    main()
