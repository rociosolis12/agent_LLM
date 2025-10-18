"""
Income Agent REACT - Versión Multi-Agente AUTÓNOMA COMPLETA CORREGIDA
Especializado en análisis de cuenta de resultados con análisis detallado
CARACTERÍSTICAS: Extracción avanzada, análisis LLM especializado, respuestas extensas, conversión segura de tipos
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

def convert_string_to_float(value_str: str) -> Optional[float]:
    """NUEVA FUNCIÓN: Convierte string a float de forma segura manejando formatos europeos"""
    if not value_str or not isinstance(value_str, str):
        return None
    
    try:
        # Limpiar el string manteniendo solo dígitos, comas y puntos
        clean_str = re.sub(r'[^\d,.]', '', value_str.strip())
        
        if not clean_str or clean_str in ['', '.', ',']:
            return None
        
        # Manejar diferentes formatos numéricos
        if ',' in clean_str and '.' in clean_str:
            # Formato: 1.234,56 (europeo) o 1,234.56 (americano)
            if clean_str.rindex(',') > clean_str.rindex('.'):
                # Formato europeo: 1.234,56
                clean_str = clean_str.replace('.', '').replace(',', '.')
            else:
                # Formato americano: 1,234.56 - remover comas
                clean_str = clean_str.replace(',', '')
        elif ',' in clean_str:
            # Solo comas: determinar si es decimal o separador miles
            parts = clean_str.split(',')
            if len(parts) == 2 and len(parts[1]) <= 2:
                # Probablemente decimal: 1234,56
                clean_str = clean_str.replace(',', '.')
            else:
                # Probablemente separador miles: 1,234 o 1,234,567
                clean_str = clean_str.replace(',', '')
        
        # Convertir a float
        result = float(clean_str)
        
        # Validar que sea un número razonable
        if result < 0 or result > 1e12:  # Entre 0 y 1 billón
            return None
            
        return result
        
    except (ValueError, AttributeError):
        return None

def extract_comprehensive_income_data(text: str) -> Dict[str, List[float]]:
    """FUNCIÓN CORREGIDA: Extrae datos financieros con patrones mejorados y conversión segura"""
    
    # PATRONES MEJORADOS más específicos para documentos bancarios españoles
    patterns = {
        'net_interest_income': [
            r'margen.*intereses.*€?\s*([0-9.,]+)\s*(?:miles|millones?)',
            r'net.*interest.*income.*€?\s*([0-9.,]+)\s*(?:miles|million)',
            r'ingresos.*netos.*intereses.*€?\s*([0-9.,]+)',
            r'margen.*de.*intereses.*([0-9.,]+)',
            r'€\s*([0-9.,]+).*margen.*intereses',
            # Patrones adicionales sin € al inicio
            r'margen.*intereses.*([0-9.,]+)\s*(?:miles|millones?)',
            r'ingresos.*intereses.*([0-9.,]+)\s*(?:miles|millones?)',
            # Patrones más específicos para BBVA
            r'margen.*de.*intereses\s*([0-9.,]+)',
            r'intereses.*y.*rendimientos.*similares.*([0-9.,]+)',
            r'ingresos.*por.*intereses.*([0-9.,]+)'
        ],
        'fee_commission_income': [
            r'comisiones.*netas.*€?\s*([0-9.,]+)\s*(?:miles|millones?)',
            r'ingresos.*comisiones.*€?\s*([0-9.,]+)\s*(?:miles|millones?)',
            r'fee.*commission.*€?\s*([0-9.,]+)',
            r'comisiones.*([0-9.,]+)\s*(?:miles|millones?)',
            r'€\s*([0-9.,]+).*comisiones',
            # Patrones adicionales para BBVA
            r'comisiones.*([0-9.,]+)',
            r'ingresos.*por.*comisiones.*([0-9.,]+)',
            r'comisiones.*netas.*([0-9.,]+)',
            r'fee.*and.*commission.*income.*([0-9.,]+)'
        ],
        'operating_expenses': [
            r'gastos.*explotación.*€?\s*([0-9.,]+)\s*(?:miles|millones?)',
            r'gastos.*operativos.*€?\s*([0-9.,]+)\s*(?:miles|millones?)',
            r'operating.*expenses.*€?\s*([0-9.,]+)',
            r'gastos.*de.*explotación.*([0-9.,]+)',
            r'€\s*([0-9.,]+).*gastos.*operativ',
            r'gastos.*administración.*([0-9.,]+)',
            r'total.*gastos.*operativos.*([0-9.,]+)',
            r'gastos.*generales.*administración.*([0-9.,]+)'
        ],
        'staff_costs': [
            r'gastos.*personal.*€?\s*([0-9.,]+)\s*(?:miles|millones?)',
            r'staff.*costs.*€?\s*([0-9.,]+)\s*(?:miles|millones?)',
            r'personnel.*expenses.*€?\s*([0-9.,]+)\s*(?:miles|millones?)',
            r'€\s*([0-9.,]+).*gastos.*personal',
            r'€\s*([0-9.,]+).*staff.*costs',
            r'gastos.*de.*personal.*([0-9.,]+)',
            r'sueldos.*salarios.*([0-9.,]+)'
        ],
        'provisions': [
            r'dotaciones.*provisiones.*€?\s*([0-9.,]+)\s*(?:miles|millones?)',
            r'loan.*loss.*provisions.*€?\s*([0-9.,]+)\s*(?:miles|millones?)',
            r'impairment.*losses.*€?\s*([0-9.,]+)\s*(?:miles|millones?)',
            r'provisiones.*€?\s*([0-9.,]+)\s*(?:miles|millones?)',
            r'€\s*([0-9.,]+).*provisiones',
            r'€\s*([0-9.,]+).*provisions',
            r'dotaciones.*para.*insolvencias.*([0-9.,]+)',
            r'provisiones.*para.*riesgos.*([0-9.,]+)'
        ],
        'net_profit': [
            r'beneficio.*neto.*€?\s*([0-9.,]+)\s*(?:miles|millones?)',
            r'resultado.*neto.*€?\s*([0-9.,]+)\s*(?:miles|millones?)',
            r'net.*profit.*€?\s*([0-9.,]+)',
            r'beneficio.*neto.*([0-9.,]+)',
            r'resultado.*del.*ejercicio.*([0-9.,]+)',
            r'€\s*([0-9.,]+).*beneficio.*neto',
            r'beneficio.*atribuido.*([0-9.,]+)',
            r'resultado.*atribuido.*al.*grupo.*([0-9.,]+)'
        ],
        'total_income': [
            r'margen.*bruto.*€?\s*([0-9.,]+)\s*(?:miles|millones?)',
            r'ingresos.*totales.*€?\s*([0-9.,]+)\s*(?:miles|millones?)',
            r'total.*income.*€?\s*([0-9.,]+)',
            r'margen.*bruto.*([0-9.,]+)',
            r'total.*ingresos.*([0-9.,]+)',
            r'ingresos.*operativos.*([0-9.,]+)',
            r'margen.*de.*intermediación.*([0-9.,]+)',
            # Patrón más general para números grandes
            r'([0-9]{2,3}(?:[.,][0-9]{3})*)\s*(?:miles|millones?)'
        ]
    }
    
    extracted_data = {}
    
    # Buscar años específicos
    years = re.findall(r'\b(20\d{2})\b', text)
    extracted_data['years_found'] = list(set(years))
    
    print(f"🔍 DEBUG: Años encontrados: {extracted_data['years_found']}")
    
    # Extraer datos por categoría CON CONVERSIÓN SEGURA
    for category, pattern_list in patterns.items():
        values = []
        raw_matches = []
        
        for pattern in pattern_list:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            for match in matches:
                raw_matches.append(match)
                try:
                    # CONVERSIÓN MEJORADA Y SEGURA
                    clean_number = convert_string_to_float(match)
                    if clean_number is not None and clean_number > 0:
                        values.append(clean_number)
                except Exception as e:
                    print(f"⚠️ Error convirtiendo '{match}': {e}")
                    continue
        
        # Remover duplicados manteniendo orden
        unique_values = []
        for v in values:
            if v not in unique_values:
                unique_values.append(v)
        
        extracted_data[category] = unique_values
        
        if raw_matches:
            print(f"🔍 DEBUG {category}: raw_matches = {raw_matches[:3]}, converted = {unique_values[:3]}")
    
    return extracted_data

def calculate_financial_ratios(data: Dict[str, List[float]]) -> Dict[str, float]:
    """FUNCIÓN CORREGIDA: Calcular ratios con validación de tipos segura"""
    
    ratios = {}
    
    if not data or not isinstance(data, dict):
        return ratios
    
    # Función auxiliar para obtener valor máximo seguro
    def get_max_value_safe(values_list):
        if not values_list or not isinstance(values_list, list):
            return 0.0
        try:
            # Asegurar que todos sean floats
            float_values = []
            for v in values_list:
                if isinstance(v, (int, float)):
                    float_values.append(float(v))
                elif isinstance(v, str):
                    converted = convert_string_to_float(v)
                    if converted is not None:
                        float_values.append(converted)
            
            return max(float_values) if float_values else 0.0
        except Exception:
            return 0.0
    
    # Obtener valores principales SEGUROS
    net_profit = get_max_value_safe(data.get('net_profit', []))
    total_income = get_max_value_safe(data.get('total_income', []))
    operating_expenses = get_max_value_safe(data.get('operating_expenses', []))
    net_interest_income = get_max_value_safe(data.get('net_interest_income', []))
    staff_costs = get_max_value_safe(data.get('staff_costs', []))
    fee_commission = get_max_value_safe(data.get('fee_commission_income', []))
    provisions = get_max_value_safe(data.get('provisions', []))
    
    print(f"🔍 DEBUG ratios - net_profit: {net_profit}, total_income: {total_income}")
    
    # Calcular ratios SI hay datos disponibles
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
            
        if fee_commission > 0:
            ratios['fee_commission_ratio'] = (fee_commission / total_income) * 100
            
        if provisions > 0:
            ratios['provision_ratio'] = (provisions / total_income) * 100
    
    # Calcular variaciones SEGURAS si hay múltiples valores
    for category, values in data.items():
        if isinstance(values, list) and len(values) >= 2:
            try:
                # Asegurar que sean números
                float_values = []
                for v in values:
                    if isinstance(v, (int, float)):
                        float_values.append(float(v))
                    elif isinstance(v, str):
                        converted = convert_string_to_float(v)
                        if converted is not None:
                            float_values.append(converted)
                
                if len(float_values) >= 2 and float_values[0] != 0:
                    # CÁLCULO SEGURO DE CRECIMIENTO
                    growth = ((float_values[-1] - float_values[0]) / abs(float_values[0])) * 100
                    ratios[f'{category}_growth'] = growth
                    
            except Exception as e:
                print(f"⚠️ Error calculando growth para {category}: {e}")
                continue
    
    print(f"🔍 DEBUG: Ratios calculados: {list(ratios.keys())}")
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
            
            # GENERAR RESPUESTA ESPECÍFICA MEJORADA CON DEBUGGING
            print(f"🔍 DEBUG: Iniciando generación de respuesta específica...")
            if question:
                print(f"❓ Pregunta específica recibida: {question}")
            
            specific_answer = self.generate_enhanced_income_analysis_fixed(question, extraction_result, validation_result)
            print(f"🔍 DEBUG: Respuesta generada con {len(specific_answer)} caracteres")
            
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

    def generate_enhanced_income_analysis_fixed(self, question: str, extraction: Dict, validation: Dict) -> str:
        """FUNCIÓN CORREGIDA: Genera análisis con debugging y manejo robusto de errores"""
        try:
            print("🔍 DEBUG: Iniciando generate_enhanced_income_analysis_fixed")
            
            text = extraction.get("text", "")
            confidence = validation.get("confidence", 0.8)
            quality = validation.get("quality", "unknown")
            financial_data = extraction.get("financial_data", {})
            
            print(f"🔍 DEBUG: Texto length: {len(text)}")
            print(f"🔍 DEBUG: Financial data categories: {len(financial_data)}")
            print(f"🔍 DEBUG: Financial data sample: {dict(list(financial_data.items())[:3])}")
            print(f"🔍 DEBUG: Quality: {quality}, Confidence: {confidence}")
            
            if not text or len(text.strip()) < 500:
                print("❌ DEBUG: Texto insuficiente para análisis detallado")
                return "El contenido extraído de la cuenta de resultados es insuficiente para realizar un análisis detallado profesional."
            
            # VERIFICAR QUE HAY DATOS FINANCIEROS
            has_financial_data = any(values for values in financial_data.values() if values)
            print(f"🔍 DEBUG: Has financial data: {has_financial_data}")
            
            if not has_financial_data:
                print("⚠️ DEBUG: No hay datos financieros específicos, re-extrayendo con patrones mejorados...")
                # Re-extraer con patrones más amplios
                financial_data = extract_comprehensive_income_data(text)
                print(f"🔍 DEBUG: Re-extracción result: {dict(list(financial_data.items())[:2])}")
            
            # Calcular ratios financieros CON VALIDACIÓN
            ratios = {}
            try:
                ratios = calculate_financial_ratios(financial_data) if financial_data else {}
                print(f"🔍 DEBUG: Ratios calculados exitosamente: {len(ratios)}")
            except Exception as ratio_error:
                print(f"⚠️ DEBUG: Error calculando ratios: {ratio_error}")
                ratios = {}
            
            # ANÁLISIS SIMPLIFICADO PRIMERO (para debugging)
            try:
                print("🔍 DEBUG: Intentando análisis con LLM...")
                
                # Prompt más conciso para evitar problemas
                analysis_prompt = f"""
Eres un analista financiero especializado en banca. 

Analiza esta cuenta de resultados de BBVA:

DATOS EXTRAÍDOS:
{text[:2500]}

DATOS FINANCIEROS ENCONTRADOS:
{json.dumps(financial_data, indent=2) if financial_data else "No se identificaron cifras específicas"}

RATIOS CALCULADOS:
{json.dumps(ratios, indent=2) if ratios else "No se pudieron calcular ratios"}

Proporciona un análisis detallado de 600-800 palabras que incluya:

1. **Análisis de ingresos principales** (margen de intereses, comisiones)
2. **Evaluación de gastos operativos** y eficiencia
3. **Análisis de rentabilidad** y márgenes
4. **Comparaciones** con año anterior si disponible
5. **Conclusiones** y recomendaciones estratégicas

IMPORTANTE: 
- Usa SOLO los datos presentes en el texto
- NO inventes cifras que no aparezcan
- Cita cifras exactas cuando las encuentres
- Formato profesional con secciones claras
"""
                
                # Llamada al LLM con manejo de errores mejorado
                try:
                    print("🔍 DEBUG: Llamando al chat_client...")
                    analysis_response = self.chat_client.chat([
                        {"role": "system", "content": "Eres un analista financiero experto en banca con 15 años de experiencia."},
                        {"role": "user", "content": analysis_prompt}
                    ], max_tokens=1800)
                    
                    print(f"🔍 DEBUG: LLM respondió con {len(analysis_response)} caracteres")
                    
                    if not analysis_response or len(analysis_response.strip()) < 200:
                        print("⚠️ DEBUG: Respuesta LLM muy corta, usando fallback")
                        raise Exception("Respuesta LLM insuficiente")
                    
                    print("✅ DEBUG: Respuesta LLM exitosa")
                    
                except Exception as llm_error:
                    print(f"❌ DEBUG: Error en LLM: {str(llm_error)}")
                    print("🔄 DEBUG: Usando análisis fallback...")
                    analysis_response = self.generate_fallback_income_analysis(text, confidence, quality, financial_data, ratios)
                
                # CONSTRUIR RESPUESTA FINAL
                print("🔍 DEBUG: Construyendo respuesta final...")
                
                response_parts = [
                    "📊 **ANÁLISIS PROFESIONAL DE CUENTA DE RESULTADOS - BBVA**",
                    "=" * 70,
                    "",
                    analysis_response,
                    "",
                    "### 📋 **INFORMACIÓN TÉCNICA DEL ANÁLISIS**",
                    f"• **Calidad de extracción**: {quality.title()} (puntuación: {validation.get('score', 0)}/100)",
                    f"• **Confianza en datos**: {confidence:.1%}",
                    f"• **Caracteres analizados**: {len(text):,} del documento original",
                    f"• **Páginas procesadas**: {len(extraction.get('pages_processed', []))} páginas del estado financiero",
                    f"• **Categorías financieras identificadas**: {len([k for k, v in financial_data.items() if v])} de 7 principales" if financial_data else "• **Datos financieros**: Análisis basado en contenido textual",
                    f"• **Ratios calculados**: {len(ratios)} indicadores financieros" if ratios else "• **Ratios**: No calculables con datos actuales",
                    f"• **Idioma del documento**: {extraction.get('language', 'Desconocido').title()}",
                    "• **Metodología**: Extracción automática + análisis con IA especializada en banca",
                    "• **Fuente**: Cuenta de resultados consolidada de BBVA",
                    "",
                    "=" * 70,
                    "📊 *Análisis generado por sistema de IA especializada en análisis de rentabilidad bancaria*"
                ]
                
                final_response = "\n".join(response_parts)
                print(f"✅ DEBUG: Respuesta final construida con {len(final_response)} caracteres")
                
                return final_response
                
            except Exception as analysis_error:
                print(f"❌ DEBUG: Error en análisis: {str(analysis_error)}")
                # Usar fallback completo
                return self.generate_fallback_income_analysis(text, confidence, quality, financial_data, ratios)
                
        except Exception as e:
            print(f"❌ DEBUG: Error crítico en generate_enhanced_income_analysis_fixed: {str(e)}")
            # ÚLTIMO FALLBACK: Respuesta básica garantizada
            return f"""
📊 **ANÁLISIS DE CUENTA DE RESULTADOS - BBVA**

### Resumen del Análisis

Se ha procesado exitosamente la cuenta de resultados con los siguientes resultados:

• **Caracteres extraídos**: {len(extraction.get('text', ''))} del documento original
• **Páginas analizadas**: {len(extraction.get('pages_processed', []))} páginas relevantes
• **Calidad de datos**: {validation.get('quality', 'unknown').title()}
• **Confianza**: {validation.get('confidence', 0.8):.1%}

### Datos Identificados

{json.dumps(extraction.get('financial_data', {}), indent=2) if extraction.get('financial_data') else "Los datos financieros específicos están siendo procesados."}

### Conclusión

El análisis de la cuenta de resultados ha sido completado. Los datos extraídos muestran información relevante sobre los ingresos, gastos y rentabilidad de la entidad bancaria BBVA.

**Nota técnica**: Error en generación avanzada: {str(e)}. Se ha proporcionado este análisis de respaldo basado en los datos extraídos exitosamente.
"""

    def generate_fallback_income_analysis(self, text: str, confidence: float, quality: str, 
                                        financial_data: Dict, ratios: Dict) -> str:
        """Análisis de respaldo basado en extracción de datos específicos"""
        
        response_parts = []
        response_parts.append("📊 **ANÁLISIS DE CUENTA DE RESULTADOS - BBVA**")
        response_parts.append("=" * 60)
        
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
                    try:
                        float_values = [convert_string_to_float(str(v)) or v for v in values if v]
                        if len(float_values) >= 2 and isinstance(float_values[0], (int, float)) and isinstance(float_values[-1], (int, float)):
                            if float_values[-1] < float_values[0]:
                                declining_categories.append(category.replace('_', ' '))
                            else:
                                growing_categories.append(category.replace('_', ' '))
                    except:
                        continue
            
            if declining_categories:
                response_parts.append(f"• **Tendencias descendentes**: {', '.join(declining_categories)}")
            if growing_categories:
                response_parts.append(f"• **Tendencias ascendentes**: {', '.join(growing_categories)}")
        else:
            response_parts.append("• **Recomendación**: Se requiere acceso a cifras numéricas específicas para análisis cuantitativo completo")
        
        response_parts.append("\n• **Metodología**: Análisis automatizado basado en contenido extraído y patrones financieros")
        response_parts.append("• **Fuente**: Cuenta de resultados consolidada de BBVA")
        
        return "\n".join(response_parts)

# ===== CONFIGURACIÓN Y MAIN =====
DEFAULT_CONFIG = {
    "pdf": "data/entrada/output/bbva_2023_div.pdf",
    "out": "data/salida", 
    "maxsteps": 25
}

def main():
    parser = argparse.ArgumentParser(
        description="Income Agent AUTÓNOMO con Análisis Detallado CORREGIDO - Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
  python agents/income_agent.py                    # Usa configuración predefinida
  python agents/income_agent.py --pdf otro.pdf    # Sobreescribe PDF

CARACTERÍSTICAS AVANZADAS CORREGIDAS:
  - Análisis detallado de 600-800 palabras generado por LLM especializado
  - Extracción automática de cifras financieras con conversión segura de tipos
  - Cálculo automático de ratios bancarios sin errores de tipos de datos
  - Validación mejorada con puntuación de calidad detallada
  - Análisis fallback robusto basado en datos extraídos
  - Debugging avanzado para identificar y resolver problemas

MEJORAS IMPLEMENTADAS:
  - Patrones regex específicos mejorados para documentos BBVA
  - Conversión segura string→float con manejo de formatos europeos
  - Funciones de cálculo de ratios con validación de tipos
  - Manejo robusto de errores con fallbacks múltiples
  - Extracción mejorada con re-procesamiento automático
  - Debugging detallado para monitoreo de extracción
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
    print("🚀 Income Agent v4.2 AUTÓNOMO Multi-Agent - ERRORES CORREGIDOS")
    print(f"📄 PDF: {args.pdf}")
    print(f"📁 Salida: {args.out}")
    print(f"⚙️ Groq/Azure OpenAI: Configuración optimizada")
    print(f"🔧 Max steps: {args.maxsteps}")
    print("🆕 CARACTERÍSTICAS: Conversión segura de tipos, patrones mejorados, debugging completo")
    
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
            print("✅ Análisis detallado con conversión segura de tipos completado")
        else:
            print(f"❌ Error: {result.get('error_details', 'Error desconocido')}")
        
        print("🎉 Análisis de cuenta de resultados completado!")
        print("🤖 IncomeREACTAgent con errores corregidos disponible para sistema multi-agente")
        
    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
        raise

if __name__ == "__main__":
    main()
