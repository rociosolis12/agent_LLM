"""
Cashflows Agent REACT - Versión Multi-Agente AUTÓNOMA COMPLETA
Especializado en análisis de estado de flujos de efectivo con wrapper autónomo para sistema multi-agente
CARACTERÍSTICAS: Respuestas específicas generadas por LLM, completamente autónomo, compatible con coordinador
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

# =============================
# Configuración y utilidades
# =============================

# Cargar .env desde el directorio raíz del proyecto
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)
os.chdir(project_root)

if not env_path.exists():
    print(f"Warning: Archivo .env no encontrado en {env_path}")

# ----- Azure OpenAI Configuration -----
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

# Validación de credenciales
if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
    raise ValueError("Azure OpenAI credentials required")

# =============================
# Cliente Azure OpenAI
# =============================

class AzureChatClient:
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION
        )
        self.deployment = AZURE_OPENAI_DEPLOYMENT

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 1000) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Azure OpenAI API error: {str(e)}")

# Inicialización del cliente
chat_client = AzureChatClient()

# =============================
# Diccionarios específicos para estado de flujos de efectivo
# =============================

CASHFLOW_TITLES_EN = [
    "statement of cash flows", "cash flow statement", "cash flows statement",
    "consolidated statement of cash flows", "cashflow statement"
]

CASHFLOW_TITLES_ES = [
    "estado de flujos de efectivo", "flujos de efectivo", "estado de flujo de efectivo",
    "estado consolidado de flujos de efectivo", "flujo de caja"
]

# Términos específicos de actividades operativas
OPERATING_HINTS = [
    "profit for the year", "net income", "depreciation and amortisation",
    "expected credit losses", "tax expense", "deposits from customers",
    "income taxes paid", "net cash from operating activities", "operating activities"
]

# Términos específicos de actividades de inversión
INVESTING_HINTS = [
    "acquisitions in investment portfolio", "proceeds from investment portfolio",
    "purchase of tangible", "purchase of intangible", "net cash from investing activities",
    "investing activities", "acquisition", "disposal"
]

# Términos específicos de actividades de financiación
FINANCING_HINTS = [
    "dividends paid", "borrowings", "repayment of borrowings",
    "net cash from financing activities", "financing activities",
    "share capital", "loan proceeds"
]

# Términos específicos de efectivo y equivalentes
CASH_HINTS = [
    "net increase in cash", "cash and cash equivalents at beginning",
    "cash and cash equivalents at end", "net change in cash",
    "beginning of period", "end of period"
]

CURRENCY_HINTS = [
    "thousands of euros", "€", "euro", "euros", "eur",
    "miles de euros", "thousand", "thousands"
]

# =============================
# Funciones auxiliares
# =============================

def normalize_text(s: str) -> str:
    s = s or ""
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

def detect_language(text: str) -> str:
    t = normalize_text(text)
    score_es = sum(1 for w in ["efectivo", "flujos", "operativas"] if w in t)
    score_en = sum(1 for w in ["cash", "flows", "operating"] if w in t)
    return "es" if score_es >= score_en else "en"

# =============================
# CLASE WRAPPER AUTÓNOMA PARA SISTEMA MULTI-AGENTE - CASHFLOWS
# =============================

class CashFlowsREACTAgent:
    """
    Wrapper REACT COMPLETAMENTE AUTÓNOMO para el Cashflows Agent
    
    Esta clase es completamente autónoma y genera respuestas específicas usando LLM
    basándose en los datos de flujos de efectivo que extrae.
    """
    
    def __init__(self):
        self.agent_type = "cashflows"
        self.max_steps = 25  # Aumentado para el wrapper
        self.chat_client = chat_client  # Cliente Azure OpenAI

    def run_final_financial_extraction_agent(self, pdf_path: str, question: str = None) -> Dict[str, Any]:
        """
        Ejecuta la extracción de flujos de efectivo Y genera respuesta específica autónomamente
        
        Args:
            pdf_path: Ruta al PDF a procesar
            question: Pregunta específica del usuario (opcional)
            
        Returns:
            Dict con el resultado y respuesta específica generada por LLM
        """
        try:
            print(f" CashFlowsREACTAgent AUTÓNOMO iniciando extracción para: {pdf_path}")
            
            pdf_file = Path(pdf_path)
            output_dir = Path("data/salida")
            
            # 1. EJECUTAR EXTRACCIÓN CORE (funcionalidad de cashflows)
            result = self._run_core_extraction(pdf_file, output_dir)
            
            # 2. VERIFICAR ÉXITO DE EXTRACCIÓN
            extraction_successful = result.get("success", False)
            
            if not extraction_successful:
                print(" Cashflows extraction failed")
                return {
                    "status": "error", 
                    "steps_taken": result.get("steps_taken", 0),
                    "session_id": f"cashflows_{pdf_file.stem}",
                    "final_response": "Cashflows extraction failed - check logs for details",
                    "agent_type": "cashflows",
                    "error_details": result.get("error", "Extraction process failed"),
                    "specific_answer": "No se pudo completar la extracción de los flujos de efectivo."
                }
            
            # 3. GENERAR RESPUESTA ESPECÍFICA USANDO LLM
            specific_answer = self._generate_llm_response(question, pdf_file, result)
            
            print("✅ Cashflows extraction completed successfully (AUTÓNOMO)")
            return {
                "status": "task_completed",
                "steps_taken": result.get("steps_taken", 5),
                "session_id": f"cashflows_{pdf_file.stem}",
                "final_response": "Cashflows extraction completed successfully - AUTONOMOUS VERSION",
                "agent_type": "cashflows",
                "files_generated": result.get("files_created", 0),
                "specific_answer": specific_answer  # ← RESPUESTA GENERADA POR LLM
            }
                
        except Exception as e:
            print(f" Error en CashFlowsREACTAgent: {str(e)}")
            return {
                "status": "error",
                "steps_taken": 0,
                "session_id": "cashflows_error",
                "final_response": f"Error in cashflows extraction: {str(e)}",
                "agent_type": "cashflows",
                "error_details": str(e),
                "specific_answer": f"Error durante la extracción de los flujos de efectivo: {str(e)}"
            }

    def _run_core_extraction(self, pdf_file: Path, output_dir: Path) -> Dict[str, Any]:
        """
        Ejecuta la extracción core de flujos de efectivo
        """
        try:
            print(f" Extrayendo flujos de efectivo de: {pdf_file}")
            
            # 1. EXTRAER TEXTO DE PÁGINAS RELEVANTES
            extracted_text = self._extract_cashflow_text(pdf_file)
            
            if not extracted_text:
                return {
                    "success": False,
                    "error": "No se encontraron datos de flujos de efectivo",
                    "steps_taken": 1
                }
            
            # 2. PARSEAR DATOS FINANCIEROS
            cashflow_data = self._parse_cashflow_data(extracted_text)
            
            # 3. GUARDAR RESULTADOS
            files_created = self._save_extraction_results(pdf_file, output_dir, extracted_text, cashflow_data)
            
            return {
                "success": True,
                "extracted_text": extracted_text,
                "cashflow_data": cashflow_data,
                "files_created": files_created,
                "steps_taken": 5
            }
            
        except Exception as e:
            print(f" Error en extracción core: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "steps_taken": 0
            }

    def _extract_cashflow_text(self, pdf_file: Path) -> str:
        """
        Extrae texto de páginas que contienen estado de flujos de efectivo
        """
        try:
            doc = fitz.open(str(pdf_file))
            cashflow_text = ""
            pages_processed = []
            
            # Patrones para identificar estado de flujos de efectivo
            cashflow_patterns = [
                r"statement\s+of\s+cash\s+flows",
                r"cash\s+flows?\s+statement",
                r"estado\s+de\s+flujos?\s+de\s+efectivo",
                r"flujos?\s+de\s+efectivo",
                r"cash\s+flows?\s+from",
                r"consolidated.*cash.*flows"
            ]
            
            # Buscar en las primeras 20 páginas
            for page_num in range(min(len(doc), 20)):
                page = doc[page_num]
                text = page.get_text()
                text_lower = text.lower()
                
                # Verificar si la página contiene flujos de efectivo
                for pattern in cashflow_patterns:
                    if re.search(pattern, text_lower, re.IGNORECASE):
                        cashflow_text += f"=== PÁGINA {page_num + 1} ===\n{text}\n\n"
                        pages_processed.append(page_num + 1)
                        print(f" Página {page_num + 1}: {len(text)} caracteres extraídos")
                        break
            
            doc.close()
            
            if cashflow_text:
                print(f" Texto total extraído: {len(cashflow_text)} caracteres de {len(pages_processed)} páginas")
            
            return cashflow_text
            
        except Exception as e:
            print(f" Error extrayendo texto: {e}")
            return ""

    def _parse_cashflow_data(self, text: str) -> List[Dict[str, Any]]:
        """
        Parsea el texto extraído para identificar datos financieros de flujos de efectivo
        """
        lines = text.split('\n')
        cashflow_data = []
        
        # Patrones específicos para GarantiBank flujos de efectivo
        cashflow_patterns = [
            # ACTIVIDADES OPERATIVAS
            (r'profit\s+for\s+the\s+year\s+.*?(\d{1,3}(?:,\d{3})+)\s+.*?(\d{1,3}(?:,\d{3})+)', 'operating', 'Profit for the year'),
            (r'depreciation\s+and\s+amortisation\s+.*?(\d{1,3}(?:,\d{3})+)\s+.*?(\d{1,3}(?:,\d{3})+)', 'operating', 'Depreciation and amortisation'),
            (r'expected\s+credit\s+losses\s+.*?(\d{1,3}(?:,\d{3})+)\s+.*?\((\d{1,3}(?:,\d{3})+)\)', 'operating', 'Expected credit losses'),
            (r'tax\s+expense\s+.*?(\d{1,3}(?:,\d{3})+)\s+.*?(\d{1,3}(?:,\d{3})+)', 'operating', 'Tax expense'),
            (r'deposits\s+from\s+customers\s+.*?(\d{1,3}(?:,\d{3})+)\s+.*?(\d{1,3}(?:,\d{3})+)', 'operating', 'Deposits from customers'),
            (r'income\s+taxes\s+paid\s+.*?\((\d{1,3}(?:,\d{3})+)\)\s+.*?\((\d{1,3}(?:,\d{3})+)\)', 'operating', 'Income taxes paid'),
            (r'net\s+cash\s+from.*?operating\s+activities\s+.*?(\d{1,3}(?:,\d{3})+)\s+.*?(\d{1,3}(?:,\d{3})+)', 'operating', 'Net cash from operating activities'),
            
            # ACTIVIDADES DE INVERSIÓN
            (r'acquisitions\s+in\s+investment\s+portfolio\s+.*?\((\d{1,3}(?:,\d{3})+)\)\s+.*?\((\d{1,3}(?:,\d{3})+)\)', 'investing', 'Acquisitions in investment portfolio'),
            (r'proceeds\s+from\s+investment\s+portfolio\s+.*?(\d{1,3}(?:,\d{3})+)\s+.*?(\d{1,3}(?:,\d{3})+)', 'investing', 'Proceeds from investment portfolio'),
            (r'purchase\s+of\s+tangible.*?assets\s+.*?\((\d{1,3}(?:,\d{3})+)\)\s+.*?\((\d{1,3}(?:,\d{3})+)\)', 'investing', 'Purchase of tangible and intangible assets'),
            (r'net\s+cash\s+from.*?investing\s+activities\s+.*?\((\d{1,3}(?:,\d{3})+)\)\s+.*?(\d{1,3}(?:,\d{3})+)', 'investing', 'Net cash from investing activities'),
            
            # EFECTIVO Y EQUIVALENTES
            (r'net\s+increase.*?cash.*?equivalents\s+.*?(\d{1,3}(?:,\d{3})+)\s+.*?(\d{1,3}(?:,\d{3})+)', 'summary', 'Net increase in cash and cash equivalents'),
            (r'cash.*?equivalents.*?beginning.*?period\s+.*?(\d{1,3}(?:,\d{3})+)\s+.*?(\d{1,3}(?:,\d{3})+)', 'summary', 'Cash and cash equivalents at beginning of period'),
            (r'cash.*?equivalents.*?end.*?period\s+.*?(\d{1,3}(?:,\d{3})+)\s+.*?(\d{1,3}(?:,\d{3})+)', 'summary', 'Cash and cash equivalents at end of period'),
            
            # INFORMACIÓN ADICIONAL
            (r'interest\s+received\s+.*?(\d{1,3}(?:,\d{3})+)\s+.*?(\d{1,3}(?:,\d{3})+)', 'additional', 'Interest received'),
            (r'interest\s+paid\s+.*?(\d{1,3}(?:,\d{3})+)\s+.*?(\d{1,3}(?:,\d{3})+)', 'additional', 'Interest paid'),
        ]
        
        full_text = ' '.join(lines).lower()
        
        for pattern, section, concept_name in cashflow_patterns:
            matches = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)
            if matches:
                try:
                    amount_2023 = float(matches.group(1).replace(',', ''))
                    amount_2022 = float(matches.group(2).replace(',', ''))
                    
                    # Manejar números negativos (gastos y salidas de efectivo)
                    if section == 'investing' and 'acquisition' in concept_name.lower():
                        amount_2023 = -abs(amount_2023)
                        amount_2022 = -abs(amount_2022)
                    elif 'paid' in concept_name.lower() or 'purchase' in concept_name.lower():
                        amount_2023 = -abs(amount_2023)
                        amount_2022 = -abs(amount_2022)
                    elif section == 'investing' and pattern.count(r'\(') > 0:
                        if 'proceeds' not in concept_name.lower():
                            amount_2023 = -abs(amount_2023) if '(' in matches.group(0) else amount_2023
                            amount_2022 = -abs(amount_2022) if '(' in matches.group(0) else amount_2022
                    
                    entry = {
                        'concept': concept_name,
                        'section': section,
                        '2023': amount_2023,
                        '2022': amount_2022
                    }
                    
                    cashflow_data.append(entry)
                    print(f" Extraído: {concept_name} -> {section} -> [{amount_2023:,}, {amount_2022:,}]")
                    
                except (ValueError, IndexError) as e:
                    print(f" Error procesando {concept_name}: {e}")
                    continue
        
        # Si no encontramos nada con patrones específicos, usar extracción más agresiva
        if not cashflow_data:
            print(" Usando extracción agresiva...")
            cashflow_data = self._aggressive_parsing(lines)
        
        print(f" Total extraído: {len(cashflow_data)} entradas")
        return cashflow_data

    def _aggressive_parsing(self, lines: List[str]) -> List[Dict[str, Any]]:
        """
        Extracción más agresiva cuando los patrones específicos fallan
        """
        cashflow_data = []
        
        for line in lines:
            line = line.strip()
            if len(line) < 15:
                continue
            
            # Buscar líneas con dos números grandes (columnas de años)
            number_matches = re.findall(r'\(?\d{1,3}(?:,\d{3})+\)?', line)
            if len(number_matches) >= 2:
                try:
                    # Limpiar y convertir números
                    amounts = []
                    for match in number_matches[:2]:
                        clean_num = re.sub(r'[(),]', '', match)
                        is_negative = '(' in match
                        amount = float(clean_num)
                        if is_negative:
                            amount = -amount
                        amounts.append(amount)
                    
                    # Filtrar números muy pequeños o años
                    if any(abs(amt) < 1000 for amt in amounts):
                        continue
                    if any(str(int(abs(amt))) in ['2022', '2023'] for amt in amounts):
                        continue
                    
                    # Limpiar concepto
                    concept = re.sub(r'\(?\d{1,3}(?:,\d{3})+\)?', '', line).strip()
                    concept = re.sub(r'\s+', ' ', concept)
                    
                    if len(concept) > 8:
                        # Determinar sección basada en el concepto
                        concept_lower = concept.lower()
                        if any(word in concept_lower for word in ['profit', 'depreciation', 'tax', 'operating']):
                            section = 'operating'
                        elif any(word in concept_lower for word in ['investment', 'purchase', 'acquisition', 'investing']):
                            section = 'investing'
                        elif any(word in concept_lower for word in ['financing', 'dividend', 'loan']):
                            section = 'financing'
                        elif any(word in concept_lower for word in ['cash', 'beginning', 'end']):
                            section = 'summary'
                        else:
                            section = 'other'
                        
                        entry = {
                            'concept': concept,
                            'section': section,
                            '2023': amounts[0],
                            '2022': amounts[1]
                        }
                        
                        cashflow_data.append(entry)
                        
                except (ValueError, IndexError):
                    continue
        
        return cashflow_data

    def _save_extraction_results(self, pdf_file: Path, output_dir: Path, 
                                extracted_text: str, cashflow_data: List[Dict]) -> int:
        """
        Guarda los resultados de la extracción
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            base = pdf_file.stem
            files_created = 0
            
            # 1. Guardar datos financieros como JSON
            if cashflow_data:
                json_path = output_dir / f"{base}_cashflow_data.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(cashflow_data, f, ensure_ascii=False, indent=2)
                files_created += 1
            
            # 2. Guardar datos como CSV
            if cashflow_data:
                try:
                    df = pd.DataFrame(cashflow_data)
                    csv_path = output_dir / f"{base}_cashflow_data.csv"
                    df.to_csv(csv_path, index=False, encoding='utf-8')
                    files_created += 1
                except Exception as e:
                    print(f" Error generando CSV: {e}")
            
            # 3. Guardar texto extraído
            text_path = output_dir / f"{base}_cashflow_text.txt"
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            files_created += 1
            
            print(f" Archivos guardados: {files_created}")
            return files_created
            
        except Exception as e:
            print(f" Error guardando resultados: {e}")
            return 0

    def _generate_llm_response(self, question: str, pdf_file: Path, extraction_result: Dict) -> str:
        """
        GENERA RESPUESTA ESPECÍFICA USANDO LLM - COMPLETAMENTE AUTÓNOMA
        
        Esta función lee los datos extraídos y usa el LLM para generar
        una respuesta específica a la pregunta del usuario sobre flujos de efectivo.
        """
        try:
            # 1. OBTENER DATOS EXTRAÍDOS
            extracted_text = extraction_result.get("extracted_text", "")
            cashflow_data = extraction_result.get("cashflow_data", [])
            
            # 2. EXTRAER NÚMEROS FINANCIEROS ESPECÍFICOS
            financial_numbers = self._extract_financial_numbers(extracted_text, cashflow_data)
            
            # 3. SI NO HAY PREGUNTA ESPECÍFICA, DAR RESPUESTA GENERAL
            if not question:
                return self._generate_general_summary(extracted_text, financial_numbers)
            
            # 4. USAR LLM PARA RESPUESTA ESPECÍFICA A LA PREGUNTA
            if self.chat_client:
                return self._ask_llm_specific_question(question, extracted_text, financial_numbers)
            else:
                # Fallback sin LLM
                return self._generate_rule_based_response(question, extracted_text, financial_numbers)
                
        except Exception as e:
            print(f" Error generando respuesta LLM: {str(e)}")
            return f"He completado la extracción de los flujos de efectivo exitosamente, pero hubo un error al generar la respuesta específica: {str(e)}"

    def _extract_financial_numbers(self, text: str, cashflow_data: List[Dict]) -> Dict[str, str]:
        """
        Extrae números financieros clave del texto y datos parseados
        """
        financial_numbers = {}
        
        # Usar datos parseados primero
        if cashflow_data:
            for item in cashflow_data:
                concept = item.get('concept', '').lower()
                section = item.get('section', '')
                
                if 'net cash from operating activities' in concept:
                    financial_numbers['operating_cash_2023'] = str(item.get('2023', ''))
                    financial_numbers['operating_cash_2022'] = str(item.get('2022', ''))
                elif 'net cash from investing activities' in concept:
                    financial_numbers['investing_cash_2023'] = str(item.get('2023', ''))
                    financial_numbers['investing_cash_2022'] = str(item.get('2022', ''))
                elif 'cash and cash equivalents at end' in concept:
                    financial_numbers['cash_end_2023'] = str(item.get('2023', ''))
                    financial_numbers['cash_end_2022'] = str(item.get('2022', ''))
                elif 'cash and cash equivalents at beginning' in concept:
                    financial_numbers['cash_beginning_2023'] = str(item.get('2023', ''))
                    financial_numbers['cash_beginning_2022'] = str(item.get('2022', ''))
                elif 'net increase' in concept and 'cash' in concept:
                    financial_numbers['net_cash_increase_2023'] = str(item.get('2023', ''))
                    financial_numbers['net_cash_increase_2022'] = str(item.get('2022', ''))
        
        return financial_numbers

    def _generate_general_summary(self, extracted_text: str, financial_numbers: Dict) -> str:
        """
        Genera resumen general sin pregunta específica
        """
        if financial_numbers:
            summary_parts = ["💸 **RESUMEN DE FLUJOS DE EFECTIVO EXTRAÍDOS**\n"]
            
            if 'operating_cash_2023' in financial_numbers:
                summary_parts.append(f"• **Flujo Operativo 2023**: €{financial_numbers['operating_cash_2023']} miles")
            
            if 'investing_cash_2023' in financial_numbers:
                summary_parts.append(f"• **Flujo de Inversión 2023**: €{financial_numbers['investing_cash_2023']} miles")
            
            if 'cash_end_2023' in financial_numbers:
                summary_parts.append(f"• **Efectivo Final 2023**: €{financial_numbers['cash_end_2023']} miles")
            
            summary_parts.append("\n**Fuente**: Estado de flujos de efectivo consolidado de GarantiBank International N.V.")
            
            return "\n".join(summary_parts)
        
        return " He extraído exitosamente el estado de flujos de efectivo consolidado. Los datos de actividades operativas, de inversión y de financiación están disponibles en los archivos generados para análisis detallado."

    def _ask_llm_specific_question(self, question: str, extracted_text: str, financial_numbers: Dict) -> str:
        """
        USA EL LLM PARA RESPONDER PREGUNTA ESPECÍFICA - FUNCIONALIDAD CLAVE
        """
        try:
            # Preparar contexto financiero para el LLM
            financial_context = ""
            if financial_numbers:
                financial_context = "DATOS FINANCIEROS IDENTIFICADOS:\n"
                for key, value in financial_numbers.items():
                    financial_context += f"- {key.replace('_', ' ').title()}: €{value} miles\n"
            
            # PROMPT ENGINEERING ESPECIALIZADO PARA FLUJOS DE EFECTIVO
            analysis_prompt = f"""Eres un analista financiero experto especializado en análisis de flujos de efectivo corporativos.

CONTEXTO:
Has extraído información del estado de flujos de efectivo consolidado de GarantiBank International N.V. para el año 2023.

{financial_context}

TEXTO EXTRAÍDO DEL ESTADO DE FLUJOS DE EFECTIVO:
{extracted_text[:2000]}

PREGUNTA DEL USUARIO:
{question}

INSTRUCCIONES:
1. Analiza la información financiera de los flujos de efectivo
2. Responde la pregunta de forma específica y profesional
3. Incluye cifras exactas cuando estén disponibles
4. Proporciona contexto relevante sobre la liquidez de GarantiBank
5. Si no tienes datos exactos, indica qué información está disponible
6. Mantén un tono profesional y conciso

FORMATO DE RESPUESTA:
- Respuesta directa y específica sobre flujos de efectivo
- Cifras con formato apropiado (€X,XXX miles)
- Análisis de liquidez cuando sea relevante
- Fuente: Estado de flujos de efectivo consolidado

RESPUESTA PROFESIONAL:"""

            # Llamar al LLM
            messages = [
                {
                    "role": "system", 
                    "content": "Eres un analista financiero experto especializado en análisis de flujos de efectivo de instituciones financieras."
                },
                {
                    "role": "user", 
                    "content": analysis_prompt
                }
            ]
            
            # Usar cliente Azure OpenAI existente
            response = self.chat_client.chat(messages, max_tokens=1000)
            
            return response.strip()
            
        except Exception as e:
            print(f" Error en LLM: {str(e)}")
            # Fallback a respuesta basada en reglas
            return self._generate_rule_based_response(question, extracted_text, financial_numbers)

    def _generate_rule_based_response(self, question: str, extracted_text: str, financial_numbers: Dict) -> str:
        """
        Fallback: Genera respuesta basada en reglas si el LLM no está disponible
        """
        question_lower = question.lower()
        
        # Detectar tipo de pregunta y responder con datos disponibles
        if any(word in question_lower for word in ['operativo', 'operativas', 'operating']):
            if 'operating_cash_2023' in financial_numbers:
                return f" **El flujo de efectivo de actividades operativas de GarantiBank International N.V. en 2023 fue €{financial_numbers['operating_cash_2023']} miles** según el estado de flujos de efectivo consolidado.\n\n**Fuente**: Statement of Cash Flows extraído"
            
        elif any(word in question_lower for word in ['inversión', 'investing', 'inversion']):
            if 'investing_cash_2023' in financial_numbers:
                return f" **El flujo de efectivo de actividades de inversión en 2023 fue €{financial_numbers['investing_cash_2023']} miles**, reflejando las adquisiciones y ventas de inversiones.\n\n**Fuente**: Estado de flujos de efectivo consolidado extraído"
                
        elif any(word in question_lower for word in ['efectivo', 'cash', 'liquidez']):
            if 'cash_end_2023' in financial_numbers:
                return f" **El efectivo y equivalentes al final de 2023 fue €{financial_numbers['cash_end_2023']} miles**, mostrando la posición de liquidez de la entidad.\n\n**Fuente**: Estado de flujos de efectivo consolidado extraído"
        
        # Respuesta genérica si no puede determinar específicamente
        return " He extraído exitosamente el estado de flujos de efectivo consolidado de GarantiBank International N.V. Los datos incluyen flujos de actividades operativas, de inversión y de financiación para 2023. Los datos detallados están disponibles en los archivos generados."

# =============================
# Sistema de herramientas básico (si se necesita para REACT compatibilidad)
# =============================

def run_cashflow_agent(pdf_path: Path, output_dir: Path, max_steps: int = 10) -> Dict[str, Any]:
    """
    Función principal para ejecutar el agente de cashflows (compatibilidad)
    """
    agent = CashFlowsREACTAgent()
    
    try:
        result = agent._run_core_extraction(pdf_path, output_dir)
        return {
            "history": [],
            "context": {"last_extraction": result},
            "steps_completed": 5,
            "finished": result.get("success", False)
        }
    except Exception as e:
        return {
            "history": [],
            "context": {},
            "steps_completed": 0,
            "finished": False,
            "error": str(e)
        }

# =============================
# CLI principal
# =============================

def main():
    # ===== CONFIGURACIÓN PREDEFINIDA =====
    DEFAULT_CONFIG = {
        "pdf": "data/entrada/output/bbva_2023_div.pdf",
        "out": "data/salida",
        "max_steps": 10
    }

    parser = argparse.ArgumentParser(
        description="Cashflows Agent AUTÓNOMO especializado en Estado de Flujos de Efectivo - Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
  python agents/cashflows_agent.py                    # Usa configuración predefinida
  python agents/cashflows_agent.py --pdf otro.pdf    # Sobreescribe PDF
  
CARACTERÍSTICAS AUTÓNOMAS:
  - Generación de respuestas específicas usando LLM
  - Prompt engineering especializado para flujos de efectivo
  - Fallback robusto con respuestas basadas en reglas
  - Extracción automática de flujos operativos, de inversión y financiación
  
Sistema Multi-Agente:
  Esta versión incluye CashFlowsREACTAgent AUTÓNOMO para integración con main_system.py
        """
    )

    # ===== ARGUMENTOS OPCIONALES =====
    parser.add_argument("--pdf", 
                       default=DEFAULT_CONFIG["pdf"], 
                       help=f"Ruta al PDF (por defecto: {DEFAULT_CONFIG['pdf']})")
    
    parser.add_argument("--out", 
                       default=DEFAULT_CONFIG["out"], 
                       help=f"Directorio de salida (por defecto: {DEFAULT_CONFIG['out']})")
    
    parser.add_argument("--max_steps", 
                       type=int, 
                       default=DEFAULT_CONFIG["max_steps"], 
                       help=f"Máximo pasos REACT (por defecto: {DEFAULT_CONFIG['max_steps']})")
    
    parser.add_argument("--question", 
                       type=str, 
                       default=None, 
                       help="Pregunta específica sobre flujos de efectivo")

    args = parser.parse_args()

    # ===== CONFIGURAR RUTAS =====
    pdf_path = Path(args.pdf)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===== VERIFICAR PDF =====
    if not pdf_path.exists():
        print(f" Error: PDF no encontrado en {pdf_path}")
        return

    # ===== MOSTRAR CONFIGURACIÓN =====
    print(f" Cashflows Agent v3.0 AUTÓNOMO Multi-Agent - Configuración Automática")
    print(f" PDF: {pdf_path}")
    print(f" Salida: {output_dir}")
    print(f" Azure OpenAI: {AZURE_OPENAI_DEPLOYMENT}")
    print(f" Max steps: {args.max_steps}")
    print(f" Multi-Agent: CashFlowsREACTAgent AUTÓNOMO class available")
    print(" CARACTERÍSTICAS: Respuestas LLM específicas, prompt engineering avanzado, completamente autónomo")

    try:
        # Crear agente y ejecutar
        agent = CashFlowsREACTAgent()
        
        if args.question:
            # Modo pregunta específica
            print(f" Pregunta específica: {args.question}")
            result = agent.run_final_financial_extraction_agent(str(pdf_path), args.question)
        else:
            # Modo extracción general
            result = agent.run_final_financial_extraction_agent(str(pdf_path))

        print("\n ==== RESUMEN DE EJECUCIÓN AUTÓNOMO ====")
        print(f"Estado: {' EXITOSO' if result.get('status') == 'task_completed' else ' ERROR'}")
        print(f"Pasos completados: {result.get('steps_taken', 0)}")
        print(f"Archivos generados: {result.get('files_generated', 0)}")

        if result.get("status") == "task_completed":
            print("\n ==== RESPUESTA ESPECÍFICA ====")
            print(result.get("specific_answer", "No hay respuesta específica disponible"))
        else:
            print(f"\n Error: {result.get('error_details', 'Error desconocido')}")

        print("\n Análisis de flujos de efectivo completado!")
        print(" Clase CashFlowsREACTAgent AUTÓNOMA disponible para sistema multi-agente")
        print(" Versión autónoma con generación de respuestas específicas usando LLM")

    except Exception as e:
        print(f" Error durante la ejecución: {e}")
        raise

if __name__ == "__main__":
    main()
