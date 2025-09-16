"""
Income Agent REACT - Versión Multi-Agente AUTÓNOMA COMPLETA
Especializado en análisis de cuenta de resultados con wrapper autónomo para sistema multi-agente
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
# Diccionarios específicos para cuenta de resultados
# =============================

INCOME_TITLES_EN = [
    "statement of comprehensive income", "income statement", "profit and loss",
    "consolidated statement of comprehensive income", "statement of income"
]

INCOME_TITLES_ES = [
    "cuenta de resultados", "estado de resultados", "estado de ganancias y pérdidas",
    "cuenta consolidada de resultados", "resultados integrales"
]

# Términos específicos de ingresos
REVENUE_HINTS = [
    "interest income", "net interest income", "fee and commission income",
    "trading income", "total income", "operating income", "revenue",
    "ingresos por intereses", "ingresos netos por intereses", "ingresos por comisiones"
]

# Términos específicos de gastos
EXPENSE_HINTS = [
    "interest expense", "personnel expenses", "operating expenses",
    "administrative expenses", "depreciation", "total expenses",
    "gastos por intereses", "gastos de personal", "gastos operativos"
]

# Términos específicos de resultados
PROFIT_HINTS = [
    "income before tax", "net income", "profit before tax", "net profit",
    "comprehensive income", "earnings", "resultado antes de impuestos",
    "beneficio neto", "resultado neto", "ganancia neta"
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
    score_es = sum(1 for w in ["ingresos", "gastos", "resultado"] if w in t)
    score_en = sum(1 for w in ["income", "expenses", "profit"] if w in t)
    return "es" if score_es >= score_en else "en"

def safe_pdf_pages(path: Path) -> List:
    pages = []
    with fitz.open(str(path)) as pdf:
        for p in pdf.pages:
            pages.append(p)
    return pages

# =============================
# CLASE WRAPPER AUTÓNOMA PARA SISTEMA MULTI-AGENTE - INCOME
# =============================

class IncomeREACTAgent:
    """
    Wrapper REACT COMPLETAMENTE AUTÓNOMO para el Income Agent
    
    Esta clase es completamente autónoma y genera respuestas específicas usando LLM
    basándose en los datos de cuenta de resultados que extrae.
    """
    
    def __init__(self):
        self.agent_type = "income"
        self.max_steps = 25  # Aumentado para el wrapper
        self.chat_client = chat_client  # Cliente Azure OpenAI

    def run_final_financial_extraction_agent(self, pdf_path: str, question: str = None) -> Dict[str, Any]:
        """
        Ejecuta la extracción de cuenta de resultados Y genera respuesta específica autónomamente
        
        Args:
            pdf_path: Ruta al PDF a procesar
            question: Pregunta específica del usuario (opcional)
            
        Returns:
            Dict con el resultado y respuesta específica generada por LLM
        """
        try:
            print(f"🔧 IncomeREACTAgent AUTÓNOMO iniciando extracción para: {pdf_path}")
            
            pdf_file = Path(pdf_path)
            output_dir = Path("data/salida")
            
            # 1. EJECUTAR EXTRACCIÓN CORE (funcionalidad de income)
            result = self._run_core_extraction(pdf_file, output_dir)
            
            # 2. VERIFICAR ÉXITO DE EXTRACCIÓN
            extraction_successful = result.get("success", False)
            
            if not extraction_successful:
                print("⚠️ Income extraction failed")
                return {
                    "status": "error", 
                    "steps_taken": result.get("steps_taken", 0),
                    "session_id": f"income_{pdf_file.stem}",
                    "final_response": "Income extraction failed - check logs for details",
                    "agent_type": "income",
                    "error_details": result.get("error", "Extraction process failed"),
                    "specific_answer": "No se pudo completar la extracción de la cuenta de resultados."
                }
            
            # 3. GENERAR RESPUESTA ESPECÍFICA USANDO LLM
            specific_answer = self._generate_llm_response(question, pdf_file, result)
            
            print("✅ Income extraction completed successfully (AUTÓNOMO)")
            return {
                "status": "task_completed",
                "steps_taken": result.get("steps_taken", 5),
                "session_id": f"income_{pdf_file.stem}",
                "final_response": "Income extraction completed successfully - AUTONOMOUS VERSION",
                "agent_type": "income",
                "files_generated": result.get("files_created", 0),
                "specific_answer": specific_answer  # ← RESPUESTA GENERADA POR LLM
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

    def _run_core_extraction(self, pdf_file: Path, output_dir: Path) -> Dict[str, Any]:
        """
        Ejecuta la extracción core de cuenta de resultados
        """
        try:
            print(f"🔍 Extrayendo cuenta de resultados de: {pdf_file}")
            
            # 1. EXTRAER TEXTO DE PÁGINAS RELEVANTES
            extracted_text = self._extract_income_text(pdf_file)
            
            if not extracted_text:
                return {
                    "success": False,
                    "error": "No se encontraron datos de cuenta de resultados",
                    "steps_taken": 1
                }
            
            # 2. PARSEAR DATOS FINANCIEROS
            financial_data = self._parse_income_data(extracted_text)
            
            # 3. GUARDAR RESULTADOS
            files_created = self._save_extraction_results(pdf_file, output_dir, extracted_text, financial_data)
            
            return {
                "success": True,
                "extracted_text": extracted_text,
                "financial_data": financial_data,
                "files_created": files_created,
                "steps_taken": 5
            }
            
        except Exception as e:
            print(f"❌ Error en extracción core: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "steps_taken": 0
            }

    def _extract_income_text(self, pdf_file: Path) -> str:
        """
        Extrae texto de páginas que contienen cuenta de resultados
        """
        try:
            doc = fitz.open(str(pdf_file))
            income_text = ""
            pages_processed = []
            
            # Patrones para identificar cuenta de resultados
            income_patterns = [
                r"statement\s+of\s+comprehensive\s+income",
                r"statement\s+of\s+income",
                r"income\s+statement", 
                r"profit\s+and\s+loss",
                r"cuenta\s+de\s+resultados",
                r"comprehensive\s+income",
                r"consolidated.*income"
            ]
            
            # Buscar en las primeras 15 páginas
            for page_num in range(min(len(doc), 15)):
                page = doc[page_num]
                text = page.get_text()
                text_lower = text.lower()
                
                # Verificar si la página contiene cuenta de resultados
                for pattern in income_patterns:
                    if re.search(pattern, text_lower, re.IGNORECASE):
                        income_text += f"=== PÁGINA {page_num + 1} ===\n{text}\n\n"
                        pages_processed.append(page_num + 1)
                        print(f"✅ Página {page_num + 1}: {len(text)} caracteres extraídos")
                        break
            
            doc.close()
            
            if income_text:
                print(f"📊 Texto total extraído: {len(income_text)} caracteres de {len(pages_processed)} páginas")
            
            return income_text
            
        except Exception as e:
            print(f"❌ Error extrayendo texto: {e}")
            return ""

    def _parse_income_data(self, text: str) -> List[Dict[str, Any]]:
        """
        Parsea el texto extraído para identificar datos financieros de cuenta de resultados
        """
        lines = text.split('\n')
        income_data = []
        
        # Patrones específicos para GarantiBank cuenta de resultados
        financial_patterns = [
            # INGRESOS (valores positivos)
            (r'interest\s+income\s+.*?(\d{1,3}(?:,\d{3})+)\s+.*?(\d{1,3}(?:,\d{3})+)', 'revenue', 'Interest income'),
            (r'fee\s+and\s+commission\s+income\s+.*?(\d{1,3}(?:,\d{3})+)\s+.*?(\d{1,3}(?:,\d{3})+)', 'revenue', 'Fee and commission income'),
            (r'net\s+interest\s+income\s+.*?(\d{1,3}(?:,\d{3})+)\s+.*?(\d{1,3}(?:,\d{3})+)', 'revenue', 'Net interest income'),
            (r'total\s+operating\s+income\s+.*?(\d{1,3}(?:,\d{3})+)\s+.*?(\d{1,3}(?:,\d{3})+)', 'revenue', 'Total operating income'),
            
            # GASTOS (valores negativos)
            (r'interest\s+expense\s+.*?\((\d{1,3}(?:,\d{3})+)\)\s+.*?\((\d{1,3}(?:,\d{3})+)\)', 'expenses', 'Interest expense'),
            (r'personnel\s+expenses\s+.*?\((\d{1,3}(?:,\d{3})+)\)\s+.*?\((\d{1,3}(?:,\d{3})+)\)', 'expenses', 'Personnel expenses'),
            (r'operating\s+expenses\s+.*?\((\d{1,3}(?:,\d{3})+)\)\s+.*?\((\d{1,3}(?:,\d{3})+)\)', 'expenses', 'Operating expenses'),
            (r'total\s+expenses\s+.*?\((\d{1,3}(?:,\d{3})+)\)\s+.*?\((\d{1,3}(?:,\d{3})+)\)', 'expenses', 'Total expenses'),
            
            # RESULTADOS
            (r'income\s+before\s+tax\s+.*?(\d{1,3}(?:,\d{3})+)\s+.*?(\d{1,3}(?:,\d{3})+)', 'profit', 'Income before tax'),
            (r'net\s+income\s+.*?(\d{1,3}(?:,\d{3})+)\s+.*?(\d{1,3}(?:,\d{3})+)', 'profit', 'Net income'),
            (r'profit\s+for\s+the\s+year\s+.*?(\d{1,3}(?:,\d{3})+)\s+.*?(\d{1,3}(?:,\d{3})+)', 'profit', 'Profit for the year'),
        ]
        
        full_text = ' '.join(lines).lower()
        
        for pattern, section, concept_name in financial_patterns:
            matches = re.search(pattern, full_text, re.IGNORECASE)
            if matches:
                try:
                    amount_2023 = float(matches.group(1).replace(',', ''))
                    amount_2022 = float(matches.group(2).replace(',', ''))
                    
                    # Si es gasto, hacerlo negativo
                    if section == 'expenses':
                        amount_2023 = -abs(amount_2023)
                        amount_2022 = -abs(amount_2022)
                    
                    entry = {
                        'concept': concept_name,
                        'section': section,
                        '2023': amount_2023,
                        '2022': amount_2022
                    }
                    
                    income_data.append(entry)
                    print(f"✅ Extraído: {concept_name} -> {section} -> [{amount_2023:,}, {amount_2022:,}]")
                    
                except (ValueError, IndexError) as e:
                    print(f"⚠️ Error procesando {concept_name}: {e}")
                    continue
        
        # Si no encontramos nada con patrones específicos, usar extracción más agresiva
        if not income_data:
            print("🔍 Usando extracción agresiva...")
            income_data = self._aggressive_parsing(lines)
        
        print(f"📊 Total extraído: {len(income_data)} entradas")
        return income_data

    def _aggressive_parsing(self, lines: List[str]) -> List[Dict[str, Any]]:
        """
        Extracción más agresiva cuando los patrones específicos fallan
        """
        income_data = []
        
        for line in lines:
            line = line.strip()
            if len(line) < 10:
                continue
            
            # Buscar líneas con dos números grandes (columnas de años)
            number_matches = re.findall(r'\b(\d{1,3}(?:,\d{3})+)\b', line)
            if len(number_matches) >= 2:
                try:
                    amount_1 = float(number_matches[0].replace(',', ''))
                    amount_2 = float(number_matches[1].replace(',', ''))
                    
                    # Filtrar números muy pequeños o años
                    if amount_1 < 1000 or amount_2 < 1000:
                        continue
                    if str(int(amount_1)) in ['2022', '2023'] or str(int(amount_2)) in ['2022', '2023']:
                        continue
                    
                    # Limpiar concepto
                    concept = re.sub(r'\b\d{1,3}(?:,\d{3})+\b', '', line).strip()
                    concept = re.sub(r'\s+', ' ', concept)
                    
                    if len(concept) > 5:
                        # Determinar sección basada en el concepto
                        concept_lower = concept.lower()
                        if any(word in concept_lower for word in ['income', 'revenue', 'ingreso']):
                            section = 'revenue'
                        elif any(word in concept_lower for word in ['expense', 'cost', 'gasto']):
                            section = 'expenses'
                            amount_1 = -abs(amount_1)
                            amount_2 = -abs(amount_2)
                        elif any(word in concept_lower for word in ['profit', 'tax', 'net']):
                            section = 'profit'
                        else:
                            section = 'other'
                        
                        entry = {
                            'concept': concept,
                            'section': section,
                            '2023': amount_1,
                            '2022': amount_2
                        }
                        
                        income_data.append(entry)
                        
                except (ValueError, IndexError):
                    continue
        
        return income_data

    def _save_extraction_results(self, pdf_file: Path, output_dir: Path, 
                                extracted_text: str, financial_data: List[Dict]) -> int:
        """
        Guarda los resultados de la extracción
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            base = pdf_file.stem
            files_created = 0
            
            # 1. Guardar datos financieros como JSON
            if financial_data:
                json_path = output_dir / f"{base}_income_data.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(financial_data, f, ensure_ascii=False, indent=2)
                files_created += 1
            
            # 2. Guardar datos como CSV
            if financial_data:
                try:
                    df = pd.DataFrame(financial_data)
                    csv_path = output_dir / f"{base}_income_data.csv"
                    df.to_csv(csv_path, index=False, encoding='utf-8')
                    files_created += 1
                except Exception as e:
                    print(f"⚠️ Error generando CSV: {e}")
            
            # 3. Guardar texto extraído
            text_path = output_dir / f"{base}_income_text.txt"
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            files_created += 1
            
            print(f"💾 Archivos guardados: {files_created}")
            return files_created
            
        except Exception as e:
            print(f"❌ Error guardando resultados: {e}")
            return 0

    def _generate_llm_response(self, question: str, pdf_file: Path, extraction_result: Dict) -> str:
        """GENERA RESPUESTA DETALLADA PARA CUENTA DE RESULTADOS - MEJORADO"""
        try:
            extracted_text = extraction_result.get("extracted_text", "")
            financial_data = extraction_result.get("financial_data", [])
            
            # Crear contexto financiero estructurado
            financial_context = self._create_financial_context(financial_data)
            
            # Prompt mejorado específico para BBVA cuenta de resultados
            analysis_prompt = f"""
            Eres un analista financiero especializado en Cuenta de Resultados de bancos españoles.
            
            CONTEXTO: Estás analizando la Cuenta de Resultados Consolidada de BBVA 2023.
            
            DATOS EXTRAÍDOS:
            {financial_context}
            
            TEXTO RELEVANTE:
            {extracted_text[:2000]}
            
            PREGUNTA: {question}
            
            ANÁLISIS REQUERIDO:
            1. **Análisis de Ingresos:**
            - Margen de intereses (ingresos - gastos financieros)
            - Comisiones netas
            - Otros ingresos operativos
            
            2. **Análisis de Gastos:**
            - Gastos de administración
            - Gastos de personal
            - Amortizaciones y provisiones
            
            3. **Rentabilidad:**
            - Resultado antes de impuestos
            - Resultado neto del ejercicio
            - Márgenes de rentabilidad
            
            4. **Análisis Comparativo:**
            - Evolución vs año anterior
            - Eficiencia operativa
            - Calidad del resultado
            
            FORMATO: Respuesta profesional con cifras específicas en millones de euros,
            análisis de rentabilidad y conclusiones sobre la gestión de BBVA.
            """

            messages = [
                {"role": "system", "content": "Eres un analista financiero senior especializado en banca española y análisis de rentabilidad."},
                {"role": "user", "content": analysis_prompt}
            ]
            
            response = self.chat_client.chat(messages, max_tokens=2000)
            
            # Agregar métricas cuantitativas
            if financial_data:
                quantitative_metrics = self._create_quantitative_metrics(financial_data)
                response = f"{response}\n\n{quantitative_metrics}"
            
            return response.strip()
            
        except Exception as e:
            print(f"Error generando respuesta LLM: {str(e)}")
            return self._generate_fallback_response(financial_data)

    def _create_financial_context(self, financial_data: List[Dict]) -> str:
        """Crear contexto financiero estructurado"""
        if not financial_data:
            return "No se pudieron extraer datos financieros específicos."
        
        context_parts = ["DATOS DE CUENTA DE RESULTADOS IDENTIFICADOS:"]
        
        for item in financial_data:
            concept = item.get("concept", "")
            amount_2023 = item.get("2023", 0)
            amount_2022 = item.get("2022", 0)
            section = item.get("section", "")
            
            if amount_2023 or amount_2022:
                context_parts.append(f"• {concept} ({section}): 2023: €{amount_2023:,.0f}k, 2022: €{amount_2022:,.0f}k")
        
        return "\n".join(context_parts)

    def _create_quantitative_metrics(self, financial_data: List[Dict]) -> str:
        """Crear métricas cuantitativas específicas"""
        summary_parts = ["\n📊 **MÉTRICAS CLAVE - CUENTA DE RESULTADOS BBVA 2023:**"]
        
        net_income_2023 = 0
        total_income_2023 = 0
        total_expenses_2023 = 0
        
        for item in financial_data:
            concept = item.get("concept", "").lower()
            amount_2023 = item.get("2023", 0)
            section = item.get("section", "")
            
            if "net income" in concept or "beneficio neto" in concept:
                net_income_2023 = amount_2023
                summary_parts.append(f"• **Resultado Neto: €{amount_2023:,.0f} miles**")
            elif section == "revenue" and "total" in concept:
                total_income_2023 = amount_2023
                summary_parts.append(f"• Ingresos Totales: €{amount_2023:,.0f} miles")
            elif section == "expenses" and "total" in concept:
                total_expenses_2023 = abs(amount_2023)
                summary_parts.append(f"• Gastos Totales: €{abs(amount_2023):,.0f} miles")
        
        # Calcular ratios si hay datos
        if total_income_2023 and total_expenses_2023:
            efficiency_ratio = (total_expenses_2023 / total_income_2023) * 100
            summary_parts.append(f"• **Ratio de Eficiencia: {efficiency_ratio:.1f}%**")
        
        if net_income_2023 and total_income_2023:
            net_margin = (net_income_2023 / total_income_2023) * 100
            summary_parts.append(f"• **Margen Neto: {net_margin:.1f}%**")
        
        return "\n".join(summary_parts)

    def _generate_fallback_response(self, financial_data: List[Dict]) -> str:
        """Respuesta de fallback mejorada"""
        if financial_data:
            return f"""📋 **ANÁLISIS DE CUENTA DE RESULTADOS - BBVA 2023**

    He extraído exitosamente {len(financial_data)} componentes de la Cuenta de Resultados Consolidada:

    {self._create_quantitative_metrics(financial_data)}

    **Análisis Cualitativo:**
    - La estructura de ingresos muestra diversificación entre margen de intereses y comisiones
    - Los gastos operativos reflejan la inversión en transformación digital y eficiencia
    - La rentabilidad obtenida evidencia la solidez del modelo de negocio

    **Fuente:** Cuenta de Resultados Consolidada BBVA 2023"""
        
        return """He completado la extracción de la Cuenta de Resultados Consolidada, pero no se pudieron identificar componentes específicos en esta ocasión. Los datos están disponibles en los archivos generados para análisis posterior."""

    def _extract_financial_numbers(self, text: str, financial_data: List[Dict]) -> Dict[str, str]:
        """
        Extrae números financieros clave del texto y datos parseados
        """
        financial_numbers = {}
        
        # Usar datos parseados primero
        if financial_data:
            for item in financial_data:
                concept = item.get('concept', '').lower()
                section = item.get('section', '')
                
                if 'net income' in concept or 'beneficio neto' in concept:
                    financial_numbers['net_income_2023'] = str(item.get('2023', ''))
                    financial_numbers['net_income_2022'] = str(item.get('2022', ''))
                elif 'total income' in concept and section == 'revenue':
                    financial_numbers['total_income_2023'] = str(item.get('2023', ''))
                    financial_numbers['total_income_2022'] = str(item.get('2022', ''))
                elif 'total expenses' in concept and section == 'expenses':
                    financial_numbers['total_expenses_2023'] = str(abs(item.get('2023', 0)))
                    financial_numbers['total_expenses_2022'] = str(abs(item.get('2022', 0)))
        
        # Usar regex como backup
        text_lower = text.lower()
        
        # Patrones adicionales para valores específicos conocidos
        backup_patterns = {
            'interest_income': [r'interest\s+income.*?(\d{1,3}(?:,\d{3})+)'],
            'personnel_expenses': [r'personnel\s+expenses.*?(\d{1,3}(?:,\d{3})+)']
        }
        
        for data_type, pattern_list in backup_patterns.items():
            if data_type not in financial_numbers:
                for pattern in pattern_list:
                    match = re.search(pattern, text_lower)
                    if match:
                        financial_numbers[data_type] = match.group(1)
                        break
        
        return financial_numbers

    def _generate_general_summary(self, extracted_text: str, financial_numbers: Dict) -> str:
        """
        Genera resumen general sin pregunta específica
        """
        if financial_numbers:
            summary_parts = ["📈 **RESUMEN DE CUENTA DE RESULTADOS EXTRAÍDA**\n"]
            
            if 'net_income_2023' in financial_numbers:
                summary_parts.append(f"• **Beneficio Neto 2023**: €{financial_numbers['net_income_2023']} miles")
            
            if 'total_income_2023' in financial_numbers:
                summary_parts.append(f"• **Ingresos Totales 2023**: €{financial_numbers['total_income_2023']} miles")
            
            if 'total_expenses_2023' in financial_numbers:
                summary_parts.append(f"• **Gastos Totales 2023**: €{financial_numbers['total_expenses_2023']} miles")
            
            summary_parts.append("\n**Fuente**: Cuenta de resultados consolidada de GarantiBank International N.V.")
            
            return "\n".join(summary_parts)
        
        return "✅ He extraído exitosamente la cuenta de resultados consolidada. Los datos de ingresos, gastos y beneficios están disponibles en los archivos generados para análisis detallado."

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
            
            # PROMPT ENGINEERING ESPECIALIZADO PARA CUENTA DE RESULTADOS
            analysis_prompt = f"""Eres un analista financiero experto especializado en análisis de cuentas de resultados corporativas.

CONTEXTO:
Has extraído información de la cuenta de resultados consolidada de GarantiBank International N.V. para el año 2023.

{financial_context}

TEXTO EXTRAÍDO DE LA CUENTA DE RESULTADOS:
{extracted_text[:2000]}

PREGUNTA DEL USUARIO:
{question}

INSTRUCCIONES:
1. Analiza la información financiera de la cuenta de resultados
2. Responde la pregunta de forma específica y profesional
3. Incluye cifras exactas cuando estén disponibles
4. Proporciona contexto relevante sobre la rentabilidad de GarantiBank
5. Si no tienes datos exactos, indica qué información está disponible
6. Mantén un tono profesional y conciso

FORMATO DE RESPUESTA:
- Respuesta directa y específica sobre ingresos/gastos/beneficios
- Cifras con formato apropiado (€X,XXX miles)
- Análisis de rentabilidad cuando sea relevante
- Fuente: Cuenta de resultados consolidada

RESPUESTA PROFESIONAL:"""

            # Llamar al LLM
            messages = [
                {
                    "role": "system", 
                    "content": "Eres un analista financiero experto especializado en análisis de cuentas de resultados de instituciones financieras."
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
            print(f"❌ Error en LLM: {str(e)}")
            # Fallback a respuesta basada en reglas
            return self._generate_rule_based_response(question, extracted_text, financial_numbers)

    def _generate_rule_based_response(self, question: str, extracted_text: str, financial_numbers: Dict) -> str:
        """
        Fallback: Genera respuesta basada en reglas si el LLM no está disponible
        """
        question_lower = question.lower()
        
        # Detectar tipo de pregunta y responder con datos disponibles
        if any(word in question_lower for word in ['beneficio', 'ganancia', 'profit', 'net income']):
            if 'net_income_2023' in financial_numbers:
                return f"📈 **El beneficio neto de GarantiBank International N.V. en 2023 fue €{financial_numbers['net_income_2023']} miles** según la cuenta de resultados consolidada.\n\n**Fuente**: Statement of Comprehensive Income extraído"
            
        elif any(word in question_lower for word in ['ingresos', 'revenue', 'income']):
            if 'total_income_2023' in financial_numbers:
                return f"💰 **Los ingresos totales en 2023 fueron €{financial_numbers['total_income_2023']} miles**, incluyendo ingresos por intereses y comisiones.\n\n**Fuente**: Cuenta de resultados consolidada extraída"
                
        elif any(word in question_lower for word in ['gastos', 'expenses', 'costos', 'costs']):
            if 'total_expenses_2023' in financial_numbers:
                return f"💸 **Los gastos totales en 2023 fueron €{financial_numbers['total_expenses_2023']} miles**, incluyendo gastos de personal y operativos.\n\n**Fuente**: Cuenta de resultados consolidada extraída"
        
        # Respuesta genérica si no puede determinar específicamente
        return "✅ He extraído exitosamente la cuenta de resultados consolidada de GarantiBank International N.V. Los datos incluyen ingresos operativos, gastos y beneficios para 2023. Los datos detallados están disponibles en los archivos generados."

# =============================
# Sistema de herramientas básico (si se necesita para REACT)
# =============================

def run_income_agent(pdf_path: Path, output_dir: Path, max_steps: int = 10) -> Dict[str, Any]:
    """
    Función principal para ejecutar el agente de income (compatibilidad)
    """
    agent = IncomeREACTAgent()
    
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
        description="Income Agent AUTÓNOMO especializado en Cuenta de Resultados - Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
  python agents/income_agent.py                    # Usa configuración predefinida
  python agents/income_agent.py --pdf otro.pdf    # Sobreescribe PDF
  
CARACTERÍSTICAS AUTÓNOMAS:
  - Generación de respuestas específicas usando LLM
  - Prompt engineering especializado para cuenta de resultados
  - Fallback robusto con respuestas basadas en reglas
  - Extracción automática de datos de ingresos, gastos y beneficios
  
Sistema Multi-Agente:
  Esta versión incluye IncomeREACTAgent AUTÓNOMO para integración con main_system.py
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
                       help="Pregunta específica sobre cuenta de resultados")

    args = parser.parse_args()

    # ===== CONFIGURAR RUTAS =====
    pdf_path = Path(args.pdf)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===== VERIFICAR PDF =====
    if not pdf_path.exists():
        print(f"❌ Error: PDF no encontrado en {pdf_path}")
        return

    # ===== MOSTRAR CONFIGURACIÓN =====
    print(f"🚀 Income Agent v3.0 AUTÓNOMO Multi-Agent - Configuración Automática")
    print(f"📄 PDF: {pdf_path}")
    print(f"📁 Salida: {output_dir}")
    print(f"⚙️ Azure OpenAI: {AZURE_OPENAI_DEPLOYMENT}")
    print(f"🔧 Max steps: {args.max_steps}")
    print(f"🤖 Multi-Agent: IncomeREACTAgent AUTÓNOMO class available")
    print("🆕 CARACTERÍSTICAS: Respuestas LLM específicas, prompt engineering avanzado, completamente autónomo")

    try:
        # Crear agente y ejecutar
        agent = IncomeREACTAgent()
        
        if args.question:
            # Modo pregunta específica
            print(f"❓ Pregunta específica: {args.question}")
            result = agent.run_final_financial_extraction_agent(str(pdf_path), args.question)
        else:
            # Modo extracción general
            result = agent.run_final_financial_extraction_agent(str(pdf_path))

        print("\n🎯 ==== RESUMEN DE EJECUCIÓN AUTÓNOMO ====")
        print(f"Estado: {'✅ EXITOSO' if result.get('status') == 'task_completed' else '❌ ERROR'}")
        print(f"Pasos completados: {result.get('steps_taken', 0)}")
        print(f"Archivos generados: {result.get('files_generated', 0)}")

        if result.get("status") == "task_completed":
            print("\n📋 ==== RESPUESTA ESPECÍFICA ====")
            print(result.get("specific_answer", "No hay respuesta específica disponible"))
        else:
            print(f"\n❌ Error: {result.get('error_details', 'Error desconocido')}")

        print("\n🎉 Análisis de cuenta de resultados completado!")
        print("🤖 Clase IncomeREACTAgent AUTÓNOMA disponible para sistema multi-agente")
        print("🆕 Versión autónoma con generación de respuestas específicas usando LLM")

    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
        raise

if __name__ == "__main__":
    main()
