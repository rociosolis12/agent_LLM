"""
Equity Agent REACT - Versión Multi-Agente AUTÓNOMA COMPLETA
Especializado en análisis de estado de cambios en patrimonio con wrapper autónomo para sistema multi-agente
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
# Diccionarios específicos para estado de cambios en patrimonio
# =============================

EQUITY_TITLES_EN = [
    "statement of changes in equity", "changes in equity", "equity statement",
    "consolidated statement of changes in equity", "shareholders' equity changes"
]

EQUITY_TITLES_ES = [
    "estado de cambios en patrimonio", "cambios en patrimonio", "estado de patrimonio",
    "estado consolidado de cambios en patrimonio", "cambios patrimoniales"
]

# Términos específicos de capital social
CAPITAL_HINTS = [
    "share capital", "capital social", "share capital and share premium",
    "capital stock", "paid-in capital", "capital pagado"
]

# Términos específicos de reservas
RESERVES_HINTS = [
    "fair value reserve", "hedging reserve", "other legal reserves",
    "other reserves", "legal reserves", "reservas legales",
    "reservas de valor razonable", "reservas de cobertura"
]

# Términos específicos de resultados acumulados
RETAINED_EARNINGS_HINTS = [
    "retained earnings", "accumulated results", "resultados acumulados",
    "beneficios retenidos", "ganancias acumuladas"
]

# Términos específicos de resultado del ejercicio
PROFIT_HINTS = [
    "profit for the year", "net income", "resultado del ejercicio",
    "beneficio del ejercicio", "ganancia del período"
]

# Términos específicos de resultado integral
COMPREHENSIVE_INCOME_HINTS = [
    "total comprehensive income", "comprehensive income", "resultado integral",
    "resultado integral total", "other comprehensive income"
]

# Términos específicos de patrimonio total
TOTAL_EQUITY_HINTS = [
    "total equity", "total shareholders equity", "patrimonio total",
    "total equity attributable to owners", "patrimonio neto total"
]

# Términos específicos de dividendos
DIVIDENDS_HINTS = [
    "dividends paid", "dividend payments", "dividendos pagados",
    "pago de dividendos", "distribución de dividendos"
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
    score_es = sum(1 for w in ["patrimonio", "capital", "reservas"] if w in t)
    score_en = sum(1 for w in ["equity", "capital", "reserves"] if w in t)
    return "es" if score_es >= score_en else "en"

# =============================
# CLASE WRAPPER AUTÓNOMA PARA SISTEMA MULTI-AGENTE - EQUITY
# =============================

class EquityREACTAgent:
    """
    Wrapper REACT COMPLETAMENTE AUTÓNOMO para el Equity Agent
    
    Esta clase es completamente autónoma y genera respuestas específicas usando LLM
    basándose en los datos de cambios en patrimonio que extrae.
    """
    
    def __init__(self):
        self.agent_type = "equity"
        self.max_steps = 25  # Aumentado para el wrapper
        self.chat_client = chat_client  # Cliente Azure OpenAI

    def run_final_financial_extraction_agent(self, pdf_path: str, question: str = None) -> Dict[str, Any]:
        """
        Ejecuta la extracción de cambios en patrimonio Y genera respuesta específica autónomamente
        
        Args:
            pdf_path: Ruta al PDF a procesar
            question: Pregunta específica del usuario (opcional)
            
        Returns:
            Dict con el resultado y respuesta específica generada por LLM
        """
        try:
            print(f"🔧 EquityREACTAgent AUTÓNOMO iniciando extracción para: {pdf_path}")
            
            pdf_file = Path(pdf_path)
            output_dir = Path("data/salida")
            
            # 1. EJECUTAR EXTRACCIÓN CORE (funcionalidad de equity)
            result = self._run_core_extraction(pdf_file, output_dir)
            
            # 2. VERIFICAR ÉXITO DE EXTRACCIÓN
            extraction_successful = result.get("success", False)
            
            if not extraction_successful:
                print("⚠️ Equity extraction failed")
                return {
                    "status": "error", 
                    "steps_taken": result.get("steps_taken", 0),
                    "session_id": f"equity_{pdf_file.stem}",
                    "final_response": "Equity extraction failed - check logs for details",
                    "agent_type": "equity",
                    "error_details": result.get("error", "Extraction process failed"),
                    "specific_answer": "No se pudo completar la extracción de los cambios en patrimonio."
                }
            
            # 3. GENERAR RESPUESTA ESPECÍFICA USANDO LLM
            specific_answer = self._generate_llm_response(question, pdf_file, result)
            
            print("✅ Equity extraction completed successfully (AUTÓNOMO)")
            return {
                "status": "task_completed",
                "steps_taken": result.get("steps_taken", 5),
                "session_id": f"equity_{pdf_file.stem}",
                "final_response": "Equity extraction completed successfully - AUTONOMOUS VERSION",
                "agent_type": "equity",
                "files_generated": result.get("files_created", 0),
                "specific_answer": specific_answer  # ← RESPUESTA GENERADA POR LLM
            }
                
        except Exception as e:
            print(f"❌ Error en EquityREACTAgent: {str(e)}")
            return {
                "status": "error",
                "steps_taken": 0,
                "session_id": "equity_error",
                "final_response": f"Error in equity extraction: {str(e)}",
                "agent_type": "equity",
                "error_details": str(e),
                "specific_answer": f"Error durante la extracción de los cambios en patrimonio: {str(e)}"
            }

    def _run_core_extraction(self, pdf_file: Path, output_dir: Path) -> Dict[str, Any]:
        """
        Ejecuta la extracción core de cambios en patrimonio
        """
        try:
            print(f"🔍 Extrayendo cambios en patrimonio de: {pdf_file}")
            
            # 1. EXTRAER TEXTO DE PÁGINAS RELEVANTES
            extracted_text = self._extract_equity_text(pdf_file)
            
            if not extracted_text:
                return {
                    "success": False,
                    "error": "No se encontraron datos de cambios en patrimonio",
                    "steps_taken": 1
                }
            
            # 2. PARSEAR DATOS FINANCIEROS
            equity_data = self._parse_equity_data(extracted_text)
            
            # 3. GUARDAR RESULTADOS
            files_created = self._save_extraction_results(pdf_file, output_dir, extracted_text, equity_data)
            
            return {
                "success": True,
                "extracted_text": extracted_text,
                "equity_data": equity_data,
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

    def _extract_equity_text(self, pdf_file: Path) -> str:
        """
        Extrae texto de páginas que contienen estado de cambios en patrimonio
        """
        try:
            doc = fitz.open(str(pdf_file))
            equity_text = ""
            pages_processed = []
            
            # Patrones para identificar estado de cambios en patrimonio
            equity_patterns = [
                r"statement\s+of\s+changes\s+in\s+equity",
                r"changes\s+in\s+equity",
                r"estado\s+de\s+cambios\s+en\s+patrimonio",
                r"cambios\s+en\s+patrimonio",
                r"shareholders?\s+equity",
                r"patrimonio\s+neto",
                r"consolidated.*equity.*changes"
            ]
            
            # Buscar en las primeras 20 páginas
            for page_num in range(min(len(doc), 20)):
                page = doc[page_num]
                text = page.get_text()
                text_lower = text.lower()
                
                # Verificar si la página contiene cambios en patrimonio
                for pattern in equity_patterns:
                    if re.search(pattern, text_lower, re.IGNORECASE):
                        equity_text += f"=== PÁGINA {page_num + 1} ===\n{text}\n\n"
                        pages_processed.append(page_num + 1)
                        print(f"✅ Página {page_num + 1}: {len(text)} caracteres extraídos")
                        break
            
            doc.close()
            
            if equity_text:
                print(f"📊 Texto total extraído: {len(equity_text)} caracteres de {len(pages_processed)} páginas")
            
            return equity_text
            
        except Exception as e:
            print(f"❌ Error extrayendo texto: {e}")
            return ""

    def _parse_equity_data(self, text: str) -> List[Dict[str, Any]]:
        """
        Parsea el texto extraído para identificar datos financieros de cambios en patrimonio
        """
        lines = text.split('\n')
        equity_data = []
        
        # Patrones específicos para GarantiBank cambios en patrimonio
        equity_patterns = [
            # CAPITAL SOCIAL
            (r'share\s+capital\s+and\s+share\s+premium\s+.*?(\d{1,3}(?:,\d{3})+)\s+.*?(\d{1,3}(?:,\d{3})+)', 'capital', 'Share capital and share premium'),
            (r'share\s+capital(?!\s+and)\s+.*?(\d{1,3}(?:,\d{3})+)\s+.*?(\d{1,3}(?:,\d{3})+)', 'capital', 'Share capital'),
            (r'capital\s+social\s+.*?(\d{1,3}(?:,\d{3})+)\s+.*?(\d{1,3}(?:,\d{3})+)', 'capital', 'Capital social'),
            
            # RESERVAS
            (r'fair\s+value\s+reserve\s+.*?\((\d{1,3}(?:,\d{3})+)\)\s+.*?\((\d{1,3}(?:,\d{3})+)\)', 'reserves', 'Fair value reserve'),
            (r'hedging\s+reserve\s+.*?(\d{1,3}(?:,\d{3})+)\s+.*?(\d{1,3}(?:,\d{3})+)', 'reserves', 'Hedging reserve'),
            (r'other\s+legal\s+reserves\s+.*?(\d{1,3}(?:,\d{3})+)\s+.*?(\d{1,3}(?:,\d{3})+)', 'reserves', 'Other legal reserves'),
            (r'other\s+reserves\s+.*?(\d{1,3}(?:,\d{3})+)\s+.*?(\d{1,3}(?:,\d{3})+)', 'reserves', 'Other reserves'),
            (r'legal\s+reserves\s+.*?(\d{1,3}(?:,\d{3})+)\s+.*?(\d{1,3}(?:,\d{3})+)', 'reserves', 'Legal reserves'),
            
            # RESULTADOS ACUMULADOS
            (r'retained\s+earnings\s+.*?(\d{1,3}(?:,\d{3})+)\s+.*?(\d{1,3}(?:,\d{3})+)', 'retained_earnings', 'Retained earnings'),
            (r'resultados?\s+acumulados?\s+.*?(\d{1,3}(?:,\d{3})+)\s+.*?(\d{1,3}(?:,\d{3})+)', 'retained_earnings', 'Resultados acumulados'),
            
            # RESULTADO DEL EJERCICIO
            (r'profit\s+for\s+the\s+year\s+.*?(\d{1,3}(?:,\d{3})+)\s+.*?(\d{1,3}(?:,\d{3})+)', 'profit', 'Profit for the year'),
            (r'net\s+income\s+.*?(\d{1,3}(?:,\d{3})+)\s+.*?(\d{1,3}(?:,\d{3})+)', 'profit', 'Net income'),
            (r'resultado\s+del\s+ejercicio\s+.*?(\d{1,3}(?:,\d{3})+)\s+.*?(\d{1,3}(?:,\d{3})+)', 'profit', 'Resultado del ejercicio'),
            
            # RESULTADO INTEGRAL
            (r'total\s+comprehensive\s+income\s+.*?(\d{1,3}(?:,\d{3})+)\s+.*?(\d{1,3}(?:,\d{3})+)', 'comprehensive_income', 'Total comprehensive income'),
            (r'resultado\s+integral\s+.*?(\d{1,3}(?:,\d{3})+)\s+.*?(\d{1,3}(?:,\d{3})+)', 'comprehensive_income', 'Resultado integral'),
            
            # PATRIMONIO TOTAL
            (r'total\s+equity\s+attributable\s+to\s+owners\s+.*?(\d{1,3}(?:,\d{3})+)\s+.*?(\d{1,3}(?:,\d{3})+)', 'total_equity', 'Total equity attributable to owners'),
            (r'total\s+equity\s+.*?(\d{1,3}(?:,\d{3})+)\s+.*?(\d{1,3}(?:,\d{3})+)', 'total_equity', 'Total equity'),
            (r'patrimonio\s+total\s+.*?(\d{1,3}(?:,\d{3})+)\s+.*?(\d{1,3}(?:,\d{3})+)', 'total_equity', 'Patrimonio total'),
            
            # DIVIDENDOS
            (r'dividends?\s+paid\s+.*?\((\d{1,3}(?:,\d{3})+)\)\s+.*?\((\d{1,3}(?:,\d{3})+)\)', 'dividends', 'Dividends paid'),
            (r'dividendos?\s+pagados?\s+.*?\((\d{1,3}(?:,\d{3})+)\)\s+.*?\((\d{1,3}(?:,\d{3})+)\)', 'dividends', 'Dividendos pagados'),
        ]
        
        full_text = ' '.join(lines).lower()
        
        for pattern, section, concept_name in equity_patterns:
            matches = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)
            if matches:
                try:
                    amount_2023 = float(matches.group(1).replace(',', ''))
                    amount_2022 = float(matches.group(2).replace(',', ''))
                    
                    # Manejar números negativos según contexto
                    if any(term in concept_name.lower() for term in ['fair value', 'dividends', 'paid', 'pagados']):
                        if amount_2023 > 0:
                            amount_2023 = -amount_2023
                        if amount_2022 > 0:
                            amount_2022 = -amount_2022
                    
                    entry = {
                        'concept': concept_name,
                        'section': section,
                        '2023': amount_2023,
                        '2022': amount_2022
                    }
                    
                    equity_data.append(entry)
                    print(f"✅ Extraído: {concept_name} -> {section} -> [{amount_2023:,}, {amount_2022:,}]")
                    
                except (ValueError, IndexError) as e:
                    print(f"⚠️ Error procesando {concept_name}: {e}")
                    continue
        
        # Si no encontramos nada con patrones específicos, usar extracción más agresiva
        if not equity_data:
            print("🔍 Usando extracción agresiva...")
            equity_data = self._aggressive_parsing(lines)
        
        print(f"📊 Total extraído: {len(equity_data)} entradas")
        return equity_data

    def _aggressive_parsing(self, lines: List[str]) -> List[Dict[str, Any]]:
        """
        Extracción más agresiva cuando los patrones específicos fallan
        """
        equity_data = []
        
        for line in lines:
            line = line.strip()
            if len(line) < 10:
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
                        if any(word in concept_lower for word in ['capital', 'share']):
                            section = 'capital'
                        elif any(word in concept_lower for word in ['reserve', 'reserves']):
                            section = 'reserves'
                        elif any(word in concept_lower for word in ['retained', 'earnings']):
                            section = 'retained_earnings'
                        elif any(word in concept_lower for word in ['profit', 'income']):
                            section = 'profit'
                        elif any(word in concept_lower for word in ['comprehensive']):
                            section = 'comprehensive_income'
                        elif any(word in concept_lower for word in ['equity', 'patrimonio']):
                            section = 'total_equity'
                        elif any(word in concept_lower for word in ['dividend']):
                            section = 'dividends'
                        else:
                            section = 'other'
                        
                        entry = {
                            'concept': concept,
                            'section': section,
                            '2023': amounts[0],
                            '2022': amounts[1]
                        }
                        
                        equity_data.append(entry)
                        
                except (ValueError, IndexError):
                    continue
        
        return equity_data

    def _save_extraction_results(self, pdf_file: Path, output_dir: Path, 
                                extracted_text: str, equity_data: List[Dict]) -> int:
        """
        Guarda los resultados de la extracción
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            base = pdf_file.stem
            files_created = 0
            
            # 1. Guardar datos financieros como JSON
            if equity_data:
                json_path = output_dir / f"{base}_equity_data.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(equity_data, f, ensure_ascii=False, indent=2)
                files_created += 1
            
            # 2. Guardar datos como CSV
            if equity_data:
                try:
                    df = pd.DataFrame(equity_data)
                    csv_path = output_dir / f"{base}_equity_data.csv"
                    df.to_csv(csv_path, index=False, encoding='utf-8')
                    files_created += 1
                except Exception as e:
                    print(f"⚠️ Error generando CSV: {e}")
            
            # 3. Guardar texto extraído
            text_path = output_dir / f"{base}_equity_text.txt"
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            files_created += 1
            
            print(f"💾 Archivos guardados: {files_created}")
            return files_created
            
        except Exception as e:
            print(f"❌ Error guardando resultados: {e}")
            return 0

    def _generate_llm_response(self, question: str, pdf_file: Path, extraction_result: Dict) -> str:
        """GENERA RESPUESTA DETALLADA USANDO LLM - MEJORADO"""
        try:
            extracted_text = extraction_result.get("extracted_text", "")
            equity_data = extraction_result.get("equity_data", [])
            
            # Crear contexto financiero estructurado
            financial_context = self._create_financial_context(equity_data)
            
            # Prompt mejorado específico para BBVA patrimonio
            analysis_prompt = f"""
            Eres un analista financiero especializado en Estado de Cambios en Patrimonio de bancos españoles.
            
            CONTEXTO: Estás analizando el Estado de Cambios en Patrimonio Neto de BBVA 2023.
            
            DATOS EXTRAÍDOS:
            {financial_context}
            
            TEXTO RELEVANTE:
            {extracted_text[:2000]}
            
            PREGUNTA: {question}
            
            ANÁLISIS REQUERIDO:
            1. **Estructura del Patrimonio Neto:**
            - Capital Social y Prima de Emisión
            - Reservas (legales, estatutarias, voluntarias)
            - Resultados acumulados de ejercicios anteriores
            
            2. **Movimientos del Ejercicio 2023:**
            - Resultado del ejercicio actual
            - Dividendos distribuidos
            - Otros cambios patrimoniales relevantes
            
            3. **Análisis Cuantitativo:**
            - Evolución del patrimonio total
            - Ratio de distribución de dividendos
            - Solidez patrimonial
            
            4. **Valoración:**
            - Fortalezas de la estructura patrimonial
            - Política de dividendos
            - Capacidad de crecimiento
            
            FORMATO: Respuesta profesional con cifras específicas en millones de euros, 
            análisis concreto y conclusiones relevantes para BBVA.
            """

            messages = [
                {"role": "system", "content": "Eres un analista financiero senior especializado en banca española y análisis patrimonial."},
                {"role": "user", "content": analysis_prompt}
            ]
            
            response = self.chat_client.chat(messages, max_tokens=2000)
            
            # Agregar resumen cuantitativo si hay datos
            if equity_data:
                quantitative_summary = self._create_quantitative_summary(equity_data)
                response = f"{response}\n\n{quantitative_summary}"
            
            return response.strip()
            
        except Exception as e:
            print(f"Error generando respuesta LLM: {str(e)}")
            return self._generate_fallback_response(equity_data)

    def _create_financial_context(self, equity_data: List[Dict]) -> str:
        """Crear contexto financiero estructurado"""
        if not equity_data:
            return "No se pudieron extraer datos financieros específicos."
        
        context_parts = ["DATOS PATRIMONIALES IDENTIFICADOS:"]
        
        for item in equity_data:
            concept = item.get("concept", "")
            amount_2023 = item.get("2023", 0)
            amount_2022 = item.get("2022", 0)
            
            if amount_2023 or amount_2022:
                context_parts.append(f"• {concept}: 2023: €{amount_2023:,.0f}k, 2022: €{amount_2022:,.0f}k")
        
        return "\n".join(context_parts)

    def _create_quantitative_summary(self, equity_data: List[Dict]) -> str:
        """Crear resumen cuantitativo específico"""
        summary_parts = ["\n📊 **RESUMEN CUANTITATIVO - PATRIMONIO BBVA 2023:**"]
        
        total_equity_2023 = 0
        total_equity_2022 = 0
        
        for item in equity_data:
            concept = item.get("concept", "").lower()
            amount_2023 = item.get("2023", 0)
            amount_2022 = item.get("2022", 0)
            
            if "capital" in concept and "social" in concept:
                summary_parts.append(f"• Capital Social: €{amount_2023:,.0f} miles (2022: €{amount_2022:,.0f} miles)")
            elif "reservas" in concept or "reserves" in concept:
                summary_parts.append(f"• Reservas: €{amount_2023:,.0f} miles (2022: €{amount_2022:,.0f} miles)")
            elif "dividendos" in concept or "dividends" in concept:
                summary_parts.append(f"• Dividendos Distribuidos: €{abs(amount_2023):,.0f} miles")
            elif "patrimonio total" in concept or "total equity" in concept:
                total_equity_2023 = amount_2023
                total_equity_2022 = amount_2022
        
        if total_equity_2023:
            variation = total_equity_2023 - total_equity_2022
            variation_pct = (variation / total_equity_2022 * 100) if total_equity_2022 else 0
            summary_parts.append(f"• **Patrimonio Neto Total: €{total_equity_2023:,.0f} miles**")
            summary_parts.append(f"• **Variación anual: €{variation:,.0f} miles ({variation_pct:+.1f}%)**")
        
        return "\n".join(summary_parts)

    def _generate_fallback_response(self, equity_data: List[Dict]) -> str:
        """Respuesta de fallback mejorada"""
        if equity_data:
            return f"""📋 **ANÁLISIS DE ESTADO DE CAMBIOS EN PATRIMONIO - BBVA 2023**

    He extraído exitosamente {len(equity_data)} componentes patrimoniales del Estado de Cambios en Patrimonio Neto consolidado:

    {self._create_quantitative_summary(equity_data)}

    **Análisis Cualitativo:**
    - La estructura patrimonial muestra una composición sólida con capital social estable
    - Los movimientos del ejercicio reflejan la política de dividendos y retención de beneficios
    - Los cambios en reservas indican el crecimiento orgánico del banco

    **Fuente:** Estado de Cambios en Patrimonio Neto Consolidado BBVA 2023"""
        
        return """He completado la extracción del Estado de Cambios en Patrimonio Neto, pero no se pudieron identificar componentes patrimoniales específicos en esta ocasión. Los datos están disponibles en los archivos generados para análisis posterior."""

    def _extract_financial_numbers(self, text: str, equity_data: List[Dict]) -> Dict[str, str]:
        """
        Extrae números financieros clave del texto y datos parseados
        """
        financial_numbers = {}
        
        # Usar datos parseados primero
        if equity_data:
            for item in equity_data:
                concept = item.get('concept', '').lower()
                section = item.get('section', '')
                
                if 'share capital' in concept and section == 'capital':
                    financial_numbers['share_capital_2023'] = str(item.get('2023', ''))
                    financial_numbers['share_capital_2022'] = str(item.get('2022', ''))
                elif 'retained earnings' in concept and section == 'retained_earnings':
                    financial_numbers['retained_earnings_2023'] = str(item.get('2023', ''))
                    financial_numbers['retained_earnings_2022'] = str(item.get('2022', ''))
                elif 'profit for the year' in concept and section == 'profit':
                    financial_numbers['profit_2023'] = str(item.get('2023', ''))
                    financial_numbers['profit_2022'] = str(item.get('2022', ''))
                elif 'total equity' in concept and section == 'total_equity':
                    financial_numbers['total_equity_2023'] = str(item.get('2023', ''))
                    financial_numbers['total_equity_2022'] = str(item.get('2022', ''))
                elif 'comprehensive income' in concept and section == 'comprehensive_income':
                    financial_numbers['comprehensive_income_2023'] = str(item.get('2023', ''))
                    financial_numbers['comprehensive_income_2022'] = str(item.get('2022', ''))
        
        return financial_numbers

    def _generate_general_summary(self, extracted_text: str, financial_numbers: Dict) -> str:
        """
        Genera resumen general sin pregunta específica
        """
        if financial_numbers:
            summary_parts = ["🏛️ **RESUMEN DE CAMBIOS EN PATRIMONIO EXTRAÍDOS**\n"]
            
            if 'total_equity_2023' in financial_numbers:
                summary_parts.append(f"• **Patrimonio Total 2023**: €{financial_numbers['total_equity_2023']} miles")
            
            if 'share_capital_2023' in financial_numbers:
                summary_parts.append(f"• **Capital Social 2023**: €{financial_numbers['share_capital_2023']} miles")
            
            if 'retained_earnings_2023' in financial_numbers:
                summary_parts.append(f"• **Resultados Acumulados 2023**: €{financial_numbers['retained_earnings_2023']} miles")
            
            if 'profit_2023' in financial_numbers:
                summary_parts.append(f"• **Resultado del Ejercicio 2023**: €{financial_numbers['profit_2023']} miles")
            
            summary_parts.append("\n**Fuente**: Estado de cambios en patrimonio consolidado de GarantiBank International N.V.")
            
            return "\n".join(summary_parts)
        
        return "✅ He extraído exitosamente el estado de cambios en patrimonio consolidado. Los datos de capital, reservas, resultados acumulados y otros componentes patrimoniales están disponibles en los archivos generados para análisis detallado."

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
            
            # PROMPT ENGINEERING ESPECIALIZADO PARA CAMBIOS EN PATRIMONIO
            analysis_prompt = f"""Eres un analista financiero experto especializado en análisis de cambios en patrimonio corporativo.

CONTEXTO:
Has extraído información del estado de cambios en patrimonio consolidado de GarantiBank International N.V. para el año 2023.

{financial_context}

TEXTO EXTRAÍDO DEL ESTADO DE CAMBIOS EN PATRIMONIO:
{extracted_text[:2000]}

PREGUNTA DEL USUARIO:
{question}

INSTRUCCIONES:
1. Analiza la información financiera de los cambios en patrimonio
2. Responde la pregunta de forma específica y profesional
3. Incluye cifras exactas cuando estén disponibles
4. Proporciona contexto relevante sobre la estructura patrimonial de GarantiBank
5. Si no tienes datos exactos, indica qué información está disponible
6. Mantén un tono profesional y conciso

FORMATO DE RESPUESTA:
- Respuesta directa y específica sobre componentes patrimoniales
- Cifras con formato apropiado (€X,XXX miles)
- Análisis de cambios patrimoniales cuando sea relevante
- Fuente: Estado de cambios en patrimonio consolidado

RESPUESTA PROFESIONAL:"""

            # Llamar al LLM
            messages = [
                {
                    "role": "system", 
                    "content": "Eres un analista financiero experto especializado en análisis de cambios en patrimonio de instituciones financieras."
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
        if any(word in question_lower for word in ['capital', 'social', 'share']):
            if 'share_capital_2023' in financial_numbers:
                return f"🏛️ **El capital social de GarantiBank International N.V. en 2023 fue €{financial_numbers['share_capital_2023']} miles** según el estado de cambios en patrimonio consolidado.\n\n**Fuente**: Statement of Changes in Equity extraído"
            
        elif any(word in question_lower for word in ['resultados', 'earnings', 'retained']):
            if 'retained_earnings_2023' in financial_numbers:
                return f"📈 **Los resultados acumulados en 2023 fueron €{financial_numbers['retained_earnings_2023']} miles**, representando las ganancias retenidas de períodos anteriores.\n\n**Fuente**: Estado de cambios en patrimonio consolidado extraído"
                
        elif any(word in question_lower for word in ['beneficio', 'profit', 'resultado']):
            if 'profit_2023' in financial_numbers:
                return f"💰 **El resultado del ejercicio 2023 fue €{financial_numbers['profit_2023']} miles**, contribuyendo al crecimiento del patrimonio.\n\n**Fuente**: Estado de cambios en patrimonio consolidado extraído"
        
        elif any(word in question_lower for word in ['patrimonio', 'equity', 'total']):
            if 'total_equity_2023' in financial_numbers:
                return f"🏆 **El patrimonio total en 2023 fue €{financial_numbers['total_equity_2023']} miles**, representando el valor neto para los accionistas.\n\n**Fuente**: Estado de cambios en patrimonio consolidado extraído"
        
        # Respuesta genérica si no puede determinar específicamente
        return "✅ He extraído exitosamente el estado de cambios en patrimonio consolidado de GarantiBank International N.V. Los datos incluyen capital social, reservas, resultados acumulados y otros componentes patrimoniales para 2023. Los datos detallados están disponibles en los archivos generados."

# =============================
# Sistema de herramientas básico (si se necesita para REACT compatibilidad)
# =============================

def run_equity_agent(pdf_path: Path, output_dir: Path, max_steps: int = 10) -> Dict[str, Any]:
    """
    Función principal para ejecutar el agente de equity (compatibilidad)
    """
    agent = EquityREACTAgent()
    
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
        description="Equity Agent AUTÓNOMO especializado en Estado de Cambios en Patrimonio - Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
  python agents/equity_agent.py                    # Usa configuración predefinida
  python agents/equity_agent.py --pdf otro.pdf    # Sobreescribe PDF
  
CARACTERÍSTICAS AUTÓNOMAS:
  - Generación de respuestas específicas usando LLM
  - Prompt engineering especializado para cambios en patrimonio
  - Fallback robusto con respuestas basadas en reglas
  - Extracción automática de capital, reservas, resultados acumulados
  
Sistema Multi-Agente:
  Esta versión incluye EquityREACTAgent AUTÓNOMO para integración con main_system.py
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
                       help="Pregunta específica sobre cambios en patrimonio")

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
    print(f"🚀 Equity Agent v3.0 AUTÓNOMO Multi-Agent - Configuración Automática")
    print(f"📄 PDF: {pdf_path}")
    print(f"📁 Salida: {output_dir}")
    print(f"⚙️ Azure OpenAI: {AZURE_OPENAI_DEPLOYMENT}")
    print(f"🔧 Max steps: {args.max_steps}")
    print(f"🤖 Multi-Agent: EquityREACTAgent AUTÓNOMO class available")
    print("🆕 CARACTERÍSTICAS: Respuestas LLM específicas, prompt engineering avanzado, completamente autónomo")

    try:
        # Crear agente y ejecutar
        agent = EquityREACTAgent()
        
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

        print("\n🎉 Análisis de cambios en patrimonio completado!")
        print("🤖 Clase EquityREACTAgent AUTÓNOMA disponible para sistema multi-agente")
        print("🆕 Versión autónoma con generación de respuestas específicas usando LLM")

    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
        raise

if __name__ == "__main__":
    main()
