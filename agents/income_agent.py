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
import fitz
from typing import Any, Dict, List, Optional
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI
import groq
import numpy as np
from typing import Tuple
from sklearn.metrics.pairwise import cosine_similarity

# ===== CONFIGURACIÓN DEL PROYECTO =====
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)
os.chdir(project_root)

if not env_path.exists():
    print(f"Warning: Archivo .env no encontrado en {env_path}")

print(" Cargar .env desde el directorio raíz del proyecto...")

# ----- Azure OpenAI Configuration -----
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_EMBEDDING_MODEL = os.getenv("AZURE_EMBEDDING_MODEL", "text-embedding-3-small")


print(" ----- Azure OpenAI Configuration -----")
print(f" Endpoint: {AZURE_OPENAI_ENDPOINT}")
print(f" API Key: {'✓' if AZURE_OPENAI_API_KEY else '✗'}")
print(f" Deployment: {AZURE_OPENAI_DEPLOYMENT}")

# Validación de credenciales
if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
    raise ValueError("Azure OpenAI credentials required")

# ----- Groq Configuration -----
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

print(" ----- Groq Configuration -----")
print(f" API Key: {'✓' if GROQ_API_KEY else '✗'}")
print(f" Model: {GROQ_MODEL}")

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

# ===== CLIENTE EMBEDDINGS =====
# ===== CLIENTE EMBEDDINGS OPTIMIZADO =====
class AzureEmbeddingClient:
    """Cliente OPTIMIZADO para generar embeddings usando Azure OpenAI con batch processing"""
    
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION
        )
        self.model = AZURE_EMBEDDING_MODEL
        
    def get_text_embedding(self, text: str, max_length: int = 8000) -> Optional[np.ndarray]:
        """Genera embedding individual (para queries)"""
        try:
            text = text[:max_length] if len(text) > max_length else text
            response = self.client.embeddings.create(model=self.model, input=text)
            embedding = np.array(response.data[0].embedding)
            norm = np.linalg.norm(embedding)
            return embedding / norm if norm > 0 else embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def get_batch_embeddings(self, texts: List[str], max_length: int = 8000) -> Optional[List[np.ndarray]]:
        """
        OPTIMIZADO: Genera embeddings para múltiples textos en una sola llamada API
        
        Args:
            texts: Lista de textos
            max_length: Longitud máxima por texto
            
        Returns:
            Lista de vectores de embedding o None si hay error
        """
        try:
            # Truncar cada texto
            truncated_texts = [text[:max_length] for text in texts if text]
            if not truncated_texts:
                return None
            
            # Generar todos los embeddings en UNA sola llamada
            response = self.client.embeddings.create(model=self.model, input=truncated_texts)
            
            # Extraer y normalizar embeddings
            embeddings = []
            for item in response.data:
                embedding = np.array(item.embedding)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                embeddings.append(embedding)
            
            return embeddings
        except Exception as e:
            print(f"Error generating batch embeddings: {e}")
            return None
    
    def find_similar_sections_optimized(
        self, 
        query_text: str, 
        text_chunks: List[str],
        chunk_embeddings_cache: Optional[List[np.ndarray]] = None,
        top_k: int = 5
    ) -> Tuple[List[tuple], List[np.ndarray]]:
        """
        VERSIÓN OPTIMIZADA con caché de embeddings
        
        Returns:
            (resultados, chunk_embeddings) para reutilizar embeddings
        """
        
        # 1. Embedding de query
        query_embedding = self.get_text_embedding(query_text)
        if query_embedding is None:
            return [], []
        
        # 2. Reutilizar embeddings de chunks si existen
        if chunk_embeddings_cache is None:
            print(f"  Generando {len(text_chunks)} embeddings en batch...")
            chunk_embeddings = self.get_batch_embeddings(text_chunks)
            if chunk_embeddings is None:
                return [], []
        else:
            chunk_embeddings = chunk_embeddings_cache
        
        # 3. Calcular similaridades (operación local, muy rápida)
        similarities = []


# Inicialización del cliente
chat_client = ChatClient()
embedding_client = AzureEmbeddingClient()

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
    """
    FUNCIÓN MEJORADA: Extrae datos financieros con patrones específicos y validación robusta
    
    Mejoras implementadas:
    - Patrones más específicos que evitan capturar años o índices
    - Validación de valores mínimos (>10) para filtrar ruido
    - Mejor manejo de formatos numéricos europeos
    - Priorización de patrones (los más específicos primero)
    """
    
    # Estructura de datos con todas las categorías
    extracted_data = {
        'years_found': [],
        'net_interest_income': [],
        'interest_income': [],
        'interest_expense': [],
        'fee_commission_income': [],
        'commission_income': [],
        'commission_expense': [],
        'operating_expenses': [],
        'staff_costs': [],
        'provisions': [],
        'net_profit': [],
        'total_income': [],
        'net_trading_income': [],
    }
    
    # Buscar años específicos (sin confundirlos con valores financieros)
    years = re.findall(r'\b(20\d{2})\b', text)
    extracted_data['years_found'] = list(set(years))
    print(f"DEBUG: Años encontrados: {sorted(extracted_data['years_found'])}")
    
    # PATRONES MEJORADOS Y PRIORITIZADOS
    # Los patrones están ordenados de más específico a más general
    # El primero que coincida se usa (con break)
    
    patterns = {
        'interest_income': [
            # Patrón 1: Más específico - con "and similar"
            r'interest\s+and\s+similar\s+income[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
            # Patrón 2: Con contexto de miles/millones
            r'interest\s+income\s+(\d{1,3}(?:[.,]\d{3})*)\s*(?:miles|thousand|million)',
            # Patrón 3: Español
            r'ingresos\s+por\s+intereses[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
            r'intereses\s+y\s+rendimientos\s+similares[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
            # Patrón 4: General pero con límite (al menos 3 dígitos)
            r'interest\s+income[:\s]+(\d{3,}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
        ],
        
        'interest_expense': [
            r'interest\s+and\s+similar\s+expense[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
            r'interest\s+expense[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
            r'gastos\s+por\s+intereses[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
            r'interest\s+and\s+similar\s+charges[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
        ],
        
        'net_interest_income': [
            r'net\s+interest\s+income[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
            r'margen\s+(?:de\s+)?intereses[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
            r'margen\s+neto\s+(?:de\s+)?intereses[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
        ],
        
        'commission_income': [
            r'fee\s+and\s+commission\s+income[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
            r'commission\s+income[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
            r'ingresos\s+por\s+comisiones[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
        ],
        
        'commission_expense': [
            r'fee\s+and\s+commission\s+expense[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
            r'commission\s+expense[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
            r'gastos\s+por\s+comisiones[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
        ],
        
        'fee_commission_income': [
            # Alias para commission_income (se combinarán después)
            r'comisiones\s+netas[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
            r'fee.*commission.*income[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
        ],
        
        'net_trading_income': [
            r'net\s+trading\s+income[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
            r'trading\s+income[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
            r'ingresos\s+por\s+operaciones\s+financieras[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
        ],
        
        'operating_expenses': [
            r'operating\s+expenses[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
            r'gastos\s+(?:de\s+)?explotaci[oó]n[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
            r'gastos\s+operativos[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
            r'administrative\s+expenses[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
            r'gastos\s+(?:generales\s+)?(?:de\s+)?administraci[oó]n[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
        ],
        
        'staff_costs': [
            r'staff\s+costs[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
            r'personnel\s+expenses[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
            r'gastos\s+de\s+personal[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
            r'sueldos\s+y\s+salarios[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
        ],
        
        'provisions': [
            r'(?:provisions?|dotaciones?)\s+(?:for\s+)?(?:credit\s+)?(?:losses?|provisiones?)[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
            r'loan\s+loss\s+provisions?[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
            r'impairment\s+losses?[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
            r'expected\s+credit\s+losses?[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
            r'dotaciones?\s+(?:para\s+)?(?:insolvencias?|provisiones?)[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
        ],
        
        'net_profit': [
            r'(?:net\s+)?profit\s+(?:for\s+the\s+year|after\s+tax)[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
            r'net\s+profit[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
            r'beneficio\s+neto[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
            r'resultado\s+(?:neto|del\s+ejercicio)[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
            r'beneficio\s+atribuido[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
        ],
        
        'total_income': [
            r'total\s+(?:operating\s+)?income[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
            r'total\s+revenue[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
            r'(?:ingresos?|margen)\s+(?:totales?|bruto)[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
            r'margen\s+(?:de\s+)?intermediaci[oó]n[:\s]+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
        ],
    }
    
    # EXTRACCIÓN CON VALIDACIÓN MEJORADA
    for category, pattern_list in patterns.items():
        values = []
        raw_matches = []
        
        for pattern in pattern_list:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            if matches:
                raw_matches.extend(matches[:5])  # Guardar primeros 5 para debug
                
                for match in matches:
                    try:
                        # Conversión segura
                        clean_number = convert_string_to_float(match)
                        
                        # 🆕 VALIDACIÓN CRÍTICA: Filtrar valores pequeños
                        if clean_number is not None:
                            # Solo aceptar valores >= 50 (evita años, índices, porcentajes)
                            if clean_number >= 50:
                                values.append(clean_number)
                            else:
                                print(f"  Valor descartado ({category}): {clean_number} (demasiado pequeño)")
                    
                    except Exception as e:
                        print(f"  Error convirtiendo '{match}' para {category}: {e}")
                        continue
                
                # Si encontró valores válidos con este patrón, no probar los siguientes
                if values:
                    break
        
        # Remover duplicados manteniendo orden
        unique_values = []
        seen = set()
        for v in values:
            if v not in seen:
                unique_values.append(v)
                seen.add(v)
        
        extracted_data[category] = unique_values
        
        # Debug mejorado
        if raw_matches:
            print(f"🔍 DEBUG {category}:")
            print(f"   Raw matches: {raw_matches[:3]}")
            print(f"   Converted: {unique_values[:3]}")
            print(f"   Total valid: {len(unique_values)}")
    
    # COMBINACIÓN DE CATEGORÍAS RELACIONADAS
    # Combinar commission_income y fee_commission_income
    if extracted_data['commission_income'] or extracted_data['fee_commission_income']:
        all_commissions = extracted_data['commission_income'] + extracted_data['fee_commission_income']
        unique_commissions = list(set(all_commissions))
        extracted_data['fee_commission_income'] = unique_commissions
        extracted_data['commission_income'] = unique_commissions
    
    # Resumen de categorías extraídas
    categories_with_data = sum(1 for v in extracted_data.values() if v and isinstance(v, list) and len(v) > 0)
    print(f"\n{'='*60}")
    print(f" RESUMEN DE EXTRACCIÓN")
    print(f"{'='*60}")
    print(f"Categorías con datos: {categories_with_data}/12")
    
    for cat, vals in extracted_data.items():
        if vals and cat != 'years_found':
            print(f"   {cat}: {len(vals)} valores - {vals[:2]}")
    print(f"{'='*60}\n")
    
    return extracted_data


def calculate_financial_ratios(data: Dict[str, List[float]]) -> Dict[str, float]:
    """FUNCIÓN MEJORADA: Calcular ratios con validación de tipos segura + alertas de anomalías"""
    
    ratios = {}
    
    if not data or not isinstance(data, dict):
        return ratios
    
    # Función auxiliar para obtener valor máximo seguro
    def get_max_value_safe(values_list):
        if not values_list or not isinstance(values_list, list):
            return 0.0
        try:
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
    
    # Extraer también interest_expense para calcular NIM
    interest_income = get_max_value_safe(data.get('interest_income', []))
    interest_expense = get_max_value_safe(data.get('interest_expense', []))
    
    print(f" DEBUG ratios - net_profit: {net_profit}, total_income: {total_income}")
    print(f" DEBUG ratios - operating_expenses: {operating_expenses}")
    
    # ========== CALCULAR VALORES FALTANTES SI ES POSIBLE ==========
    print(f"\n{'='*60}")
    print(f" CALCULANDO VALORES FALTANTES")
    print(f"{'='*60}")
    
    # 1. Calcular net_interest_income si falta
    if net_interest_income == 0 and interest_income > 0:
        net_interest_income = interest_income - interest_expense
        print(f" Net Interest Income calculado: €{net_interest_income:,.0f}")
        print(f"   (Interest income €{interest_income:,.0f} - Interest expense €{interest_expense:,.0f})")
    
    # 2. Calcular total_income si falta
    if total_income == 0:
        trading_income = get_max_value_safe(data.get('net_trading_income', []))
        total_income = net_interest_income + fee_commission + trading_income
        
        if total_income > 0:
            print(f" Total Income estimado: €{total_income:,.0f}")
            print(f"   Componentes:")
            print(f"   - Net Interest Income: €{net_interest_income:,.0f}")
            print(f"   - Fee & Commission: €{fee_commission:,.0f}")
            print(f"   - Trading Income: €{trading_income:,.0f}")
        else:
            print(f" No se puede calcular Total Income (datos insuficientes)")
    else:
        print(f" Total Income extraído del documento: €{total_income:,.0f}")
    
    # 3. Mensaje sobre operating_expenses
    if operating_expenses == 0:
        print(f" Operating Expenses no encontrado en el documento")
        print(f"   (No es crítico para análisis de partes relacionadas)")
    else:
        print(f" Operating Expenses extraído: €{operating_expenses:,.0f}")
    
    print(f"{'='*60}\n")
    
    # ========== CALCULAR RATIOS SI HAY DATOS DISPONIBLES ==========
    if total_income > 0:
        # 1. Net Profit Margin
        if net_profit > 0:
            ratios['net_profit_margin'] = (net_profit / total_income) * 100
        
        # 2. Cost-Income Ratio y Efficiency Ratio
        if operating_expenses > 0:
            ratios['cost_income_ratio'] = (operating_expenses / total_income) * 100
            ratios['efficiency_ratio'] = (operating_expenses / total_income) * 100
            
            # Validación de Cost-Income Ratio
            cir = ratios['cost_income_ratio']
            
            if cir > 75:
                print(f" ALERTA CRÍTICA: Cost-income ratio muy alto ({cir:.1f}%)")
                print(f"   Operating expenses: €{operating_expenses:,.0f}")
                print(f"   Total income: €{total_income:,.0f}")
                print(f"    Esto indica baja eficiencia operativa (>75% es problemático)")
                ratios['cost_income_ratio_flag'] = 'HIGH'
                
            elif cir > 60:
                print(f" ADVERTENCIA: Cost-income ratio por encima del óptimo ({cir:.1f}%)")
                print(f"   Rango óptimo para bancos: 45-60%")
                ratios['cost_income_ratio_flag'] = 'ABOVE_OPTIMAL'
                
            elif cir < 30:
                print(f" ALERTA: Cost-income ratio inusualmente bajo ({cir:.1f}%)")
                print(f"   Verificar si los datos de gastos están completos")
                ratios['cost_income_ratio_flag'] = 'SUSPICIOUSLY_LOW'
                
            else:
                print(f" Cost-income ratio en rango óptimo ({cir:.1f}%)")
                ratios['cost_income_ratio_flag'] = 'OPTIMAL'
        
        # 3. Interest Income Ratio
        if net_interest_income > 0:
            ratios['interest_income_ratio'] = (net_interest_income / total_income) * 100
        
        # 4. Staff Cost Ratio
        if staff_costs > 0:
            ratios['staff_cost_ratio'] = (staff_costs / total_income) * 100
        
        # 5. Fee Commission Ratio
        if fee_commission > 0:
            ratios['fee_commission_ratio'] = (fee_commission / total_income) * 100
        
        # 6. Provision Ratio
        if provisions > 0:
            ratios['provision_ratio'] = (provisions / total_income) * 100
    
    # ========== CALCULAR NET INTEREST MARGIN (NIM) ==========
    if interest_income > 0 and interest_expense >= 0:
        nim = interest_income - interest_expense
        if nim > 0 and total_income > 0:
            nim_pct = (nim / total_income) * 100
            
            # Validación de NIM
            if nim_pct > 100:
                print(f" ALERTA CRÍTICA: NIM anormalmente alto ({nim_pct:.2f}%)")
                print(f"   Esto indica error en extracción de datos:")
                print(f"   - Interest income: €{interest_income:,.0f}")
                print(f"   - Interest expense: €{interest_expense:,.0f}")
                print(f"   - Total income: €{total_income:,.0f} ← VERIFICAR")
                print(f"   NIM típico para bancos: 2-5%")
                # No guardar el ratio si es anómalo
            elif nim_pct > 10:
                print(f" NIM alto pero posible: {nim_pct:.2f}%")
                ratios['net_interest_margin_pct'] = nim_pct
                ratios['net_interest_margin_flag'] = 'HIGH'
            else:
                ratios['net_interest_margin_pct'] = nim_pct
                print(f" Net Interest Margin calculado: {nim_pct:.2f}%")
    
    # ========== VALIDAR PROFIT MARGIN ==========
    if 'net_profit_margin' in ratios:
        npm = ratios['net_profit_margin']
        if npm > 50:
            print(f" ALERTA: Net profit margin muy alto ({npm:.1f}%) - verificar datos")
            ratios['net_profit_margin_flag'] = 'SUSPICIOUSLY_HIGH'
        elif npm < 0:
            print(f" Pérdidas detectadas: Net profit margin negativo ({npm:.1f}%)")
            ratios['net_profit_margin_flag'] = 'NEGATIVE'
        elif npm > 20:
            print(f" Rentabilidad excelente: Net profit margin {npm:.1f}%")
            ratios['net_profit_margin_flag'] = 'EXCELLENT'
    
    # ========== CALCULAR VARIACIONES (GROWTH RATES) ==========
    for category, values in data.items():
        if isinstance(values, list) and len(values) >= 2:
            try:
                float_values = []
                for v in values:
                    if isinstance(v, (int, float)):
                        float_values.append(float(v))
                    elif isinstance(v, str):
                        converted = convert_string_to_float(v)
                        if converted is not None:
                            float_values.append(converted)
                
                if len(float_values) >= 2 and float_values[0] != 0:
                    growth = ((float_values[-1] - float_values[0]) / abs(float_values[0])) * 100
                    ratios[f'{category}_growth'] = growth
                    
            except Exception as e:
                print(f"⚠️ Error calculando growth para {category}: {e}")
                continue
    
    # ========== RESUMEN FINAL ==========
    print(f" DEBUG: Ratios calculados: {list(ratios.keys())}")
    
    # Resumen de validaciones
    flags_found = [k for k in ratios.keys() if k.endswith('_flag')]
    if flags_found:
        print(f" Flags de validación generados: {flags_found}")
    
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
            print(f"IncomeREACTAgent AUTÓNOMO iniciando extracción para: {pdf_path}")
            
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
            
            # Validación cruzada con Balance Agent 
            cross_validation_result = None
            try:
                balance_file = output_dir / f"balance_{pdf_file.stem}_summary.json"
                if balance_file.exists():
                    import json
                    with open(balance_file, 'r', encoding='utf-8') as f:
                        balance_result = json.load(f)
                        balance_data = balance_result.get('financial_data', {})
                    
                    if balance_data:
                        income_data = extraction_result.get("financial_data", {})
                        cross_validation_result = self.cross_validate_with_balance(income_data, balance_data)
                        
                        print(f"\n Validación cruzada completada:")
                        print(f"   Estado: {'Consistente' if cross_validation_result['consistent'] else '❌ Inconsistente'}")
                        print(f"   Ratios calculados: {list(cross_validation_result['ratios_calculated'].keys())}")
                    else:
                        print(f"Archivo balance encontrado pero sin datos financieros")
                else:
                    print(f"No se encontró archivo de balance: {balance_file}")
                    print(f"   Saltando validación cruzada (ejecutar Balance Agent primero)")
            
            except Exception as e:
                print(f" Error en validación cruzada: {e}")
                import traceback
                traceback.print_exc()
            
            # Guardar resultados 
            save_result = self.save_income_results_enhanced(
                pdf_file, output_dir, extraction_result, validation_result,
                cross_validation_result  
            )
            
            # GENERAR RESPUESTA ESPECÍFICA MEJORADA CON DEBUGGING
            print(f" DEBUG: Iniciando generación de respuesta específica...")
            if question:
                print(f" Pregunta específica recibida: {question}")
            
            specific_answer = self.generate_enhanced_income_analysis_fixed(question, extraction_result, validation_result)
            print(f" DEBUG: Respuesta generada con {len(specific_answer)} caracteres")
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            print(" Income extraction completed successfully (AUTÓNOMO)")
            
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
                    "cross_validation": cross_validation_result,
                    "confidence": validation_result.get("confidence", 0.8),
                    "quality": validation_result.get("quality", "unknown")
                }
            }
            
        except Exception as e:
            print(f" Error en IncomeREACTAgent: {str(e)}")
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
        """NUEVA FUNCIÓN: Extracción mejorada de datos de cuenta de resultados con búsqueda semántica"""
        try:
            print(f"Extrayendo cuenta de resultados de: {pdf_file}")
            
            # Páginas más probables para cuenta de resultados en documentos bancarios
            with fitz.open(pdf_file) as pdf:
                total_pdf_pages = len(pdf)

            target_pages = list(range(1, min(10, total_pdf_pages))) 
                        
            extracted_text = ""
            total_chars = 0
            financial_data = {}
            relevant_pages = []
            all_text_chunks = []  # NUEVO: Para búsqueda semántica
            
            with fitz.open(pdf_file) as pdf:
                for page_num in range(len(pdf)):  # Procesar todo el documento
                # O implementar búsqueda inteligente:
                    if page_num < 20 or relevance_score > 5:  # Primeras 20 + páginas relevantes
                # procesar página
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
                        
                        # NUEVO: Almacenar chunks para búsqueda semántica posterior
                        page_chunks = self._split_text_into_chunks(text, chunk_size=1000)
                        all_text_chunks.extend([
                            {'page': page_num + 1, 'chunk': chunk, 'score': relevance_score}
                            for chunk in page_chunks
                        ])
                        
                        print(f" Página {page_num + 1}: {len(text)} caracteres extraídos (relevance: {relevance_score})")
                        
                        # NUEVA: Extracción de datos financieros específicos
                        page_financial_data = extract_comprehensive_income_data(text)
                        for key, values in page_financial_data.items():
                            if key not in financial_data:
                                financial_data[key] = []
                            financial_data[key].extend(values)
            
            # Si no se encontró contenido relevante, extraer páginas por defecto
            if total_chars < 1000:
                print(" Poco contenido relevante encontrado, extrayendo páginas por defecto...")
                with fitz.open(pdf_file) as pdf:
                    for page_num in range(min(10, len(pdf))):
                        page = pdf[page_num]
                        text = page.get_text()
                        extracted_text += f"\n=== PÁGINA {page_num + 1} (DEFAULT) ===\n{text}"
                        total_chars += len(text)
                        
                        # NUEVO: También añadir estos chunks
                        page_chunks = self._split_text_into_chunks(text, chunk_size=1000)
                        all_text_chunks.extend([
                            {'page': page_num + 1, 'chunk': chunk, 'score': 0}
                            for chunk in page_chunks
                        ])
            
            print(f" Texto total extraído: {total_chars} caracteres de {len(relevant_pages)} páginas")

            confidence = 0.8  # Valor por defecto
            language = detect_language(extracted_text) if extracted_text else "unknown"
            
            # ===== BÚSQUEDA SEMÁNTICA OPTIMIZADA CON BATCH EMBEDDINGS =====
            semantic_results = {}
            if all_text_chunks:
                print(f" Aplicando búsqueda semántica optimizada ({len(all_text_chunks)} chunks)...")
                
                chunk_texts = [item['chunk'] for item in all_text_chunks]
                
                # OPTIMIZADO: Generar embeddings de chunks en BATCH CON RETRY LOGIC
                print(f" Generando embeddings en batch...")

                max_retries = 3
                retry_delay = 65  # Segundos
                chunk_embeddings = None

                for attempt in range(max_retries):
                    try:
                        chunk_embeddings = embedding_client.get_batch_embeddings(chunk_texts)
                        print(f" ✓ {len(chunk_embeddings)} embeddings generados en batch")
                        break  # Éxito, salir
                        
                    except Exception as e:
                        error_str = str(e)
                        
                        if '429' in error_str and attempt < max_retries - 1:
                            print(f" ⚠️ Rate limit reached (Error 429)")
                            print(f" ⏳ Esperando {retry_delay}s antes del reintento {attempt + 2}/{max_retries}...")
                            import time
                            time.sleep(retry_delay)
                        else:
                            print(f" ❌ Error en batch embeddings: {error_str}")
                            break

                if chunk_embeddings is None:
                    print(" Error generando embeddings, saltando búsqueda semántica")

                else:
                    print(f"   {len(chunk_embeddings)} embeddings generados en batch")
                    
                    # Queries semánticas para encontrar secciones relevantes de cuenta de resultados
                    semantic_queries = {
                        'interest_income': "margen de intereses e ingresos financieros netos del banco",
                        'commissions': "ingresos por comisiones y servicios bancarios prestados",
                        'operating_expenses': "gastos operativos de personal y administración",
                        'provisions': "provisiones para riesgos crediticios y deterioro de activos",
                        'net_profit': "beneficio neto resultado del ejercicio después de impuestos",
                        'margins': "rentabilidad márgenes de intermediación y eficiencia operativa"
                    }
                    
                    for category, query in semantic_queries.items():
                        try:
                            # OPTIMIZADO: Pasar embeddings cacheados para reutilizarlos
                            similar_sections, _ = embedding_client.find_similar_sections_optimized(
                                query, 
                                chunk_texts,
                                chunk_embeddings_cache=chunk_embeddings,  # Reutilizar embeddings
                                top_k=3
                            )
                            
                            relevant_chunks = []
                            for idx, score, chunk_text in similar_sections:
                                if score >= 0.65:  # Umbral de similaridad
                                    chunk_info = all_text_chunks[idx]
                                    relevant_chunks.append({
                                        'score': score,
                                        'text': chunk_text,
                                        'page': chunk_info['page'],
                                        'relevance_score': chunk_info['score']
                                    })
                                    print(f"  {category} (score {score:.2f}, page {chunk_info['page']})")
                            
                            if relevant_chunks:
                                semantic_results[category] = relevant_chunks
                        
                        except Exception as e:
                            print(f"  Error en búsqueda semántica para {category}: {e}")
                    
                    # Enriquecer texto extraído con chunks semánticamente relevantes
                    if semantic_results:
                        print(f"Categorías encontradas con embeddings: {len(semantic_results)}")
                        enriched_text = "\n\n=== SECCIONES RELEVANTES (BÚSQUEDA SEMÁNTICA) ===\n"
                        
                        for category, chunks in semantic_results.items():
                            enriched_text += f"\n--- {category.upper()} ---\n"
                            for chunk_data in chunks[:2]:  # Top 2 por categoría
                                enriched_text += f"[Página {chunk_data['page']}, Score: {chunk_data['score']:.2f}]\n"
                                enriched_text += chunk_data['text'][:500] + "...\n\n"
                        
                        # Prefijar al texto original
                        extracted_text = enriched_text + "\n\n=== TEXTO COMPLETO EXTRAÍDO ===\n" + extracted_text[:5000]
            
            # Integrar transacciones con partes relacionadas
            related_party_data = self.extract_related_party_transactions(pdf_file)
            if related_party_data:
                print(f" Datos de partes relacionadas extraídos exitosamente")
                
                # Combinar con financial_data existente eliminando duplicados
                for key, values in related_party_data.items():
                    if values:  # Solo si hay valores
                        if key in financial_data:
                            # Convertir a set para detectar duplicados
                            existing_set = set(financial_data[key])
                            new_values = [v for v in values if v not in existing_set]
                            
                            if new_values:
                                financial_data[key].extend(new_values)
                                print(f"   {key}: +{len(new_values)} valores nuevos agregados")
                                print(f"      Total ahora: {len(financial_data[key])} valores únicos")
                            else:
                                print(f"   {key}: todos los valores ya existían (duplicados omitidos)")
                        else:
                            # Nueva categoría
                            financial_data[key] = values
                            print(f"   {key}: {len(values)} valores nuevos")

            else:
                print(f" No se encontraron datos de partes relacionadas")

            
            # Ajustar confianza si se encontraron resultados semánticos
            if semantic_results:
                confidence = min(1.0, confidence + 0.1)  # Bonus por búsqueda semántica exitosa
            
            return {
                "success": True,
                "text": extracted_text,
                "total_characters": total_chars,
                "pages_processed": relevant_pages,
                "financial_data": financial_data,
                "confidence": confidence,
                "language": language,
                "semantic_results": semantic_results,  
                "semantic_categories_found": len(semantic_results),  # NUEVO
                "total_chunks_analyzed": len(all_text_chunks)  # NUEVO
            }
            
        except Exception as e:
            print(f" Error en extract_income_data_enhanced: {e}")
            return {"success": False, "error": str(e)}    

    def extract_related_party_transactions(self, pdf_file: Path) -> Dict[str, Any]:
        """
        Extrae transacciones con partes relacionadas de la Nota 2
        Específico para documentos como GarantiBank que detallan estas transacciones
        """
        try:
            related_party_data = {
                'interest_income': [],
                'interest_expense': [],
                'commission_income': [],
                'commission_expense': [],
            }
            
            with fitz.open(pdf_file) as pdf:
                for page_num in range(len(pdf)):
                    page = pdf[page_num]
                    text = page.get_text()
                    text_lower = normalize_text(text)
                    
                    # Buscar sección de Related Party Disclosures
                    if 'related party' in text_lower or 'partes vinculadas' in text_lower:
                        
                        # ========== PATRONES MEJORADOS (NUEVO) ==========
                        
                        # Interest income
                        interest_income_patterns = [
                            r'interest\s+and\s+similar\s+income\s+([0-9.,]+)',
                            r'interest\s+income\s+([0-9.,]+)',
                            r'ingresos\s+por\s+intereses\s+([0-9.,]+)',
                        ]
                        for pattern in interest_income_patterns:
                            matches = re.findall(pattern, text_lower, re.IGNORECASE)
                            for match in matches:
                                value = convert_string_to_float(match)
                                if value and value > 0:
                                    related_party_data['interest_income'].append(value)
                        
                        # Interest expense
                        interest_expense_patterns = [
                            r'interest\s+and\s+similar\s+expense\s+([0-9.,]+)',
                            r'interest\s+expense\s+([0-9.,]+)',
                            r'gastos\s+por\s+intereses\s+([0-9.,]+)',
                        ]
                        for pattern in interest_expense_patterns:
                            matches = re.findall(pattern, text_lower, re.IGNORECASE)
                            for match in matches:
                                value = convert_string_to_float(match)
                                if value and value > 0:
                                    related_party_data['interest_expense'].append(value)
                        
                        # Commission income
                        commission_income_patterns = [
                            r'fee\s+and\s+commission\s+income\s+([0-9.,]+)',
                            r'commission\s+income\s+([0-9.,]+)',
                            r'ingresos\s+por\s+comisiones\s+([0-9.,]+)',
                        ]
                        for pattern in commission_income_patterns:
                            matches = re.findall(pattern, text_lower, re.IGNORECASE)
                            for match in matches:
                                value = convert_string_to_float(match)
                                if value and value > 0:
                                    related_party_data['commission_income'].append(value)
                        
                        # Commission expense
                        commission_expense_patterns = [
                            r'fee\s+and\s+commission\s+expense\s+([0-9.,]+)',
                            r'commission\s+expense\s+([0-9.,]+)',
                            r'gastos\s+por\s+comisiones\s+([0-9.,]+)',
                        ]
                        for pattern in commission_expense_patterns:
                            matches = re.findall(pattern, text_lower, re.IGNORECASE)
                            for match in matches:
                                value = convert_string_to_float(match)
                                if value and value > 0:
                                    related_party_data['commission_expense'].append(value)
            
            return related_party_data
            
        except Exception as e:
            print(f"Error extrayendo transacciones con partes relacionadas: {e}")
            return {}

    def _split_text_into_chunks(self, text: str, chunk_size: int = 1000) -> List[str]:
        """
        Divide texto en chunks para procesamiento con embeddings
        
        Args:
            text: Texto a dividir
            chunk_size: Tamaño aproximado de cada chunk en caracteres
            
        Returns:
            Lista de chunks de texto
        """
        if not text:
            return []
        
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 por el espacio
            
            # Si agregar la palabra excede el límite y ya hay contenido, crear nuevo chunk
            if current_length + word_length > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            
            current_chunk.append(word)
            current_length += word_length
        
        # Agregar el último chunk si existe
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def validate_income_data_enhanced(self, extraction: Dict[str, Any]) -> Dict[str, Any]:
        """FUNCIÓN MEJORADA: Validación avanzada de datos de cuenta de resultados con scoring granular"""
        try:
            text = extraction.get("text", "")
            confidence = extraction.get("confidence", 0.0)
            financial_data = extraction.get("financial_data", {})
            
            # Sistema de scoring más granular y específico
            quality_score = 0
            max_possible_score = 130  # Aumentado para incluir nuevas categorías
            validation_details = []
            
            text_lower = normalize_text(text)
            
            # ========== CATEGORÍAS PRINCIPALES (Peso: 80 puntos) ==========
            
            # 1. Ingresos por intereses (15 puntos)
            if any(normalize_text(term) in text_lower for term in ["interest income", "margen intereses", "ingresos intereses"]):
                quality_score += 15
                validation_details.append(" Ingresos por intereses encontrados")
            
            # 2. Gastos por intereses (10 puntos) - crítico para NIM
            if any(normalize_text(term) in text_lower for term in ["interest expense", "gastos intereses", "interest charges"]):
                quality_score += 10
                validation_details.append(" Gastos por intereses encontrados")
            
            # 3. Comisiones (15 puntos)
            if any(normalize_text(term) in text_lower for term in ["commission", "comisiones", "fee income"]):
                quality_score += 15
                validation_details.append(" Ingresos por comisiones encontrados")
            
            # 4. Gastos operativos (20 puntos) - muy importante
            if any(normalize_text(term) in text_lower for term in ["operating expenses", "gastos explotación", "gastos operativos"]):
                quality_score += 20
                validation_details.append(" Gastos operativos encontrados")
            
            # 5. Gastos de personal (15 puntos)
            if any(normalize_text(term) in text_lower for term in ["staff costs", "gastos personal", "personnel expenses"]):
                quality_score += 15
                validation_details.append(" Gastos de personal encontrados")
            
            # 6. Provisiones (10 puntos)
            if any(normalize_text(term) in text_lower for term in ["provisions", "provisiones", "impairment"]):
                quality_score += 10
                validation_details.append(" Provisiones encontradas")
            
            # 7. Beneficio neto (15 puntos)
            if any(normalize_text(term) in text_lower for term in ["net profit", "beneficio neto", "net income"]):
                quality_score += 15
                validation_details.append(" Beneficio neto encontrado")
            
            # ========== DATOS FINANCIEROS ESPECÍFICOS (Peso: 20 puntos) ==========
            
            categories_with_data = 0
            if financial_data:
                categories_with_data = sum(1 for values in financial_data.values() if values)
                
                # Bonificación progresiva por categorías con datos
                if categories_with_data >= 6:
                    data_bonus = 20
                    validation_details.append(f" Excelente cobertura de datos: {categories_with_data} categorías")
                elif categories_with_data >= 4:
                    data_bonus = 15
                    validation_details.append(f" Buena cobertura de datos: {categories_with_data} categorías")
                elif categories_with_data >= 2:
                    data_bonus = 10
                    validation_details.append(f" Cobertura moderada de datos: {categories_with_data} categorías")
                else:
                    data_bonus = 5
                    validation_details.append(f" Cobertura limitada de datos: {categories_with_data} categorías")
                
                quality_score += data_bonus
            
            # ========== BONIFICACIONES ADICIONALES (Peso: 30 puntos) ==========
            
            # 8. Datos multi-año (20 puntos) - muy valioso para análisis comparativo
            has_multi_year = False
            multi_year_categories = 0
            
            if financial_data:
                for category, values in financial_data.items():
                    if isinstance(values, list) and len(values) >= 2:
                        # Verificar que los valores sean diferentes (no duplicados)
                        unique_values = set(values)
                        if len(unique_values) >= 2:
                            multi_year_categories += 1
                
                if multi_year_categories >= 3:
                    quality_score += 20
                    has_multi_year = True
                    validation_details.append(f"Datos comparativos multi-año: {multi_year_categories} categorías con múltiples períodos")
                elif multi_year_categories >= 1:
                    quality_score += 10
                    has_multi_year = True
                    validation_details.append(f"Datos parciales multi-año: {multi_year_categories} categorías")
            
            # Transacciones con partes relacionadas (10 puntos)
            has_related_party_data = False
            if any(normalize_text(term) in text_lower for term in ["related party", "partes vinculadas", "partes relacionadas"]):
                quality_score += 10
                has_related_party_data = True
                validation_details.append("Información de partes relacionadas detectada")
            
            # ========== CÁLCULO DE DATA COMPLETENESS ==========
            
            # Calcular completitud de datos como porcentaje
            data_completeness = quality_score / max_possible_score
            
            # ========== VALIDACIÓN DE RATIOS Y PENALIZACIONES ==========
            
            # 🆕 NUEVO: Calcular ratios y aplicar penalizaciones por anomalías
            ratios_penalty = 0
            has_ratio_warnings = False
            
            if financial_data:
                try:
                    # Importar la función de cálculo de ratios
                    ratios = calculate_financial_ratios(financial_data)
                    
                    # Verificar flags de ratios anómalos
                    if ratios.get('cost_income_ratio_flag') == 'HIGH':
                        ratios_penalty += 15  # Penalización fuerte
                        has_ratio_warnings = True
                        validation_details.append("Cost-income ratio anormalmente alto (>75%)")
                        
                    elif ratios.get('cost_income_ratio_flag') == 'SUSPICIOUSLY_LOW':
                        ratios_penalty += 10
                        has_ratio_warnings = True
                        validation_details.append("Cost-income ratio sospechosamente bajo (<30%)")
                    
                    if ratios.get('net_profit_margin_flag') == 'SUSPICIOUSLY_HIGH':
                        ratios_penalty += 10
                        has_ratio_warnings = True
                        validation_details.append("Margen de beneficio sospechosamente alto (>50%)")
                    
                    # Aplicar penalización
                    quality_score = max(0, quality_score - ratios_penalty)
                    
                    if ratios_penalty > 0:
                        validation_details.append(f"🔻 Penalización por ratios anómalos: -{ratios_penalty} puntos")
                    
                except Exception as e:
                    print(f"No se pudieron calcular ratios para validación: {e}")
            
            # ========== DETERMINACIÓN DE CALIDAD ==========
            
            # Ajustar umbrales considerando el nuevo max_possible_score
            quality_percentage = (quality_score / max_possible_score) * 100
            
            if quality_percentage >= 70:
                quality = "excellent"
            elif quality_percentage >= 55:
                quality = "good"
            elif quality_percentage >= 35:
                quality = "fair"
            else:
                quality = "poor"
            
            # ========== CÁLCULO DE CONFIANZA FINAL MEJORADO ==========
            
            # Fórmula ponderada más sofisticada
            # - 40% confianza base del agente (basada en embeddings/relevancia)
            # - 60% completitud de datos (basada en scoring)
            
            base_confidence = confidence
            completeness_factor = data_completeness
            
            # Componente de completitud ponderado
            weighted_completeness = completeness_factor * 0.6
            
            # Componente de confianza base ponderado
            weighted_base = base_confidence * 0.4
            
            # Confianza combinada
            combined_confidence = weighted_base + weighted_completeness
            
            # Bonificaciones adicionales a la confianza
            confidence_bonuses = 0.0
            
            if has_multi_year:
                confidence_bonuses += 0.05  # +5% por datos comparativos
                
            if has_related_party_data:
                confidence_bonuses += 0.03  # +3% por datos de partes relacionadas
            
            # Penalizaciones a la confianza
            confidence_penalties = 0.0
            
            if has_ratio_warnings:
                confidence_penalties += 0.15  # -15% si hay ratios anómalos
            
            if categories_with_data < 3:
                confidence_penalties += 0.10  # -10% si hay muy pocas categorías
            
            # Aplicar bonificaciones y penalizaciones
            final_confidence = combined_confidence + confidence_bonuses - confidence_penalties
            
            # Asegurar que esté en rango [0, 1]
            final_confidence = max(0.0, min(1.0, final_confidence))
            
            # ========== LOGGING DETALLADO ==========
            
            print(f"\n{'='*60}")
            print(f"VALIDACIÓN DE DATOS - INCOME STATEMENT")
            print(f"{'='*60}")
            print(f"Calidad: {quality.upper()} ({quality_percentage:.1f}% de completitud)")
            print(f"Puntuación: {quality_score}/{max_possible_score} puntos")
            print(f"Confianza base: {base_confidence:.3f}")
            print(f"Completitud de datos: {completeness_factor:.3f}")
            print(f"Categorías con datos: {categories_with_data}")
            
            if confidence_bonuses > 0:
                print(f"Bonificaciones aplicadas: +{confidence_bonuses:.3f}")
            
            if confidence_penalties > 0:
                print(f"Penalizaciones aplicadas: -{confidence_penalties:.3f}")
            
            print(f"Confianza final: {final_confidence:.3f}")
            print(f"{'='*60}\n")
            
            # ========== RETURN MEJORADO ==========
            
            return {
                "success": True,
                "quality": quality,
                "confidence": final_confidence,
                "score": quality_score,
                "max_score": max_possible_score,
                "quality_percentage": quality_percentage,
                "details": validation_details,
                "financial_categories_found": categories_with_data,
                "data_completeness": completeness_factor,
                "has_comparative_data": has_multi_year,
                "has_related_party_info": has_related_party_data,
                "has_ratio_warnings": has_ratio_warnings,
                "confidence_breakdown": {
                    "base": base_confidence,
                    "completeness": completeness_factor,
                    "bonuses": confidence_bonuses,
                    "penalties": confidence_penalties,
                    "final": final_confidence
                },
                "multi_year_categories": multi_year_categories,
                "ratios_penalty_applied": ratios_penalty
            }
            
        except Exception as e:
            print(f"Error en validación: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "quality": "poor",
                "confidence": 0.0,
                "score": 0
            }
        
    def cross_validate_with_balance(self, income_data: Dict, balance_data: Dict) -> Dict:
        """
        Valida consistencia entre cuenta de resultados y balance
        
        Args:
            income_data: Diccionario con datos financieros del Income Agent
                        Formato esperado: {'net_profit': [1250], 'total_income': [5000], ...}
            balance_data: Diccionario con datos del Balance Agent
                         Formato esperado: {'total_assets': [62562], 'total_liabilities': [9458], ...}
        
        Returns:
            Diccionario con resultados de validación cruzada
        """
        validation_results = {
            'consistent': True,
            'warnings': [],
            'cross_checks': [],
            'ratios_calculated': {}
        }
        
        print(f"\n{'='*60}")
        print(f"VALIDACIÓN CRUZADA: Income Statement ↔ Balance Sheet")
        print(f"{'='*60}")
        
        try:
            # Función auxiliar para obtener valor máximo seguro
            def get_safe_value(data_dict, key):
                if not data_dict or key not in data_dict:
                    return None
                values = data_dict[key]
                if not values or not isinstance(values, list):
                    return None
                # Obtener el valor más reciente (último en la lista)
                return max(values) if values else None
            
            # ========== CHECK 1: ROA (Return on Assets) ==========
            total_assets = get_safe_value(balance_data, 'total_assets')
            net_profit = get_safe_value(income_data, 'net_profit')
            
            if total_assets and net_profit:
                if net_profit > total_assets:
                    validation_results['warnings'].append(
                        f" CRÍTICO: Beneficio neto (€{net_profit:,.0f}) > Activos totales (€{total_assets:,.0f})"
                    )
                    validation_results['consistent'] = False
                    print(f" Inconsistencia detectada: profit > assets")
                else:
                    roa = (net_profit / total_assets) * 100
                    validation_results['ratios_calculated']['ROA'] = roa
                    validation_results['cross_checks'].append(
                        f" ROA calculado: {roa:.2f}%"
                    )
                    print(f" ROA: {roa:.2f}%")
                    
                    # Validar rango razonable de ROA para bancos (típicamente 0.5-2%)
                    if roa < 0:
                        validation_results['warnings'].append(
                            f" ROA negativo ({roa:.2f}%) - banco en pérdidas"
                        )
                    elif roa > 3:
                        validation_results['warnings'].append(
                            f" ROA muy alto ({roa:.2f}%) - verificar datos"
                        )
            else:
                missing = []
                if not total_assets:
                    missing.append("total_assets")
                if not net_profit:
                    missing.append("net_profit")
                validation_results['warnings'].append(
                    f"No se puede calcular ROA - faltan: {', '.join(missing)}"
                )
                print(f" ROA no calculado - datos insuficientes")
            
            # ========== CHECK 2: ROE (Return on Equity) ==========
            total_equity = get_safe_value(balance_data, 'total_equity')
            
            if total_equity and net_profit:
                if net_profit > total_equity * 2:  # Muy inusual
                    validation_results['warnings'].append(
                        f" Beneficio neto muy alto vs patrimonio - verificar"
                    )
                
                roe = (net_profit / total_equity) * 100
                validation_results['ratios_calculated']['ROE'] = roe
                validation_results['cross_checks'].append(
                    f" ROE calculado: {roe:.2f}%"
                )
                print(f" ROE: {roe:.2f}%")
                
                # Validar rango razonable de ROE para bancos (típicamente 8-15%)
                if roe < 0:
                    validation_results['warnings'].append(
                        f" ROE negativo ({roe:.2f}%) - rentabilidad negativa"
                    )
                elif roe > 25:
                    validation_results['warnings'].append(
                        f" ROE muy alto ({roe:.2f}%) - verificar datos"
                    )
            
            # ========== CHECK 3: Net Interest Margin vs Earning Assets ==========
            net_interest_income = get_safe_value(income_data, 'net_interest_income')
            loans_to_customers = get_safe_value(balance_data, 'loans_to_customers')
            
            if net_interest_income and loans_to_customers and loans_to_customers > 0:
                nim_on_loans = (net_interest_income / loans_to_customers) * 100
                validation_results['ratios_calculated']['NIM_on_loans'] = nim_on_loans
                validation_results['cross_checks'].append(
                    f" NIM sobre préstamos: {nim_on_loans:.2f}%"
                )
                print(f" NIM/Loans: {nim_on_loans:.2f}%")
                
                # Validar rango razonable (típicamente 2-5% para bancos)
                if nim_on_loans > 10:
                    validation_results['warnings'].append(
                        f" NIM sobre préstamos muy alto ({nim_on_loans:.2f}%) - verificar"
                    )
                elif nim_on_loans < 0:
                    validation_results['warnings'].append(
                        f" NIM sobre préstamos negativo - posible error"
                    )
            
            # ========== CHECK 4: Provisiones vs Préstamos ==========
            provisions = get_safe_value(income_data, 'provisions')
            
            if provisions and loans_to_customers and loans_to_customers > 0:
                provision_ratio = (provisions / loans_to_customers) * 100
                validation_results['ratios_calculated']['provision_ratio'] = provision_ratio
                validation_results['cross_checks'].append(
                    f" Ratio de provisiones sobre préstamos: {provision_ratio:.2f}%"
                )
                print(f" Provisions/Loans: {provision_ratio:.2f}%")
                
                # Validar rango razonable (típicamente <2% en condiciones normales)
                if provision_ratio > 5:
                    validation_results['warnings'].append(
                        f" Provisiones muy altas ({provision_ratio:.2f}%) - alta morosidad"
                    )
            
            # ========== CHECK 5: Leverage Ratio ==========
            total_liabilities = get_safe_value(balance_data, 'total_liabilities')
            
            if total_assets and total_equity and total_equity > 0:
                leverage_ratio = total_assets / total_equity
                validation_results['ratios_calculated']['leverage_ratio'] = leverage_ratio
                validation_results['cross_checks'].append(
                    f" Ratio de apalancamiento: {leverage_ratio:.2f}x"
                )
                print(f" Leverage: {leverage_ratio:.2f}x")
                
                # Validar rango razonable para bancos (típicamente 10-20x)
                if leverage_ratio > 30:
                    validation_results['warnings'].append(
                        f" Apalancamiento muy alto ({leverage_ratio:.2f}x) - riesgo elevado"
                    )
                elif leverage_ratio < 5:
                    validation_results['warnings'].append(
                        f" Apalancamiento muy bajo ({leverage_ratio:.2f}x) - verificar datos"
                    )
            
            # ========== CHECK 6: Ecuación Contable Fundamental ==========
            if total_assets and total_liabilities and total_equity:
                expected_assets = total_liabilities + total_equity
                difference = abs(total_assets - expected_assets)
                tolerance = total_assets * 0.01  # 1% de tolerancia
                
                if difference > tolerance:
                    validation_results['warnings'].append(
                        f" CRÍTICO: Ecuación contable no balancea - "
                        f"Activos: €{total_assets:,.0f}, Pasivos+Patrimonio: €{expected_assets:,.0f}"
                    )
                    validation_results['consistent'] = False
                    print(f" Balance sheet no cuadra: diferencia de €{difference:,.0f}")
                else:
                    validation_results['cross_checks'].append(
                        f" Ecuación contable validada (Activos = Pasivos + Patrimonio)"
                    )
                    print(f" Balance sheet cuadra correctamente")
            
            # ========== RESUMEN FINAL ==========
            print(f"\n{'='*60}")
            print(f" RESUMEN DE VALIDACIÓN CRUZADA")
            print(f"{'='*60}")
            print(f"Estado general: {' CONSISTENTE' if validation_results['consistent'] else '❌ INCONSISTENTE'}")
            print(f"Checks realizados: {len(validation_results['cross_checks'])}")
            print(f"Advertencias: {len(validation_results['warnings'])}")
            print(f"Ratios calculados: {len(validation_results['ratios_calculated'])}")
            
            if validation_results['warnings']:
                print(f"\n ADVERTENCIAS:")
                for warning in validation_results['warnings']:
                    print(f"  {warning}")
            
            print(f"{'='*60}\n")
            
        except Exception as e:
            error_msg = f"Error en validación cruzada: {e}"
            validation_results['warnings'].append(error_msg)
            validation_results['consistent'] = False
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
        
        return validation_results
    
    def save_income_results_enhanced(self, pdf_file: Path, output_dir: Path, 
                                    extraction: Dict, validation: Dict,
                                    cross_validation: Dict = None) -> Dict[str, Any]:  
        """NUEVA FUNCIÓN: Guardar resultados mejorados"""
        try:
            basename = pdf_file.stem
            files_created = 0
            
            # 1. Guardar resumen JSON extendido
            summary = {
                "extraction": {
                    "total_characters": extraction.get("total_characters", 0),
                    "pages_processed": extraction.get("pages_processed", []),
                    "financial_data": extraction.get("financial_data", {}),
                    "confidence": extraction.get("confidence", 0.8),
                    "language": extraction.get("language", "unknown"),
                    "has_related_party_data": extraction.get("has_related_party_data", False),  
                },
                "validation": validation,
                "cross_validation": cross_validation, 
                "processing_timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "quality_metrics": {
                    "data_categories_found": validation.get("financial_categories_found", 0),
                    "quality_score": validation.get("score", 0),
                    "final_confidence": validation.get("confidence", 0.8),
                    "has_ratio_warnings": validation.get("has_ratio_warnings", False),  
                }
            }
            
            summary_file = output_dir / f"{basename}_income_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            files_created += 1
            
            # 2. Guardar datos financieros específicos
            if extraction.get("financial_data"):
                financial_data_file = output_dir / f"{basename}_financial_data.json"
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
            
            quality_file = output_dir / f"{basename}_income_quality.txt"
            with open(quality_file, "w", encoding="utf-8") as f:
                f.write(quality_report)
            files_created += 1
            
            print(f" Archivos guardados: {files_created}")
            
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
            print(" DEBUG: Iniciando generate_enhanced_income_analysis_fixed")
            
            text = extraction.get("text", "")
            confidence = validation.get("confidence", 0.8)
            quality = validation.get("quality", "unknown")
            financial_data = extraction.get("financial_data", {})
            
            print(f" DEBUG: Texto length: {len(text)}")
            print(f" DEBUG: Financial data categories: {len(financial_data)}")
            print(f" DEBUG: Financial data sample: {dict(list(financial_data.items())[:3])}")
            print(f" DEBUG: Quality: {quality}, Confidence: {confidence}")
            
            if not text or len(text.strip()) < 500:
                print(" DEBUG: Texto insuficiente para análisis detallado")
                return "El contenido extraído de la cuenta de resultados es insuficiente para realizar un análisis detallado profesional."
            
            # VERIFICAR QUE HAY DATOS FINANCIEROS
            has_financial_data = any(values for values in financial_data.values() if values)
            print(f" DEBUG: Has financial data: {has_financial_data}")
            
            if not has_financial_data:
                print(" DEBUG: No hay datos financieros específicos, re-extrayendo con patrones mejorados...")
                # Re-extraer con patrones más amplios
                financial_data = extract_comprehensive_income_data(text)
                print(f" DEBUG: Re-extracción result: {dict(list(financial_data.items())[:2])}")
            
            # Calcular ratios financieros CON VALIDACIÓN
            ratios = {}
            try:
                ratios = calculate_financial_ratios(financial_data) if financial_data else {}
                print(f" DEBUG: Ratios calculados exitosamente: {len(ratios)}")
            except Exception as ratio_error:
                print(f" DEBUG: Error calculando ratios: {ratio_error}")
                ratios = {}
            
            # ANÁLISIS SIMPLIFICADO PRIMERO (para debugging)
            try:
                print(" DEBUG: Intentando análisis con LLM...")

                analysis_prompt = f"""
                Eres un analista financiero senior especializado en banca con experiencia en análisis de subsidiarias internacionales.

                CONTEXTO DEL DOCUMENTO:
                - Entidad: GarantiBank International N.V. (subsidiaria 100% de BBVA a través de Garanti BBVA)
                - Tipo de documento: Estados Financieros Individuales (no consolidados)
                - Periodo: Ejercicio 2023 con comparativa 2022

                DATOS FINANCIEROS EXTRAÍDOS:
                {json.dumps(financial_data, indent=2) if financial_data else "Datos en procesamiento"}

                RATIOS CALCULADOS:
                {json.dumps(ratios, indent=2) if ratios else "No disponibles"}

                FRAGMENTOS DE TEXTO CLAVE (primeros 3000 caracteres):
                {text[:3000]}

                INSTRUCCIONES PARA EL ANÁLISIS:

                1. **Estructura de Ingresos** (150-200 palabras):
                - Margen de intereses: Analiza ingresos vs gastos por intereses
                - Comisiones: Evalúa ingresos por servicios bancarios
                - Otros ingresos: Trading income, dividendos, etc.
                - Calcula Net Interest Margin si es posible
                - Identifica fuentes principales de ingresos

                2. **Análisis de Gastos y Eficiencia** (150-200 palabras):
                - Gastos operativos: Personal + Administrativos
                - Cost-to-income ratio y su interpretación
                - Eficiencia operativa vs benchmarks sectoriales (45-60% típico)
                - Identifica áreas de mayor gasto

                3. **Calidad Crediticia y Provisiones** (100-150 palabras):
                - Dotaciones para provisiones crediticias
                - Pérdidas esperadas (ECL)
                - Ratio de cobertura de morosidad
                - Impacto en rentabilidad

                4. **Rentabilidad y Márgenes** (150-200 palabras):
                - Beneficio neto y evolución
                - ROE (si se puede calcular con patrimonio)
                - ROA (si se puede calcular con activos)
                - Net profit margin
                - Comparación con ejercicio anterior

                5. **Conclusiones Estratégicas** (100-150 palabras):
                - Fortalezas identificadas
                - Áreas de mejora
                - Posicionamiento competitivo
                - Recomendaciones para inversores/management

                REGLAS CRÍTICAS:
                ✅ USA SOLO cifras presentes en los datos proporcionados
                ✅ Si un dato no está disponible, indícalo explícitamente
                ✅ Cita cifras exactas cuando las menciones (ej: "€4,157 miles")
                ✅ Compara 2023 vs 2022 cuando ambos años estén disponibles
                ✅ Interpreta ratios en contexto bancario (no industrial)
                ✅ Formato profesional con secciones numeradas y bullets
                ❌ NO inventes cifras que no aparecen en los datos
                ❌ NO uses datos de tu conocimiento general sobre BBVA
                ❌ NO hagas suposiciones sin evidencia en el texto

                NOTA IMPORTANTE: Esta es una subsidiaria del grupo Garanti BBVA, no el grupo consolidado BBVA S.A.

                Genera un análisis de 600-800 palabras siguiendo esta estructura.
                """

                
                # Llamada al LLM con manejo de errores mejorado
                try:
                    print(" DEBUG: Llamando al chat_client...")
                    analysis_response = self.chat_client.chat([
                        {"role": "system", "content": "Eres un analista financiero experto en banca con 15 años de experiencia."},
                        {"role": "user", "content": analysis_prompt}
                    ], max_tokens=1800)
                    
                    print(f" DEBUG: LLM respondió con {len(analysis_response)} caracteres")
                    
                    if not analysis_response or len(analysis_response.strip()) < 200:
                        print(" DEBUG: Respuesta LLM muy corta, usando fallback")
                        raise Exception("Respuesta LLM insuficiente")
                    
                    print(" DEBUG: Respuesta LLM exitosa")
                    
                except Exception as llm_error:
                    print(f" DEBUG: Error en LLM: {str(llm_error)}")
                    print(" DEBUG: Usando análisis fallback...")
                    analysis_response = self.generate_fallback_income_analysis(text, confidence, quality, financial_data, ratios)
                
                # CONSTRUIR RESPUESTA FINAL
                print(" DEBUG: Construyendo respuesta final...")
                
                response_parts = [
                    " **ANÁLISIS PROFESIONAL DE CUENTA DE RESULTADOS - BBVA**",
                    "=" * 70,
                    "",
                    analysis_response,
                    "",
                    "###  **INFORMACIÓN TÉCNICA DEL ANÁLISIS**",
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
                    " *Análisis generado por sistema de IA especializada en análisis de rentabilidad bancaria*"
                ]
                
                final_response = "\n".join(response_parts)
                print(f" DEBUG: Respuesta final construida con {len(final_response)} caracteres")
                
                return final_response
                
            except Exception as analysis_error:
                print(f" DEBUG: Error en análisis: {str(analysis_error)}")
                # Usar fallback completo
                return self.generate_fallback_income_analysis(text, confidence, quality, financial_data, ratios)
                
        except Exception as e:
            print(f" DEBUG: Error crítico en generate_enhanced_income_analysis_fixed: {str(e)}")
            # ÚLTIMO FALLBACK: Respuesta básica garantizada
            return f"""
 **ANÁLISIS DE CUENTA DE RESULTADOS - BBVA**

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
        response_parts.append(" **ANÁLISIS DE CUENTA DE RESULTADOS - BBVA**")
        response_parts.append("=" * 60)
        
        text_lower = normalize_text(text)
        
        # Análisis de ingresos principales
        response_parts.append("\n###  **ANÁLISIS DE INGRESOS PRINCIPALES**")
        
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
                    response_parts.append("  -  ATENCIÓN: Caída significativa que requiere análisis estratégico")
        elif any(term in text_lower for term in ["commission", "comisiones"]):
            response_parts.append("• **Ingresos por comisiones**: Fuente complementaria de ingresos identificada")
        
        # Análisis de gastos
        response_parts.append("\n###  **ANÁLISIS DE GASTOS OPERATIVOS**")
        
        if financial_data.get('operating_expenses'):
            amounts = financial_data['operating_expenses']
            response_parts.append(f"• **Gastos operativos**: {amounts} (miles de euros)")
            if 'efficiency_ratio' in ratios:
                efficiency = ratios['efficiency_ratio']
                response_parts.append(f"  - Ratio de eficiencia: {efficiency:.1f}%")
                if efficiency < 50:
                    response_parts.append("  -  Eficiencia operativa superior al promedio sectorial")
                elif efficiency > 60:
                    response_parts.append("  -  Oportunidades de mejora en eficiencia operativa")
        
        if financial_data.get('staff_costs'):
            amounts = financial_data['staff_costs']
            response_parts.append(f"• **Gastos de personal**: {amounts} (miles de euros)")
            if 'staff_cost_ratio' in ratios:
                staff_ratio = ratios['staff_cost_ratio']
                response_parts.append(f"  - Ratio sobre ingresos: {staff_ratio:.1f}%")
        
        # Provisiones y calidad crediticia
        response_parts.append("\n###  **PROVISIONES Y CALIDAD CREDITICIA**")
        
        if financial_data.get('provisions'):
            amounts = financial_data['provisions']
            response_parts.append(f"• **Provisiones**: {amounts} (miles de euros)")
            response_parts.append("• Las provisiones reflejan la gestión prudente del riesgo crediticio")
        elif any(term in text_lower for term in ["provision", "provisiones"]):
            response_parts.append("• **Provisiones**: Identificadas como parte de la gestión de riesgos")
        
        # Rentabilidad
        response_parts.append("\n###  **ANÁLISIS DE RENTABILIDAD**")
        
        if financial_data.get('net_profit'):
            amounts = financial_data['net_profit']
            response_parts.append(f"• **Beneficio neto**: {amounts} (miles de euros)")
            if 'net_profit_margin' in ratios:
                margin = ratios['net_profit_margin']
                response_parts.append(f"  - Margen neto: {margin:.1f}%")
                if margin > 15:
                    response_parts.append("  -  Rentabilidad sólida para el sector bancario")
                elif margin < 10:
                    response_parts.append("  -  Margen por debajo del promedio sectorial")
        
        # Ratios adicionales
        if ratios:
            response_parts.append("\n###  **RATIOS FINANCIEROS CALCULADOS**")
            for ratio_name, value in ratios.items():
                if not ratio_name.endswith('_growth'):
                    response_parts.append(f"• **{ratio_name.replace('_', ' ').title()}**: {value:.2f}%")
        
        # Conclusiones
        response_parts.append("\n###  **CONCLUSIONES BASADAS EN DATOS EXTRAÍDOS**")
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
    print(" Income Agent v4.2 AUTÓNOMO Multi-Agent - ERRORES CORREGIDOS")
    print(f" PDF: {args.pdf}")
    print(f" Salida: {args.out}")
    print(f" Groq/Azure OpenAI: Configuración optimizada")
    print(f" Max steps: {args.maxsteps}")
    print("CARACTERÍSTICAS: Conversión segura de tipos, patrones mejorados, debugging completo")
    
    try:
        # VERIFICAR PDF
        pdf_path = Path(args.pdf)
        output_dir = Path(args.out)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not pdf_path.exists():
            print(f"Error: PDF no encontrado en {pdf_path}")
            return
        
        # CREAR AGENTE Y EJECUTAR
        agent = IncomeREACTAgent()
        
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
            print(" ==== ANÁLISIS DETALLADO GENERADO ====")
            analysis = result.get("specific_answer", "No hay respuesta específica disponible")
            print(f"Longitud del análisis: {len(analysis)} caracteres")
            
            summary = result.get("extraction_summary", {})
            print(f"Caracteres procesados: {summary.get('total_characters', 0):,}")
            print(f"Categorías financieras: {summary.get('financial_data_categories', 0)}")
            print(f"Confianza: {summary.get('confidence', 0.8):.1%}")
            print(f"Calidad: {summary.get('quality', 'unknown').title()}")
            print(" Análisis detallado con conversión segura de tipos completado")
        else:
            print(f" Error: {result.get('error_details', 'Error desconocido')}")
        
        print(" Análisis de cuenta de resultados completado!")
        print(" IncomeREACTAgent con errores corregidos disponible para sistema multi-agente")
        
    except Exception as e:
        print(f" Error durante la ejecución: {e}")
        raise

if __name__ == "__main__":
    main()
