"""
CashFlows Agent REACT - Versión Multi-Agente AUTÓNOMA COMPLETA
Especializado en análisis de estado de flujos de efectivo con patrón REACT exitoso
CARACTERÍSTICAS: Tool calls, análisis detallado con LLM, respuestas extensas de 600-800 palabras
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
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

print(" ----- Groq Configuration -----")
print(f" API Key: {'✓' if GROQ_API_KEY else '✗'}")
print(f" Model: {GROQ_MODEL}")

# ===== CLIENTE CHAT ORIGINAL (MANTENIDO) =====
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

# ===== CLIENTE EMBEDDINGS AZURE OPTIMIZADO =====
class AzureEmbeddingClient:
    """Cliente optimizado para generar embeddings usando Azure OpenAI con batch processing"""
    
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
            
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            
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
            response = self.client.embeddings.create(
                model=self.model,
                input=truncated_texts
            )
            
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
        ⚡ VERSIÓN OPTIMIZADA con caché de embeddings
        
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
        for i, chunk_embedding in enumerate(chunk_embeddings):
            similarity = cosine_similarity(
                [query_embedding], 
                [chunk_embedding]
            )[0][0]
            similarities.append((i, float(similarity), text_chunks[i]))
        
        # 4. Ordenar y devolver top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k], chunk_embeddings

# ===== INICIALIZACIÓN =====
chat_client = ChatClient()
embedding_client = AzureEmbeddingClient()  

# ===== HERRAMIENTAS ESPECÍFICAS PARA CASHFLOWS CON EXTRACCIÓN  =====

@dataclass
class AnalyzeCashflowStructureTool:
    """
    Herramienta para analizar la estructura del PDF y localizar el estado de flujos de efectivo.
    
    CAMBIO IMPLEMENTADO: Detección automática del tamaño del PDF
    - No usa valores hardcoded para anchor_page, max_pages o extend
    - Busca en TODAS las páginas del PDF reducido (1 a N)
    - Se adapta automáticamente al tamaño del documento
    """
    name: str = "analyze_cashflow_structure"
    description: str = "Analiza estructura del PDF para localizar el estado de flujos de efectivo"
    
    def run(self, pdf_path: str, anchor_page: int = None, max_pages: int = None, 
            extend: int = None, **kwargs) -> Dict[str, Any]:
        """
        Analiza el PDF para encontrar el Statement of Cash Flows.
        
        Args:
            pdf_path: Ruta al archivo PDF
            anchor_page: Página ancla (None = detección automática)
            max_pages: Máximo de páginas (None = usar total del PDF)
            extend: Rango de búsqueda (None = buscar en todas)
            
        Returns:
            Dict con páginas seleccionadas, score y estado de búsqueda
        """
        try:
            # PASO 1: Detectar automáticamente el tamaño del PDF
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                
                # Si no se especifica anchor_page, usar página central
                if anchor_page is None:
                    anchor_page = total_pages // 2
                
                # Si no se especifica max_pages, usar total de páginas del PDF
                if max_pages is None:
                    max_pages = total_pages
                    
                # Si no se especifica extend, buscar en TODO el PDF reducido
                if extend is None:
                    extend = total_pages
            
            print(f"  Analizando estructura en PDF de {total_pages} páginas")
            print(f"   Buscando desde página 1 hasta {total_pages}")
            
            # PASO 2: Definir páginas objetivo - TODAS las páginas del PDF reducido
            target_pages = list(range(1, total_pages + 1))
            
            # PASO 3: Buscar indicadores de flujos de efectivo en cada página
            with pdfplumber.open(pdf_path) as pdf:
                found_cashflow = False
                best_pages = []
                
                for page_num in target_pages:
                    page = pdf.pages[page_num - 1]
                    text = page.extract_text() or ""
                    text_lower = text.lower()
                    
                    # Indicadores expandidos de flujos de efectivo
                    cashflow_indicators = [
                        "statement of cash flows",
                        "cash flow statement",
                        "flujos de efectivo",
                        "operating activities",
                        "investing activities",
                        "financing activities",
                        "net cash provided",
                        "net cash used",
                        "cash and cash equivalents",
                        "actividades operativas",
                        "actividades de inversión",
                        "actividades de financiación",
                        "efectivo generado",
                        "efectivo utilizado",
                        "flujo de caja"
                    ]
                    
                    # Calcular score de relevancia
                    score = sum(1 for indicator in cashflow_indicators if indicator in text_lower)
                    
                    if score > 0:
                        best_pages.append((page_num, score))
                        print(f"Flujos de efectivo encontrados en página {page_num} (score: {score})")
                        found_cashflow = True
                
                # PASO 4: Ordenar páginas por relevancia y seleccionar las mejores
                best_pages.sort(key=lambda x: x[1], reverse=True)
                selected_pages = [page for page, score in best_pages[:6]]
                
                # PASO 5: Retornar resultados
                return {
                    "success": True,
                    "pages_selected": selected_pages or target_pages[:6],
                    "cashflow_found": found_cashflow,
                    "anchor_page_used": anchor_page,
                    "relevance_scores": dict(best_pages),
                    "total_pages_searched": len(target_pages)  # Nuevo campo para debugging
                }
                
        except Exception as e:
            print(f"Error en AnalyzeCashflowStructureTool: {str(e)}")
            return {"success": False, "error": str(e)}


@dataclass
class ExtractCashflowStatementTool:
    name: str = "extractcashflowstatement"
    description: str = "Extrae el contenido del estado de flujos de efectivo con análisis avanzado y búsqueda semántica"
    
    def run(self, pdf_path: str, analysis_json: Dict = None, extract_semantic_chunks: bool = True, **kwargs) -> Dict[str, Any]:
        try:
            pages_to_process = analysis_json.get("pages_selected", [6, 7, 8, 9, 10]) if analysis_json else [6, 7, 8, 9, 10]
            print(f" Extrayendo páginas con análisis avanzado: {pages_to_process}")
            
            extracted_text = ""
            total_chars = 0
            financial_data = {}
            all_text_chunks = []  
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num in pages_to_process:
                    if page_num <= len(pdf.pages):
                        page = pdf.pages[page_num - 1]
                        text = page.extract_text() or ""
                        
                        # Buscar contenido relevante de flujos de efectivo
                        text_lower = text.lower()
                        cashflow_keywords = [
                            "cash", "flow", "operating", "investing", "financing", 
                            "activities", "net cash", "efectivo", "flujo", "actividades"
                        ]
                        
                        if any(keyword in text_lower for keyword in cashflow_keywords):
                            extracted_text += f"\n=== PÁGINA {page_num} ===\n{text}"
                            total_chars += len(text)
                            
                            # Extraer datos tabulares con pdfplumber
                            print(f"Extrayendo datos tabulares de página {page_num}")
                            table_data = self._extract_table_data(page)
                            
                            # Combinar datos de tabla con los ya extraídos
                            for key, values in table_data.items():
                                if key not in financial_data:
                                    financial_data[key] = []
                                financial_data[key].extend(values)
                                
                                if values:
                                    print(f"   ✓ {key}: {values[:2]}")

                            # Dividir en chunks para embeddings
                            page_chunks = self._split_text_into_chunks(text, chunk_size=1000)
                            all_text_chunks.extend([
                                {'page': page_num, 'chunk': chunk}
                                for chunk in page_chunks
                            ])
                            
                            print(f"Página {page_num}: {len(text)} caracteres extraídos")
                            
                            # NUEVO: Extracción avanzada de datos financieros
                            page_data = self._extract_financial_numbers(text)
                            for key, values in page_data.items():
                                if key not in financial_data:
                                    financial_data[key] = []
                                financial_data[key].extend(values)
            
            if not extracted_text:
                # Fallback: extraer todas las páginas en el rango
                print(" Fallback: extrayendo páginas 1-20")
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num in range(1, min(20, len(pdf.pages) + 1)):
                        page = pdf.pages[page_num - 1]
                        text = page.extract_text() or ""
                        extracted_text += f"\n=== PÁGINA {page_num} ===\n{text}"
                        total_chars += len(text)
                        
                        # También añadir chunks en fallback
                        page_chunks = self._split_text_into_chunks(text, chunk_size=1000)
                        all_text_chunks.extend([
                            {'page': page_num, 'chunk': chunk}
                            for chunk in page_chunks
                        ])
            
            # ===== BÚSQUEDA SEMÁNTICA OPTIMIZADA CON EMBEDDINGS =====
            semantic_results = {}
            if all_text_chunks and extract_semantic_chunks:
                print(f" Aplicando búsqueda semántica optimizada ({len(all_text_chunks)} chunks)...")
                
                chunk_texts = [item['chunk'] for item in all_text_chunks]
                
                # OPTIMIZADO: Generar embeddings de chunks en BATCH (1 sola llamada)
                print(f"   Generando embeddings en batch...")
                chunk_embeddings = embedding_client.get_batch_embeddings(chunk_texts)
                
                if chunk_embeddings is None:
                    print("   Error generando embeddings, saltando búsqueda semántica")
                else:
                    print(f"   {len(chunk_embeddings)} embeddings generados en batch")
                    
                    # Queries semánticas específicas para flujos de efectivo
                    semantic_queries = {
                        'operating_activities': "flujos de efectivo de actividades operativas net cash operating activities",
                        'investing_activities': "flujos de efectivo de actividades de inversión cash from investing",
                        'financing_activities': "flujos de efectivo de actividades de financiación cash from financing",
                        'net_cash_change': "variación neta en efectivo y equivalentes net change in cash",
                        'cash_beginning': "efectivo al inicio del período cash at beginning of period",
                        'cash_ending': "efectivo al final del período cash at end of period"
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
                                if score > 0.50:  # Umbral de similaridad
                                    chunk_info = all_text_chunks[idx]
                                    relevant_chunks.append({
                                        'score': score,
                                        'text': chunk_text,
                                        'page': chunk_info['page']
                                    })
                                    print(f"   {category} (score {score:.2f}, page {chunk_info['page']})")
                            
                            if relevant_chunks:
                                semantic_results[category] = relevant_chunks
                        
                        except Exception as e:
                            print(f"   Error en búsqueda semántica para {category}: {e}")
                    
                    # Enriquecer texto con resultados semánticos
                    if semantic_results:
                        print(f" Categorías encontradas con embeddings: {len(semantic_results)}")
                        enriched_text = "\n\n=== SECCIONES RELEVANTES (BÚSQUEDA SEMÁNTICA) ===\n"
                        
                        for category, chunks in semantic_results.items():
                            enriched_text += f"\n--- {category.upper().replace('_', ' ')} ---\n"
                            for chunk_data in chunks[:2]:  # Top 2 por categoría
                                enriched_text += f"[Página {chunk_data['page']}, Score: {chunk_data['score']:.2f}]\n"
                                enriched_text += chunk_data['text'][:500] + "...\n\n"
                        
                        # Prefijar al texto original
                        extracted_text = enriched_text + "\n\n=== TEXTO COMPLETO EXTRAÍDO ===\n" + extracted_text[:5000]
            
            # Crear chunks básicos (compatibilidad con código existente)
            chunks = []
            if extracted_text:
                chunks = self._create_semantic_chunks(extracted_text)
            
            confidence = 1.0 if total_chars > 2000 else 0.8 if total_chars > 1000 else 0.6
            
            # Ajustar confianza con embeddings
            if semantic_results:
                confidence = min(1.0, confidence + 0.1)  # Bonus por búsqueda semántica exitosa
            
            return {
                "success": True,
                "text": extracted_text,
                "total_characters": total_chars,
                "chunks": chunks,
                "confidence": confidence,
                "pages_processed": pages_to_process,
                "financial_data": financial_data,
                "semantic_results": semantic_results,  
                "semantic_categories_found": len(semantic_results),  
                "total_chunks_analyzed": len(all_text_chunks)  
            }
            
        except Exception as e:
            print(f"❌ Error en ExtractCashflowStatementTool: {e}")
            return {"success": False, "error": str(e)}
    
    def _extract_financial_numbers(self, text: str) -> Dict[str, List[float]]:
        """
        Extrae números financieros con PATTERNS MEJORADOS para formato tabular.
        
        MEJORAS: Patterns multi-línea, números negativos, filtrado de valores absurdos
        """
        
        # PATTERNS MEJORADOS: [\s\S]{0,100}? captura números en líneas posteriores
        patterns = {
            'operating_cash': [
                r'operating.*activities[\s\S]{0,100}?([\d.,]+)',
                r'actividades.*operativas[\s\S]{0,100}?([\d.,]+)',
                r'(?:net cash|cash flow).*?(?:from|provided).*?operat\w*[\s\S]{0,200}?([\d.,]+)',
                r'cash.*from.*operations[\s\S]{0,100}?€?\s*([\d.,]+)',
                r'operativ[\w]*[\s\S]{0,50}?([\d.,]+)'
            ],
            
            'investing_cash': [
                r'investing.*activities[\s\S]{0,100}?([\d.,]+)',
                r'actividades.*inversión[\s\S]{0,100}?([\d.,]+)',
                r'(?:net cash|cash flow).*?(?:from|used).*?invest\w*[\s\S]{0,200}?([\d.,]+)',
                r'cash.*from.*investing[\s\S]{0,100}?€?\s*([\d.,]+)',
                r'invers[\w]*[\s\S]{0,50}?([\d.,]+)'
            ],
            
            'financing_cash': [
                r'financing.*activities[\s\S]{0,100}?([\d.,]+)',
                r'actividades.*financiación[\s\S]{0,100}?([\d.,]+)',
                r'(?:net cash|cash flow).*?(?:from|used).*?financ\w*[\s\S]{0,200}?([\d.,]+)',
                r'cash.*from.*financing[\s\S]{0,100}?€?\s*([\d.,]+)',
                r'financi[\w]*[\s\S]{0,50}?([\d.,]+)'
            ],
            
            'net_change_cash': [
                r'net (?:increase|decrease|change).*?cash[\s\S]{0,100}?([\d.,]+)',
                r'variación neta.*?efectivo[\s\S]{0,100}?([\d.,]+)',
                r'net.*?cash[\s\S]{0,50}?([\d.,]+)'
            ],
            
            # NUEVAS CATEGORÍAS para validación de consistencia
            'cash_beginning': [
                r'cash.*?(?:beginning|start|inicio).*?(?:of period|period|año|year)[\s\S]{0,100}?([\d.,]+)',
                r'efectivo al inicio[\s\S]{0,100}?([\d.,]+)',
                r'(?:beginning|inicio).*?cash[\s\S]{0,100}?([\d.,]+)'
            ],
            
            'cash_ending': [
                r'cash.*?(?:end|close|final).*?(?:of period|period|año|year)[\s\S]{0,100}?([\d.,]+)',
                r'efectivo al final[\s\S]{0,100}?([\d.,]+)',
                r'(?:end|final).*?cash[\s\S]{0,100}?([\d.,]+)'
            ]
        }
        
        extracted_data = {}
        
        for category, pattern_list in patterns.items():
            values = []
            
            for pattern in pattern_list:
                # CAMBIO CRÍTICO: Añadir re.DOTALL flag
                matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
                
                for match in matches:
                    try:
                        # Limpiar número: remover comas, mantener punto decimal
                        clean_number = match.replace(',', '').replace(' ', '')
                        
                        # NUEVO: Manejar paréntesis como números negativos
                        # En contabilidad: (208,961) = -208,961
                        original_match = match
                        if '(' in original_match and ')' in original_match:
                            clean_number = '-' + re.sub(r'[^\d.]', '', clean_number)
                        else:
                            clean_number = re.sub(r'[^\d.-]', '', clean_number)
                        
                        # Validar que no esté vacío
                        if clean_number and clean_number not in ['-', '.', '-.']:
                            number = float(clean_number)
                            
                            # NUEVO: Filtrar valores absurdos (fechas, códigos)
                            if abs(number) < 10000000:  # Máximo 10M en miles de EUR
                                values.append(number)
                            
                    except (ValueError, AttributeError):
                        continue
            
            # NUEVO: Eliminar duplicados manteniendo orden
            seen = set()
            unique_values = []
            for v in values:
                if v not in seen:
                    seen.add(v)
                    unique_values.append(v)
            
            # Top 3 valores por categoría (evita ruido)
            extracted_data[category] = unique_values[:3]
        
        return extracted_data

    def _extract_table_data(self, page) -> Dict[str, Any]:
        """
        Extrae datos tabulares del Statement of Cash Flows usando page.extract_tables().
        Complementa _extract_financial_numbers() capturando estructura tabular.
        """
        tables = page.extract_tables()
        financial_values = {}
        
        if not tables:
            return financial_values
        
        for table in tables:
            if not table:
                continue
            
            for row in table:
                if not row:
                    continue
                
                # Convertir None a string para evitar errores
                row_text = ' '.join([str(cell) if cell else '' for cell in row]).lower()
                
                # CATEGORÍA 1: Operating Activities
                if 'net cash' in row_text and 'operating' in row_text:
                    for cell in row[1:]:  # Saltar columna de labels
                        if cell and re.search(r'\d', str(cell)):
                            try:
                                clean_value = re.sub(r'[^\d.-]', '', str(cell))
                                if clean_value and clean_value not in ['-', '.']:
                                    value = float(clean_value)
                                    financial_values.setdefault('operating_cash', []).append(value)
                            except (ValueError, AttributeError):
                                pass
                
                # CATEGORÍA 2: Investing Activities
                elif 'investing' in row_text and ('activities' in row_text or 'cash' in row_text):
                    for cell in row[1:]:
                        if cell and re.search(r'\d', str(cell)):
                            try:
                                clean_value = re.sub(r'[^\d.-]', '', str(cell))
                                if clean_value and clean_value not in ['-', '.']:
                                    value = float(clean_value)
                                    # Manejar paréntesis como negativos
                                    if '(' in str(cell) and ')' in str(cell):
                                        value = -abs(value)
                                    financial_values.setdefault('investing_cash', []).append(value)
                            except (ValueError, AttributeError):
                                pass
                
                # CATEGORÍA 3: Financing Activities
                elif 'financing' in row_text and ('activities' in row_text or 'cash' in row_text):
                    for cell in row[1:]:
                        if cell and re.search(r'\d', str(cell)):
                            try:
                                clean_value = re.sub(r'[^\d.-]', '', str(cell))
                                if clean_value and clean_value not in ['-', '.']:
                                    value = float(clean_value)
                                    if '(' in str(cell) and ')' in str(cell):
                                        value = -abs(value)
                                    financial_values.setdefault('financing_cash', []).append(value)
                            except (ValueError, AttributeError):
                                pass
                
                # CATEGORÍA 4: Net Change in Cash
                elif ('net' in row_text and 'increase' in row_text and 'cash' in row_text) or \
                     ('net' in row_text and 'change' in row_text and 'cash' in row_text):
                    for cell in row[1:]:
                        if cell and re.search(r'\d', str(cell)):
                            try:
                                clean_value = re.sub(r'[^\d.-]', '', str(cell))
                                if clean_value and clean_value not in ['-', '.']:
                                    value = float(clean_value)
                                    financial_values.setdefault('net_change_cash', []).append(value)
                            except (ValueError, AttributeError):
                                pass
                
                # CATEGORÍA 5: Cash at Beginning
                elif 'cash' in row_text and ('beginning' in row_text or 'inicio' in row_text):
                    for cell in row[1:]:
                        if cell and re.search(r'\d', str(cell)):
                            try:
                                clean_value = re.sub(r'[^\d.-]', '', str(cell))
                                if clean_value and clean_value not in ['-', '.']:
                                    value = float(clean_value)
                                    financial_values.setdefault('cash_beginning', []).append(value)
                            except (ValueError, AttributeError):
                                pass
                
                # CATEGORÍA 6: Cash at End
                elif 'cash' in row_text and ('end' in row_text or 'final' in row_text):
                    for cell in row[1:]:
                        if cell and re.search(r'\d', str(cell)):
                            try:
                                clean_value = re.sub(r'[^\d.-]', '', str(cell))
                                if clean_value and clean_value not in ['-', '.']:
                                    value = float(clean_value)
                                    financial_values.setdefault('cash_ending', []).append(value)
                            except (ValueError, AttributeError):
                                pass
        
        return financial_values

    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Divide texto en chunks para procesamiento con embeddings"""
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
    
    def _create_semantic_chunks(self, text: str) -> List[str]:
        """Crear chunks semánticos más inteligentes (método original mejorado)"""
        lines = text.split('\n')
        chunks = []
        current_chunk = ""
        
        section_headers = [
            'operating activities', 'investing activities', 'financing activities',
            'actividades operativas', 'actividades de inversión', 'actividades de financiación',
            'cash flow', 'flujo de efectivo'
        ]
        
        for line in lines:
            line_lower = line.lower()
            
            # Si encontramos una nueva sección importante, guardar chunk anterior
            if any(header in line_lower for header in section_headers) and current_chunk:
                if len(current_chunk) > 100:
                    chunks.append(current_chunk.strip())
                current_chunk = line
            else:
                current_chunk += "\n" + line
                
                # También dividir por tamaño
                if len(current_chunk) > 1200:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks


@dataclass
class ValidateCashflowQualityTool:
    name: str = "validatecashflowquality"
    description: str = "Valida la calidad de los datos extraídos de flujos de efectivo con análisis avanzado"
    
    def run(self, extraction: Dict, **kwargs) -> Dict[str, Any]:
        try:
            text = extraction.get("text", "")
            confidence = extraction.get("confidence", 0.0)
            financial_data = extraction.get("financial_data", {})
            
            # Criterios de validación mejorados para flujos de efectivo
            quality_score = 0
            validation_details = []
            
            text_lower = text.lower()
            
            # Verificar secciones principales (peso: 25 cada una)
            if "operating" in text_lower or "operativ" in text_lower:
                quality_score += 25
                validation_details.append(" Actividades operativas encontradas")
            
            if "investing" in text_lower or "inversión" in text_lower:
                quality_score += 25
                validation_details.append(" Actividades de inversión encontradas")
            
            if "financing" in text_lower or "financiación" in text_lower:
                quality_score += 25
                validation_details.append(" Actividades de financiación encontradas")
            
            if "cash and cash equivalents" in text_lower or "efectivo y equivalentes" in text_lower:
                quality_score += 15
                validation_details.append(" Efectivo y equivalentes encontrados")
            
            # Bonificaciones por datos financieros específicos encontrados
            if financial_data:
                operating = financial_data.get('operating_cash', [])
                investing = financial_data.get('investing_cash', [])
                financing = financial_data.get('financing_cash', [])
                net_change = financial_data.get('net_change_cash', [])
                
                # Bonus por datos numéricos encontrados
                if operating:
                    quality_score += 10
                    validation_details.append(f"✓ Flujos operativos: {operating[0]:,.0f}")
                
                if investing:
                    quality_score += 10
                    validation_details.append(f"✓ Flujos de inversión: {investing[0]:,.0f}")
                
                if financing:
                    quality_score += 10
                    validation_details.append(f"✓ Flujos de financiación: {financing[0]:,.0f}")
                
                # Validar consistencia matemática
                if operating and investing and net_change:
                    try:
                        calculated_net = operating[0] + investing[0]
                        if financing:
                            calculated_net += financing[0]
                        
                        actual_net = net_change[0]
                        diff_pct = abs(calculated_net - actual_net) / abs(actual_net) * 100
                        
                        if diff_pct < 5:
                            quality_score += 15
                            validation_details.append("✓ Validación matemática: Consistente")
                        else:
                            validation_details.append(f"Discrepancia: {diff_pct:.1f}%")
                    except:
                        pass

            # Determinar calidad
            if quality_score >= 80:
                quality = "excellent"
            elif quality_score >= 60:
                quality = "good"
            elif quality_score >= 40:
                quality = "fair"
            else:
                quality = "poor"
            
            final_confidence = min(confidence + (quality_score / 100 * 0.2), 1.0)
            
            print(f" Validación completada: {quality} (puntuación: {quality_score}/100, confianza: {final_confidence:.3f})")
            
            return {
                "success": True,
                "quality": quality,
                "confidence": final_confidence,
                "score": quality_score,
                "details": validation_details,
                "financial_data_found": len(financial_data) if financial_data else 0
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

@dataclass
class SaveCashflowResultsTool:
    name: str = "savecashflowresults"
    description: str = "Guarda los resultados del análisis de flujos de efectivo con información extendida"
    
    def run(self, output_dir: str, pdf_name: str, analysis: Dict, extraction: Dict, validation: Dict, **kwargs) -> Dict[str, Any]:
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            base_name = Path(pdf_name).stem
            files_created = 0
            
            # 1. Guardar resumen JSON extendido
            summary = {
                "analysis": analysis,
                "extraction": {
                    "total_characters": extraction.get("total_characters", 0),
                    "confidence": extraction.get("confidence", 0),
                    "pages_processed": extraction.get("pages_processed", []),
                    "financial_data": extraction.get("financial_data", {})  # NUEVO
                },
                "validation": validation,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "extraction_quality_metrics": {  # NUEVO
                    "data_categories_found": validation.get("financial_data_found", 0),
                    "quality_score": validation.get("score", 0),
                    "final_confidence": validation.get("confidence", 0.8)
                }
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
            
            # 3. Guardar reporte de calidad extendido
            quality_report = f"""
REPORTE DE CALIDAD EXTENDIDO - FLUJOS DE EFECTIVO
=================================================
PDF: {pdf_name}
Fecha: {time.strftime("%Y-%m-%d %H:%M:%S")}

RESULTADOS DE VALIDACIÓN:
- Calidad: {validation.get('quality', 'unknown')}
- Puntuación: {validation.get('score', 0)}/100
- Confianza final: {validation.get('confidence', 0):.3f}
- Datos financieros encontrados: {validation.get('financial_data_found', 0)} categorías

DETALLES DE VALIDACIÓN:
{chr(10).join(validation.get('details', []))}

EXTRACCIÓN DETALLADA:
- Caracteres procesados: {extraction.get('total_characters', 0)}
- Páginas procesadas: {extraction.get('pages_processed', [])}
- Chunks generados: {len(extraction.get('chunks', []))}
- Datos financieros extraídos: {extraction.get('financial_data', {})}

MÉTRICAS DE CALIDAD:
- Cobertura de secciones: {'Completa' if validation.get('score', 0) >= 75 else 'Parcial' if validation.get('score', 0) >= 50 else 'Limitada'}
- Precisión de extracción: {validation.get('confidence', 0.8)*100:.1f}%
- Recomendación: {'Análisis confiable' if validation.get('score', 0) >= 60 else 'Requiere revisión manual'}
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
                "output_directory": str(output_path),
                "summary_data": summary  # NUEVO: Devolver resumen para uso posterior
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

INSTRUCCIONES SIMPLES:
1. Usa "analyzecashflowstructure" para localizar el estado de flujos de efectivo
2. Usa "extractcashflowstatement" para extraer el contenido
3. Usa "validatecashflowquality" para validar los datos
4. Usa "savecashflowresults" para guardar los resultados
5. Al terminar, responde "CASHFLOWEXTRACTIONCOMPLETED"

IMPORTANTE:
- Menciona CLARAMENTE el nombre de la herramienta que quieres usar
- NO uses JSON - solo menciona el nombre de la herramienta
- Ejecuta las 4 herramientas en orden secuencial

EMPEZAR AHORA con analyzecashflowstructure.
"""

# ===== CLASE CASHFLOWS REACT AGENT CON ANÁLISIS MEJORADO =====
class CashFlowsREACTAgent:
    """Agente REACT especializado en flujos de efectivo con análisis detallado"""
    
    def __init__(self):
        self.agent_type = "cashflows"
        self.max_steps = 10
        self.chat_client = chat_client
    
    def run_final_financial_extraction_agent(self, pdf_path: str, question: str = None) -> Dict[str, Any]:
        """Ejecuta la extracción de flujos de efectivo con patrón REACT"""
        try:
            print(f" CashFlowsREACTAgent AUTÓNOMO iniciando extracción para: {pdf_path}")
            
            pdf_file = Path(pdf_path)
            output_dir = Path("data/salida")
            
            # Configurar contexto de herramientas
            tools_ctx = {
                "pdfpath": str(pdf_file),
                "outputdir": str(output_dir),
                "lastanalysis": {},
                "lastextraction": {},
                "lastvalidation": {}
            }
            
            # Ejecutar patrón REACT
            history = [{"role": "system", "content": REACT_SYSTEM_PROMPT}]
            finished = False
            steps = 0
            
            print(f" Iniciando CashFlows Agent MEJORADO para {pdf_file.name}")
            print(f" PDF: {pdf_file}")
            print(f" Output: {output_dir}")
            #print(f" Anchor page: {tools_ctx['anchorpage']}")
            
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
                    "session_id": f"cashflows_{pdf_file.stem}",
                    "final_response": "Proceso interrumpido por límite de pasos",
                    "agent_type": "cashflows",
                    "error_details": "Max steps reached",
                    "specific_answer": "El análisis de flujos de efectivo fue interrumpido por límite de pasos."
                }
            
            # Generar respuesta específica MEJORADA
            specific_answer = self.generate_enhanced_analysis(question, tools_ctx)
            
            print(" Análisis completado exitosamente")
            print(" Cashflow extraction completed successfully (AUTÓNOMO)")
            
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
    
    def execute_react_step(self, history: List[Dict[str, str]], tools_ctx: Dict[str, Any]) -> Tuple[List[Dict[str, str]], bool]:
        """Ejecuta un paso del patrón REACT - MISMA LÓGICA EXITOSA"""
        try:
            assistant_text = self.chat_client.chat(history, max_tokens=100)
            history.append({"role": "assistant", "content": assistant_text})
            
            print(f" Asistente respondió: {len(assistant_text)} caracteres")
            print(f" RESPUESTA COMPLETA:")
            print(assistant_text)
            print("=" * 50)
            
            # DETECCIÓN DE FRASES DE FINALIZACIÓN
            completion_phrases = [
                "cashflowextractioncompleted", "extraction completed successfully",
                "archivos guardados exitosamente", "task completed", "analysis completed"
            ]
            
            for phrase in completion_phrases:
                if phrase.lower() in assistant_text.lower():
                    print(f" FRASE DE FINALIZACIÓN DETECTADA: '{phrase}'")
                    return history, True
            
            # DETECCIÓN ROBUSTA DE TOOL CALLS
            tool_name = None
            tool_names = ["analyzecashflowstructure", "extractcashflowstatement", "validatecashflowquality", "savecashflowresults"]
            
            for tool in tool_names:
                if tool in assistant_text.lower():
                    tool_name = tool
                    print(f" Tool detectada: {tool_name}")
                    break
            
            if not tool_name:
                print(" No hay tool_call detectado, continuando flujo")
                return history, False
            
            # Preparar parámetros según la herramienta
            if tool_name == "analyzecashflowstructure":
                params = {
                    "pdf_path": tools_ctx["pdfpath"],
                    "anchor_page": None,
                    "max_pages": None,
                    "extend": None  
                }
                
            elif tool_name == "extractcashflowstatement":
                params = {
                    "pdf_path": tools_ctx["pdfpath"],
                    "analysis_json": tools_ctx.get("lastanalysis", {}),
                    "extract_semantic_chunks": True
                }
                
            elif tool_name == "validatecashflowquality":
                params = {
                    "extraction": tools_ctx.get("lastextraction", {"text": assistant_text, "confidence": 0.8})
                }
                
            elif tool_name == "savecashflowresults":
                params = {
                    "output_dir": tools_ctx["outputdir"],
                    "pdf_name": str(tools_ctx["pdfpath"]),
                    "analysis": tools_ctx.get("lastanalysis", {}),
                    "extraction": tools_ctx.get("lastextraction", {}),
                    "validation": tools_ctx.get("lastvalidation", {})
                }
            
            # EJECUTAR HERRAMIENTA
            tool_obj = TOOLS_REGISTRY.get(tool_name)
            if not tool_obj:
                print(f" Herramienta no encontrada: {tool_name}")
                return history, False
            
            try:
                print(f" EJECUTANDO {tool_name} con parámetros mejorados")
                result = tool_obj.run(**params)
                
                if result.get("success", False):
                    print(f" {tool_name} ejecutado EXITOSAMENTE")
                    
                    # Actualizar contexto
                    if tool_name == "analyzecashflowstructure":
                        tools_ctx["lastanalysis"] = result
                        
                    elif tool_name == "extractcashflowstatement":
                        tools_ctx["lastextraction"] = result
                        
                    elif tool_name == "validatecashflowquality":
                        tools_ctx["lastvalidation"] = result
                        
                    elif tool_name == "savecashflowresults":
                        tools_ctx["files_created"] = result.get("files_created", 0)
                        # FORZAR FINALIZACIÓN después de guardar
                        feedback = " Archivos del flujo de efectivo guardados exitosamente. CASHFLOW_EXTRACTION_COMPLETED."
                        history.append({"role": "user", "content": feedback})
                        return history, False
                    
                    feedback = f" {tool_name} ejecutado correctamente."
                else:
                    feedback = f" Error en {tool_name}: {result.get('error', 'Error desconocido')}"
                
                history.append({"role": "user", "content": feedback})
                return history, False
                
            except Exception as e:
                print(f" Error ejecutando {tool_name}: {str(e)}")
                error_feedback = f"Error ejecutando {tool_name}: {str(e)}"
                history.append({"role": "user", "content": error_feedback})
                return history, False
        
        except Exception as e:
            print(f" Error en execute_react_step: {str(e)}")
            return history, False
    
    def generate_enhanced_analysis(self, question: str, tools_ctx: Dict[str, Any]) -> str:
        """NUEVA FUNCIÓN: Genera análisis extendido y detallado con LLM especializado"""
        try:
            extraction = tools_ctx.get("lastextraction", {})
            validation = tools_ctx.get("lastvalidation", {})
            
            if not extraction or not extraction.get("success"):
                return "No se pudieron extraer datos de flujos de efectivo del documento."
            
            # Extraer el contenido real del PDF
            text = extraction.get("text", "")
            confidence = validation.get("confidence", 0.8)
            quality = validation.get("quality", "unknown")
            financial_data = extraction.get("financial_data", {})
            
            if not text or len(text.strip()) < 200:
                return "El contenido extraído de flujos de efectivo es insuficiente para realizar un análisis detallado."
            
            # ANÁLISIS INTELIGENTE CON LLM - PROMPT ESPECIALIZADO
            analysis_prompt = f"""
Actúa como un analista financiero senior especializado en banca internacional con 15 años de experiencia. 

Analiza de forma EXHAUSTIVA y DETALLADA el siguiente contenido extraído del estado de flujos de efectivo de GarantiBank International N.V.:

CONTENIDO EXTRAÍDO DEL PDF:
{text[:4000]}

DATOS FINANCIEROS ESPECÍFICOS IDENTIFICADOS:
{json.dumps(financial_data, indent=2) if financial_data else "No se identificaron datos específicos"}

INSTRUCCIONES PARA ANÁLISIS PROFESIONAL:
1. **Estructura tu análisis en las siguientes secciones obligatorias:**
   - Resumen ejecutivo de la posición de liquidez
   - Análisis detallado de actividades operativas
   - Evaluación de actividades de inversión
   - Análisis de actividades de financiación
   - Análisis de variaciones en efectivo y equivalentes
   - Ratios de liquidez y gestión de efectivo (si calculable)
   - Identificación de riesgos y oportunidades
   - Comparación con períodos anteriores (si disponible)
   - Conclusiones estratégicas y recomendaciones

2. **Utiliza ÚNICAMENTE las cifras y datos presentes en el texto extraído**
   - NO inventes números que no aparezcan en el contenido
   - Si una cifra no está disponible, menciona específicamente que "requiere análisis adicional"
   - Cita cifras exactas cuando estén disponibles

3. **Proporciona interpretación profesional y contextual:**
   - Explica el significado de las tendencias observadas
   - Relaciona los flujos con la estrategia bancaria
   - Identifica implicaciones para la solidez financiera
   - Menciona aspectos regulatorios relevantes si aplicables

4. **Formato profesional:**
   - Usa terminología técnica apropiada para banca
   - Estructura clara con subtítulos
   - Párrafos bien desarrollados
   - Longitud objetivo: 700-900 palabras

5. **Enfoque específico bancario:**
   - Considera las particularidades de las entidades financieras
   - Analiza impacto en ratios de liquidez bancarios
   - Evalúa calidad de los flujos operativos
   - Comenta sobre gestión de riesgo de liquidez

Genera un análisis que demuestre expertise profesional y que sea útil tanto para stakeholders internos como externos de la entidad bancaria.
"""

            try:
                # Usar el LLM para generar análisis inteligente
                analysis_response = self.chat_client.chat([
                    {"role": "system", "content": "Eres un analista financiero senior especializado en banca internacional con amplia experiencia en análisis de flujos de efectivo."},
                    {"role": "user", "content": analysis_prompt}
                ], max_tokens=2000)  # Aumentado para respuestas más extensas
                
                # Construir respuesta final con información técnica
                response_parts = [
                    " **ANÁLISIS PROFESIONAL DE FLUJOS DE EFECTIVO - GarantiBank International N.V.**",
                    "=" * 85,
                    "",
                    analysis_response,
                    "",
                    "###  **INFORMACIÓN TÉCNICA DEL ANÁLISIS**",
                    f"• **Calidad de extracción**: {quality.title()} (puntuación: {validation.get('score', 0)}/100)",
                    f"• **Confianza en datos**: {confidence:.1%}",
                    f"• **Caracteres analizados**: {len(text):,} del documento original",
                    f"• **Páginas procesadas**: {len(extraction.get('pages_processed', []))} páginas del estado financiero",
                    f"• **Datos financieros específicos**: {len(financial_data)} categorías identificadas" if financial_data else "• **Datos financieros**: Análisis basado en contenido textual",
                    "• **Metodología**: Extracción automática + análisis con IA especializada en banca",
                    "• **Fuente**: Estado de flujos de efectivo consolidado de GarantiBank International N.V.",
                    "",
                    "=" * 85,
                    " *Análisis generado por sistema de IA especializada en análisis financiero bancario*"
                ]
                
                return "\n".join(response_parts)
                
            except Exception as llm_error:
                print(f"Error en análisis LLM: {str(llm_error)}")
                # Fallback: análisis básico si el LLM falla
                return self.generate_fallback_analysis(text, confidence, quality, financial_data)
                
        except Exception as e:
            return f"Error al generar análisis específico de flujos de efectivo: {str(e)}"

    def generate_fallback_analysis(self, text: str, confidence: float, quality: str, financial_data: Dict) -> str:
        """Análisis de respaldo basado en extracción de datos específicos del texto"""
        
        response_parts = []
        response_parts.append(" **ANÁLISIS DE FLUJOS DE EFECTIVO - GarantiBank International N.V.**")
        response_parts.append("=" * 75)
        
        text_lower = text.lower()
        
        # Análisis por secciones con datos específicos encontrados
        response_parts.append("\n###  **ACTIVIDADES OPERATIVAS**")
        if "operating" in text_lower or "operativ" in text_lower:
            if financial_data.get('operating_cash'):
                amounts = financial_data['operating_cash']
                response_parts.append(f"• Flujos operativos identificados: {amounts} (miles de euros)")
            else:
                response_parts.append("• Los flujos de actividades operativas han sido identificados en el documento")
            
            response_parts.append("• Las actividades operativas reflejan la capacidad del banco para generar efectivo")
            response_parts.append("• Incluye depósitos de clientes, préstamos, y operaciones comerciales principales")
        else:
            response_parts.append("• Los flujos operativos requieren análisis adicional detallado")
        
        response_parts.append("\n###  **ACTIVIDADES DE INVERSIÓN**")
        if "investing" in text_lower or "inversión" in text_lower:
            if financial_data.get('investing_cash'):
                amounts = financial_data['investing_cash']
                response_parts.append(f"• Flujos de inversión identificados: {amounts} (miles de euros)")
            else:
                response_parts.append("• Actividades de inversión detectadas en el estado financiero")
                
            response_parts.append("• Incluye inversiones en valores, adquisiciones de activos fijos")
            response_parts.append("• Refleja la estrategia de crecimiento y expansión del banco")
        else:
            response_parts.append("• Las actividades de inversión requieren análisis específico adicional")
        
        response_parts.append("\n###  **ACTIVIDADES DE FINANCIACIÓN**")
        if "financing" in text_lower or "financiación" in text_lower:
            if financial_data.get('financing_cash'):
                amounts = financial_data['financing_cash']
                response_parts.append(f"• Flujos de financiación identificados: {amounts} (miles de euros)")
            else:
                response_parts.append("• Actividades de financiación detectadas en el documento")
                
            response_parts.append("• Incluye emisión de deuda, dividendos pagados, y cambios en capital")
            response_parts.append("• Indica la política de estructura de capital del banco")
        else:
            response_parts.append("• Las actividades de financiación requieren análisis detallado")
        
        response_parts.append("\n###  **POSICIÓN DE EFECTIVO**")
        if financial_data.get('net_change_cash'):
            amounts = financial_data['net_change_cash']
            response_parts.append(f"• Variación neta en efectivo: {amounts} (miles de euros)")
        
        response_parts.append("• La gestión de liquidez es crítica para operaciones bancarias")
        response_parts.append("• El efectivo y equivalentes garantizan solvencia operacional")
        
        response_parts.append("\n###  **CONCLUSIONES BASADAS EN DATOS EXTRAÍDOS**")
        response_parts.append(f"• **Calidad del análisis**: {quality.title()} con {confidence:.1%} de confianza")
        response_parts.append(f"• **Contenido procesado**: {len(text):,} caracteres de estado financiero")
        
        if financial_data:
            total_categories = len(financial_data)
            response_parts.append(f"• **Datos específicos**: {total_categories} categorías financieras identificadas")
        else:
            response_parts.append("• **Recomendación**: Se requiere acceso a cifras numéricas específicas para análisis cuantitativo completo")
        
        response_parts.append("\n• **Metodología**: Análisis automatizado basado en contenido extraído del PDF")
        response_parts.append("• **Fuente**: Estado de flujos de efectivo de GarantiBank International N.V.")
        response_parts.append("• **Nota**: Para análisis más profundo se recomienda revisión manual de cifras específicas")
        
        return "\n".join(response_parts)

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
        description="CashFlows Agent AUTÓNOMO con Análisis Detallado - Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
  python agents/cashflows_agent.py                    # Usa configuración predefinida
  python agents/cashflows_agent.py --pdf otro.pdf    # Sobreescribe PDF

CARACTERÍSTICAS AVANZADAS:
  - Análisis detallado de 700-900 palabras generado por LLM especializado
  - Extracción avanzada de datos financieros específicos
  - Validación mejorada con puntuación de calidad
  - Chunks semánticos inteligentes por secciones
  - Fallback robusto con análisis basado en datos extraídos

MEJORAS IMPLEMENTADAS:
  - Búsqueda ampliada de páginas relevantes
  - Identificación automática de cifras financieras
  - Análisis profesional con terminología bancaria
  - Informes de calidad extendidos
  - Respuestas estructuradas y contextualizadas
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
    print(" CashFlows Agent v4.0 AUTÓNOMO Multi-Agent - Análisis Detallado")
    print(f" PDF: {args.pdf}")
    print(f" Salida: {args.out}")
    print(f" Groq/Azure OpenAI: Configuración dual")
    print(f" Max steps: {args.maxsteps}")
    print(" CARACTERÍSTICAS: Análisis extenso con LLM, extracción avanzada, respuestas 700-900 palabras")
    
    try:
        # VERIFICAR PDF
        pdf_path = Path(args.pdf)
        output_dir = Path(args.out)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not pdf_path.exists():
            print(f" Error: PDF no encontrado en {pdf_path}")
            return
        
        # CREAR AGENTE Y EJECUTAR
        agent = CashFlowsREACTAgent()
        
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
            print(" Análisis detallado con LLM especializado completado")
        else:
            print(f" Error: {result.get('error_details', 'Error desconocido')}")
        
        print(" Análisis de flujos de efectivo completado!")
        print(" CashFlowsREACTAgent con análisis detallado disponible para sistema multi-agente")
        
    except Exception as e:
        print(f" Error durante la ejecución: {e}")
        raise

if __name__ == "__main__":
    main()
