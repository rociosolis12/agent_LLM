"""
Router Inteligente para Preguntas Financieras
Determina qué agente debe responder cada pregunta basándose en análisis semántico
"""

import re
from typing import Dict, List, Optional, Tuple
from config import AGENTS_CONFIG, ROUTER_CONFIG

class FinancialQuestionRouter:
    """Router inteligente que analiza preguntas y las envía al agente correcto"""
    
    def __init__(self):
        self.agents_config = AGENTS_CONFIG
        self.confidence_threshold = ROUTER_CONFIG["confidence_threshold"]
        
        # Diccionario de rutas con análisis semántico mejorado
        self.routing_rules = {
            "balance": {
                "primary_keywords": [
                    # Español - Conceptos de Balance
                    "balance", "activo", "pasivo", "patrimonio neto", "posición financiera",
                    "activos corrientes", "pasivos corrientes", "liquidez", "solvencia",
                    "endeudamiento", "estructura financiera", "capital de trabajo",
                    "inmovilizado", "circulante", "existencias", "inventarios",
                    "deudores", "acreedores", "tesorería disponible", "bancos",
                    
                    # English - Balance Sheet Concepts
                    "balance sheet", "assets", "liabilities", "equity", "financial position", 
                    "current assets", "current liabilities", "working capital",
                    "inventory", "receivables", "payables", "cash and equivalents"
                ],
                "question_patterns": [
                    r"cuál.*total.*activ", r"cuánto.*activ", r"valor.*activ",
                    r"cuál.*total.*pasiv", r"cuánto.*pasiv", r"situación.*financier",
                    r"what.*total.*asset", r"how.*much.*asset", r"financial.*position"
                ],
                "priority": 1
            },
            
            "income": {
                "primary_keywords": [
                    # Español - Conceptos de Resultados
                    "ingresos", "gastos", "beneficio", "pérdidas", "resultado", "margen",
                    "ventas", "facturación", "cifra de negocios", "ingresos ordinarios",
                    "gastos operativos", "gastos financieros", "amortizaciones",
                    "beneficio neto", "beneficio bruto", "ebitda", "ebit",
                    "rentabilidad", "roi", "roe", "roa", "margen operativo",
                    "cuenta de resultados", "estado de resultados","resultado integral",
                    
                    # English - Income Statement Concepts  
                    "revenue", "income", "profit", "loss", "earnings", "sales",
                    "operating income", "expenses", "ebitda", "ebit", "margins",
                    "profitability", "return on", "income statement"
                ],
                "question_patterns": [
                    r"cuál.*benefici", r"cuánto.*benefici", r"resultado.*",
                    r"cuánto.*ingres", r"ventas.*", r"margen.*", r"rentabilidad.*",
                    r"what.*profit", r"how.*much.*revenue", r"net.*income"
                ],
                "priority": 1
            },
            
            "cashflows": {
                "primary_keywords": [
                    # Español - Conceptos de Flujos
                    "flujos", "efectivo", "tesorería", "cash", "liquidez", "caja",
                    "flujos operativos", "actividades operativas", "operación",
                    "flujos de inversión", "actividades de inversión", "inversiones",
                    "flujos de financiación", "actividades de financiación",
                    "efectivo inicial", "efectivo final", "generación de efectivo",
                    "cobros", "pagos", "movimientos de efectivo",
                    
                    # English - Cash Flow Concepts
                    "cash flows", "cash flow statement", "operating cash flows",
                    "investing activities", "financing activities", "cash generation",
                    "free cash flow", "cash receipts", "cash payments"
                ],
                "question_patterns": [
                    r"flujo.*efectiv", r"efectivo.*", r"cash.*flow", r"tesorería.*",
                    r"actividades.*operativ", r"operativ.*cash", r"inversión.*efectiv",
                    r"financiación.*efectiv", r"generación.*efectiv"
                ],
                "priority": 1
            },
            
            "equity": {
                "primary_keywords": [
                    # Español - Conceptos de Patrimonio
                    "patrimonio", "capital", "reservas", "fondos propios",
                    "cambios en patrimonio", "variaciones patrimonio", "movimientos patrimonio",
                    "capital social", "prima de emisión", "reservas legales",
                    "resultados acumulados", "autocartera", "acciones propias",
                    "dividendos", "distribución resultados", "ampliaciones capital",
                    
                    # English - Equity Concepts
                    "equity", "shareholders equity", "changes in equity", "capital stock",
                    "retained earnings", "reserves", "dividends", "treasury shares",
                    "equity movements", "capital contributions"
                ],
                "question_patterns": [
                    r"patrimonio.*", r"capital.*social", r"reservas.*",
                    r"cambios.*patrimonio", r"dividend.*", r"equity.*change"
                ],
                "priority": 1
            }
        }
    
    def route_question(self, question: str) -> Tuple[str, float, Dict]:
        """
        Analiza la pregunta y determina el agente más apropiado
        
        Returns:
            Tuple[agente, confianza, detalles]
        """
        question_lower = question.lower().strip()
        
        if not question_lower:
            return ROUTER_CONFIG["fallback_agent"], 0.0, {"error": "Pregunta vacía"}
        
        # Calcular puntuaciones para cada agente
        agent_scores = self._calculate_agent_scores(question_lower)
        
        # Encontrar el mejor agente
        best_agent, best_score = max(agent_scores.items(), key=lambda x: x[1])
        
        # Detalles del análisis
        routing_details = {
            "question": question,
            "scores": agent_scores,
            "best_agent": best_agent,
            "confidence": best_score,
            "threshold_met": best_score >= self.confidence_threshold
        }
        
        # Si no supera el umbral, usar agente por defecto
        if best_score < self.confidence_threshold:
            fallback_agent = ROUTER_CONFIG["fallback_agent"]
            routing_details["fallback_used"] = True
            routing_details["reason"] = f"Confianza {best_score:.2f} < {self.confidence_threshold}"
            return fallback_agent, best_score, routing_details
        
        return best_agent, best_score, routing_details
    
    def _calculate_agent_scores(self, question: str) -> Dict[str, float]:
        """Calcula puntuaciones para cada agente"""
        scores = {agent: 0.0 for agent in self.routing_rules.keys()}
        
        for agent, rules in self.routing_rules.items():
            score = 0.0
            
            # 1. Puntuación por palabras clave (peso: 0.6)
            keyword_matches = 0
            for keyword in rules["primary_keywords"]:
                if self._contains_whole_word(keyword, question):
                    keyword_matches += 2  # Coincidencia exacta
                elif keyword in question:
                    keyword_matches += 1  # Coincidencia parcial
            
            keyword_score = min(keyword_matches / len(rules["primary_keywords"]) * 0.6, 0.6)
            score += keyword_score
            
            # 2. Puntuación por patrones (peso: 0.4)
            pattern_matches = 0
            for pattern in rules["question_patterns"]:
                if re.search(pattern, question, re.IGNORECASE):
                    pattern_matches += 1
            
            pattern_score = min(pattern_matches / len(rules["question_patterns"]) * 0.4, 0.4)
            score += pattern_score
            
            # 3. Aplicar lógica contextual adicional
            score = self._apply_contextual_boost(question, agent, score)
            
            scores[agent] = round(score, 3)
        
        return scores
    
    def _contains_whole_word(self, word: str, text: str) -> bool:
        """Verifica si una palabra aparece completa en el texto"""
        pattern = r'\b' + re.escape(word.lower()) + r'\b'
        return bool(re.search(pattern, text.lower()))
    
    def _apply_contextual_boost(self, question: str, agent: str, base_score: float) -> float:
        """Aplica boost contextual basado en patrones específicos"""
        boost = 0.0
        
        # Boost para preguntas cuantitativas
        if any(word in question for word in ["cuánto", "cuál", "valor", "importe", "total"]):
            if agent == "balance" and any(word in question for word in ["activo", "pasivo"]):
                boost += 0.1
            elif agent == "income" and any(word in question for word in ["beneficio", "ingreso", "venta"]):
                boost += 0.1
        
        # Boost para preguntas comparativas
        if any(word in question for word in ["comparar", "diferencia", "cambio", "variación"]):
            boost += 0.05
        
        # Boost para preguntas sobre años específicos
        if re.search(r'\b20\d{2}\b', question):
            boost += 0.05
        
        # Boost para ratios financieros
        if any(word in question for word in ["ratio", "índice", "margen"]):
            if agent == "income":
                boost += 0.1
        
        return min(base_score + boost, 1.0)
    
    def get_routing_explanation(self, question: str) -> str:
        """Proporciona explicación del enrutamiento"""
        agent, confidence, details = self.route_question(question)
        
        explanations = {
            "balance": "Esta pregunta se refiere al **balance de situación** (activos, pasivos, patrimonio)",
            "income": "Esta pregunta se refiere a la **cuenta de resultados** (ingresos, gastos, beneficios)", 
            "cashflows": "Esta pregunta se refiere a los **flujos de efectivo** (movimientos de tesorería)",
            "equity": "Esta pregunta se refiere a **cambios en patrimonio** (capital, reservas, dividendos)"
        }
        
        explanation = explanations.get(agent, "Clasificación por análisis contextual")
        
        if details.get("fallback_used"):
            explanation += f"\n⚠️ Confianza baja ({confidence:.2f}), usando agente por defecto."
        else:
            explanation += f"\n✅ Confianza: {confidence:.2f}"
        
        return explanation
    
    def suggest_better_questions(self) -> List[str]:
        """Sugiere ejemplos de mejores preguntas"""
        return [
            "**Balance**: '¿Cuál es el total de activos en 2023?' o '¿Cuánto es el pasivo corriente?'",
            "**Resultados**: '¿Cuál fue el beneficio neto?' o '¿Cuántos fueron los ingresos por ventas?'", 
            "**Flujos**: '¿Qué efectivo generaron las operaciones?' o '¿Cuál es el efectivo final?'",
            "**Patrimonio**: '¿Cómo cambió el capital social?' o '¿Cuánto se pagó en dividendos?'"
        ]
