"""
financial_coordinator.py - Coordinador Principal Multi-Agente Financiero

VERSIÓN COMPLETA con Integración del Extractor PDF y Agente Predictor
Incluye métodos async, estructuración de datos para predictor, y uso del PDF extraído
"""

import asyncio
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import logging

from config import (
    DATA_INPUT_DIR, DATA_OUTPUT_DIR, FINANCIAL_AGENTS_CONFIG,
    PDF_EXTRACTOR_CONFIG, PREDICTOR_AGENT_CONFIG, get_pdf_paths
)

class FinancialCoordinator:
    """Coordinador principal que gestiona todos los agentes del sistema"""
    
    def __init__(self):
        self.agents = {}
        
        # 🔥 KEYWORDS ACTUALIZADOS CON TÉRMINOS PREDICTIVOS
        self.agent_keywords = {
            'balance': [
                'activos', 'pasivos', 'patrimonio', 'balance', 'assets', 'liabilities',
                'total equity', 'total activos', 'total assets', 'total patrimonio',
                'balance sheet', 'liquidez', 'solvencia', 'financiera', 'financial position',
                'situación financiera', 'estado financiero'
            ],
            'income': [
                'ingresos', 'beneficio', 'ganancia', 'pérdida', 'resultado', 'income',
                'profit', 'revenue', 'ventas', 'gastos', 'expenses', 'rentabilidad',
                'margen', 'beneficio neto', 'personal', 'net income', 'earnings',
                'cuenta de resultados', 'comprehensive income'
            ],
            'cashflows': [
                'efectivo', 'flujos', 'cash', 'tesorería', 'liquidez', 'operaciones',
                'inversión', 'financiación', 'cash flows', 'movimientos',
                'flujos de efectivo', 'cash flow', 'treasury', 'dinero',
                'operating cash', 'free cash flow'
            ],
            'equity': [
                'capital', 'reservas', 'dividendos', 'patrimonio', 'equity',
                'capital social', 'retained earnings', 'cambios en patrimonio',
                'share capital', 'distribución', 'aportes', 'accionistas',
                'changes in equity', 'shareholders'
            ]
        }

        self.session_data = {
            "session_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "questions_processed": 0,
            "agents_used": set(),
            "conversation_history": []
        }

        # Setup logging
        self.logger = self._setup_logger()
        
        # Cargar agentes
        self.predictor_available = self._try_load_predictor()
        self._load_agents()

    def _setup_logger(self):
        """Configurar logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def _try_load_predictor(self) -> bool:
        """🔥 NUEVO: Intenta cargar PredictorAgent con configuración actualizada"""
        try:
            # Verificar configuración
            if not PREDICTOR_AGENT_CONFIG.get('enabled', False):
                self.logger.info("🔮 PredictorAgent deshabilitado en configuración")
                return False
            
            from predictor_agent import PredictorAgent
            self.predictor_class = PredictorAgent
            self.logger.info("✅ PredictorAgent cargado exitosamente")

            # 🔥 KEYWORDS PREDICTIVOS ACTUALIZADOS
            self.agent_keywords['predictor'] = [
                'predicción', 'forecast', 'tendencia futura', 'proyección', 'escenario',
                'riesgo futuro', 'crecimiento esperado', 'estimación futura', 'previsión',
                'análisis prospectivo', 'modeling', 'simulación', 'futuro', 'predictivo',
                'proyectar', 'anticipar', 'evaluar riesgos', 'tendencias', 'próximos años'
            ]

            return True
            
        except ImportError as e:
            self.logger.warning(f"⚠️ PredictorAgent no disponible: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Error cargando PredictorAgent: {str(e)}")
            return False

    def _load_agents(self) -> bool:
        """🔥 ACTUALIZADO: Carga agentes con configuración del PDF extraído"""
        self.logger.info("🔧 Cargando agentes especializados...")
        
        try:
            import sys
            sys.path.append(str(Path(__file__).parent / "agents"))

            from balance_agent import BalanceREACTAgent
            from income_agent import IncomeREACTAgent
            from cashflows_agent import CashFlowsREACTAgent
            from equity_agent import EquityREACTAgent

            # 🔥 CONFIGURACIÓN ACTUALIZADA: Agentes usan PDF extraído

            self.agents = {
                'balance': BalanceREACTAgent(),
                'income': IncomeREACTAgent(),
                'cashflows': CashFlowsREACTAgent(),
                'equity': EquityREACTAgent()
            }

            # 🔥 AGREGAR PREDICTOR si está disponible
            if self.predictor_available:
                predictor_config = PREDICTOR_AGENT_CONFIG
                self.agents['predictor'] = self.predictor_class(config=predictor_config)
                self.logger.info("✅ Predictor Agent cargado con configuración completa")

            self.logger.info("✅ Balance Agent cargado (usando PDF extraído)")
            self.logger.info("✅ Income Agent cargado (usando PDF extraído)")
            self.logger.info("✅ CashFlows Agent cargado (usando PDF extraído)")
            self.logger.info("✅ Equity Agent cargado (usando PDF extraído)")

            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error cargando agentes: {str(e)}")
            return False

    def route_question(self, question: str) -> tuple[str, float]:
        """Enrutamiento básico por keywords"""
        question_lower = question.lower()
        scores = {}

        for agent_type, keywords in self.agent_keywords.items():
            # Solo considerar predictor si está disponible
            if agent_type == 'predictor' and not self.predictor_available:
                continue

            score = sum(1 for keyword in keywords if keyword in question_lower)
            scores[agent_type] = score / len(keywords) if keywords else 0

        if not scores:
            return 'income', 0.5

        best_agent = max(scores, key=scores.get)
        confidence = scores[best_agent]

        # Fallback al income agent si la confianza es muy baja
        if confidence < 0.1:
            best_agent = 'income'
            confidence = 0.5

        return best_agent, confidence

    def route_question_improved(self, question: str) -> tuple[str, float]:
        """🔥 ENRUTAMIENTO MEJORADO con pesos específicos y detección predictiva"""
        question_lower = question.lower()
        scores = {}

        # 🔥 PESOS ESPECÍFICOS ACTUALIZADOS
        specific_weights = {
            # Balance Sheet terms
            'activos': ('balance', 4.0),  
            'total activos': ('balance', 5.0),  
            'total de activos': ('balance', 5.0),  
            'total assets': ('balance', 3.5),
            'patrimonio neto': ('balance', 3.0),
            'total patrimonio': ('balance', 3.5),
            'situación financiera': ('balance', 3.0),
            'balance sheet': ('balance', 3.0),
            'financial position': ('balance', 3.0),
            
            # Income Statement terms
            'beneficio neto': ('income', 3.5),
            'net income': ('income', 3.5),
            'comprehensive income': ('income', 3.5),
            'cuenta de resultados': ('income', 3.0),
            'rentabilidad': ('income', 2.5),
            'ingresos totales': ('income', 2.5),
            
            # Cash Flow terms
            'flujos de efectivo': ('cashflows', 3.5),
            'cash flow': ('cashflows', 3.5),
            'cash flows': ('cashflows', 3.5),
            'free cash flow': ('cashflows', 3.0),
            'operating cash': ('cashflows', 2.5),
            
            # Equity terms
            'cambios en patrimonio': ('equity', 3.5),
            'changes in equity': ('equity', 3.5),
            'capital social': ('equity', 2.5),
            'dividendos': ('equity', 2.5),
            'retained earnings': ('equity', 2.5)
        }

        # 🔥 AGREGAR PESOS PREDICTIVOS si está disponible
        if self.predictor_available:
            predictor_weights = {
                'predicción': ('predictor', 4.0),
                'forecast': ('predictor', 4.0),
                'tendencia futura': ('predictor', 4.0),
                'proyección': ('predictor', 3.5),
                'escenarios futuros': ('predictor', 3.5),
                'riesgo futuro': ('predictor', 3.5),
                'crecimiento esperado': ('predictor', 3.0),
                'estimación futura': ('predictor', 3.0),
                'análisis prospectivo': ('predictor', 3.5),
                '¿qué pasará': ('predictor', 3.0),
                'próximos años': ('predictor', 2.5),
                'evaluación riesgos': ('predictor', 3.0)
            }
            specific_weights.update(predictor_weights)

        # Calcular scores por términos específicos
        for term, (agent, weight) in specific_weights.items():
            if term in question_lower:
                scores[agent] = scores.get(agent, 0) + weight

        # Si no hay coincidencias específicas, usar método básico
        if not scores:
            return self.route_question(question)

        # Seleccionar agente con mayor puntuación
        selected_agent = max(scores, key=scores.get)
        max_possible_score = 4.0  # Score máximo posible
        confidence = min(scores[selected_agent] / max_possible_score, 1.0)

        self.logger.info(f"🎯 Enrutamiento: {question[:50]}... → {selected_agent} (confianza: {confidence:.2f})")
        
        return selected_agent, confidence

    # 🔥 NUEVO MÉTODO: Procesar request general (para pipeline)
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesar request general del pipeline (sin pregunta específica)
        """
        try:
            self.logger.info("📊 Procesando request general del pipeline")
            
            # Verificar que el PDF extraído existe
            if not self._validate_extracted_pdf():
                return {
                    "success": False,
                    "error": "PDF extraído no disponible para procesamiento"
                }

            # Procesar agentes especializados en paralelo
            results = await self._process_specialized_agents_parallel(request)
            
            # Estructurar datos para predictor
            structured_data = self._structure_data_for_predictor(results)
            
            success_count = sum(1 for r in results.values() if r and r.get('success', False))
            total_agents = len(results)
            
            return {
                "success": success_count >= (total_agents * 0.5),  # Al menos 50% exitosos
                "agents_results": results,
                "structured_for_predictor": structured_data,
                "success_rate": (success_count / total_agents) * 100 if total_agents > 0 else 0,
                "successful_agents": success_count,
                "total_agents": total_agents,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error en process_request: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _process_specialized_agents_parallel(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Procesar agentes especializados en paralelo"""
        tasks = []
        agent_names = ['balance', 'income', 'cashflows', 'equity']
        
        pdf_path = self._get_extracted_pdf_path()
        
        for agent_name in agent_names:
            if agent_name in self.agents:
                task = self._process_agent_safe(agent_name, request, pdf_path)
                tasks.append(task)
        
        # Ejecutar en paralelo con timeout
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Mapear resultados
        agents_results = {}
        for i, agent_name in enumerate(agent_names):
            if i < len(results):
                result = results[i]
                if isinstance(result, Exception):
                    agents_results[agent_name] = {
                        "success": False,
                        "error": str(result),
                        "agent": agent_name
                    }
                else:
                    agents_results[agent_name] = result
            else:
                agents_results[agent_name] = {
                    "success": False,
                    "error": "Agent not processed",
                    "agent": agent_name
                }
        
        return agents_results

    async def _process_agent_safe(self, agent_name: str, request: Dict[str, Any], pdf_path: str) -> Dict[str, Any]:
        """Procesar agente individual de forma segura"""
        try:
            agent = self.agents[agent_name]
            
            # Simular procesamiento async (adaptar según implementación real)
            if hasattr(agent, 'run_final_financial_extraction_agent'):
                result = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    agent.run_final_financial_extraction_agent,
                    pdf_path
                )
            else:
                result = {"success": False, "error": "Agent method not found"}
            
            return {
                "success": result.get("status") == "task_completed",
                "agent": agent_name,
                "data": result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "agent": agent_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    
    async def process_question(self, question: str, pdf_path: Optional[str] = None) -> Dict[str, Any]:
        """🔥 ACTUALIZADO: Procesar pregunta específica con PDF extraído"""
        self.logger.info(f"❓ Procesando pregunta: {question}")

        # 1. Enrutar pregunta
        selected_agent, confidence = self.route_question_improved(question)
        self.logger.info(f"📍 Pregunta enrutada a: {selected_agent} (confianza: {confidence:.2f})")

        # 2. Verificar agente disponible
        if selected_agent not in self.agents:
            return self._handle_agent_unavailable(selected_agent, question)

        # 3. Determinar PDF (priorizar PDF extraído)
        pdf_to_use = self._get_extracted_pdf_path() or self._determine_pdf_path(pdf_path)
        if not pdf_to_use:
            return self._handle_no_pdf_error()

        # 4. Procesar predictor con flujo especial
        if selected_agent == 'predictor' and self.predictor_available:
            return await self._process_prediction_question(question, pdf_to_use, confidence)

        # 5. Ejecutar agente especializado
        self.logger.info(f"🚀 Ejecutando {selected_agent} agent...")
        agent_response = await self._execute_agent_async(selected_agent, question, pdf_to_use)

        # 6. Procesar respuesta
        final_response = self._forward_agent_response(question, selected_agent, confidence, agent_response)
        
        # 7. Actualizar estadísticas
        self._update_session_stats(selected_agent, question, final_response)

        return final_response

    async def process_question_with_predictions(self, question: str, pdf_path: Optional[str] = None) -> Dict[str, Any]:
        """🔥 ACTUALIZADO: Procesar pregunta con análisis predictivo"""
        if not self.predictor_available:
            self.logger.warning("⚠️ Predicciones no disponibles - ejecutando solo análisis histórico")
            return await self.process_question(question, pdf_path)

        self.logger.info("🔮 Procesando pregunta con análisis predictivo...")

        # 1. Obtener respuesta histórica normal
        historical_response = await self.process_question(question, pdf_path)

        if not historical_response.get("success", False):
            return {
                **historical_response,
                "prediction_status": "skipped",
                "prediction_reason": "Error en análisis histórico"
            }

        # 2. Ejecutar todos los agentes para recopilar datos completos
        pdf_to_use = self._get_extracted_pdf_path() or self._determine_pdf_path(pdf_path)
        self.logger.info("📊 Recopilando datos históricos completos para predicción...")
        
        historical_results = {}
        for agent_name in ['balance', 'income', 'cashflows', 'equity']:
            if agent_name in self.agents:
                self.logger.info(f"🔄 Ejecutando {agent_name} agent para datos históricos...")
                try:
                    agent = self.agents[agent_name]
                    result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        agent.run_final_financial_extraction_agent,
                        pdf_to_use
                    )
                    historical_results[agent_name] = result
                except Exception as e:
                    self.logger.error(f"❌ Error en {agent_name}: {e}")
                    historical_results[agent_name] = {"status": "error", "error": str(e)}

        # 3. Verificar datos suficientes
        successful_agents = [name for name, result in historical_results.items() 
                           if result.get("status") == "task_completed"]
        
        min_required = PREDICTOR_AGENT_CONFIG.get('required_successful_agents', 3)
        if len(successful_agents) < min_required:
            return {
                **historical_response,
                "prediction_status": "skipped",
                "prediction_reason": f"Datos insuficientes. Se necesitan {min_required}, obtenidos {len(successful_agents)}"
            }

        # 4. Ejecutar predictor
        self.logger.info("🚀 Ejecutando análisis predictivo con datos completos...")
        try:
            predictor_agent = self.agents['predictor']
            
            # Crear input estructurado para predictor
            predictor_input = {
                "financial_data": {
                    "agents_results": historical_results,
                    "structured_for_predictor": self._structure_data_for_predictor(historical_results)
                },
                "config": PREDICTOR_AGENT_CONFIG,
                "request_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "question": question,
                    "data_source": "extracted_pdf_statements"
                }
            }
            
            prediction_result = await predictor_agent.generate_predictions(predictor_input)

            # 5. Combinar respuestas
            combined_response = {
                **historical_response,
                "prediction_included": True,
                "prediction_answer": prediction_result.get("predictions", "No se pudieron generar predicciones"),
                "prediction_status": "success" if prediction_result.get("success") else "error",
                "prediction_data": prediction_result.get("predictions", {}),
                "prediction_horizon": prediction_result.get("prediction_horizon", 12),
                "files_generated_prediction": prediction_result.get("files_generated", 0)
            }

            self.logger.info("✅ Análisis histórico y predictivo completado exitosamente")
            return combined_response

        except Exception as e:
            self.logger.error(f"❌ Error en predicción: {e}")
            return {
                **historical_response,
                "prediction_status": "error",
                "prediction_error": str(e)
            }

    async def _process_prediction_question(self, question: str, pdf_path: str, confidence: float) -> Dict[str, Any]:
        """🔥 ACTUALIZADO: Procesar pregunta dirigida específicamente al predictor"""
        self.logger.info("🔮 Procesando pregunta predictiva - recopilando datos históricos...")

        # 1. Ejecutar todos los agentes históricos para obtener datos base
        historical_results = {}
        for agent_name in ['balance', 'income', 'cashflows', 'equity']:
            if agent_name in self.agents:
                self.logger.info(f"📊 Ejecutando {agent_name} agent para datos base...")
                try:
                    agent = self.agents[agent_name]
                    result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        agent.run_final_financial_extraction_agent,
                        pdf_path
                    )
                    historical_results[agent_name] = result
                except Exception as e:
                    self.logger.error(f"❌ Error en {agent_name}: {e}")
                    historical_results[agent_name] = {"status": "error", "error": str(e)}

        # 2. Validar datos suficientes
        successful_agents = [name for name, result in historical_results.items() 
                           if result.get("status") == "task_completed"]
        
        min_required = PREDICTOR_AGENT_CONFIG.get('required_successful_agents', 3)
        if len(successful_agents) < min_required:
            return {
                "success": False,
                "error": f"Datos insuficientes para predicción. Se necesitan {min_required} agentes exitosos, obtenidos {len(successful_agents)}",
                "agent_used": "predictor",
                "confidence": confidence,
                "successful_agents": successful_agents,
                "timestamp": datetime.now().isoformat()
            }

        # 3. Ejecutar predictor con datos históricos
        self.logger.info("🚀 Ejecutando predictor con datos históricos completos...")
        try:
            predictor_agent = self.agents['predictor']
            
            # Preparar input para predictor
            predictor_input = {
                "financial_data": {
                    "agents_results": historical_results,
                    "structured_for_predictor": self._structure_data_for_predictor(historical_results)
                },
                "config": PREDICTOR_AGENT_CONFIG,
                "request_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "question": question,
                    "data_source": "extracted_pdf_statements",
                    "prediction_type": "direct_question"
                }
            }
            
            result = await predictor_agent.generate_predictions(predictor_input)
            
            # Formatear como respuesta estándar
            return self._forward_agent_response(question, 'predictor', confidence, {
                "status": "task_completed" if result.get("success") else "error",
                "specific_answer": self._format_prediction_answer(result, question),
                "predictions": result.get("predictions", {}),
                "files_generated": 0,  # Se calcularía según implementación
                "steps_taken": 1,
                "session_id": self.session_data["session_id"]
            })
            
        except Exception as e:
            self.logger.error(f"❌ Error ejecutando predictor: {e}")
            return {
                "success": False,
                "error": f"Error ejecutando predictor: {str(e)}",
                "agent_used": "predictor",
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            }

    def _format_prediction_answer(self, prediction_result: Dict[str, Any], question: str) -> str:
        """Formatear respuesta de predicciones para presentación"""
        if not prediction_result.get("success"):
            return "No se pudieron generar predicciones debido a errores en el procesamiento."
        
        predictions = prediction_result.get("predictions", {})
        if not predictions:
            return "Se procesó la solicitud pero no se generaron predicciones específicas."
        
        # Formato básico de respuesta
        answer_parts = [
            f"Basándome en el análisis de los estados financieros extraídos, he generado {len(predictions)} predicciones:",
            ""
        ]
        
        # Añadir predicciones específicas
        for pred_type, pred_data in predictions.items():
            if isinstance(pred_data, dict):
                pred_name = pred_type.replace('_', ' ').title()
                answer_parts.append(f"• {pred_name}: {pred_data}")
            else:
                answer_parts.append(f"• {pred_type}: {pred_data}")
        
        # Añadir disclaimer
        answer_parts.extend([
            "",
            f"Horizonte de predicción: {prediction_result.get('prediction_horizon', 12)} meses",
            "Nota: Estas predicciones se basan en datos históricos y deben considerarse como estimaciones."
        ])
        
        return "\n".join(answer_parts)

    def should_include_predictions(self, question: str) -> bool:
        """🔥 MEJORADO: Detectar si la pregunta requiere análisis predictivo"""
        if not self.predictor_available:
            return False

        prediction_indicators = [
            'predicción', 'forecast', 'futuro', 'proyección', 'tendencia',
            'escenario', 'estimación', 'crecimiento esperado', 'riesgo futuro',
            'previsión', 'análisis prospectivo', '¿qué pasará', 'próximos años',
            'evaluación riesgos', 'perspectivas', 'outlook', 'anticipar'
        ]

        question_lower = question.lower()
        return any(indicator in question_lower for indicator in prediction_indicators)

    # 🔥 MÉTODOS AUXILIARES ACTUALIZADOS

    def _validate_extracted_pdf(self) -> bool:
        """Validar que el PDF extraído existe y es accesible"""
        extracted_pdf_path = self._get_extracted_pdf_path()
        if not extracted_pdf_path:
            return False
        
        pdf_path = Path(extracted_pdf_path)
        return pdf_path.exists() and pdf_path.stat().st_size > 0

    def _get_extracted_pdf_path(self) -> Optional[str]:
        """Obtener ruta del PDF extraído"""
        pdf_paths = get_pdf_paths()
        extracted_path = pdf_paths.get('output_pdf')
        
        if extracted_path and Path(extracted_path).exists():
            return extracted_path
        return None

    async def _execute_agent_async(self, agent_name: str, question: str, pdf_path: str) -> Dict[str, Any]:
        """Ejecutar agente de forma asíncrona"""
        try:
            agent = self.agents[agent_name]
            
            # Ejecutar en executor para no bloquear
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                agent.run_final_financial_extraction_agent,
                pdf_path,
                question
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error ejecutando {agent_name}: {e}")
            return {
                "success": False,
                "status": "error",
                "error": f"Error ejecutando agente {agent_name}: {str(e)}"
            }

    def _forward_agent_response(self, question: str, agent: str, confidence: float, agent_response: Dict) -> Dict[str, Any]:
        """Procesar respuesta del agente y formatear para retorno - MEJORADO"""
        
        if agent_response.get("status") != "task_completed":
            return {
                "success": False,
                "question": question,
                "error": agent_response.get("error", "El agente no completó la tarea exitosamente"),
                "agent_used": agent,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            }

        # ✅ MEJORA: Extraer respuesta específica más robusta
        answer = (agent_response.get("specific_answer") or 
                agent_response.get("detailed_analysis") or 
                agent_response.get("analysis") or 
                agent_response.get("final_response") or
                "El agente no generó una respuesta específica para la pregunta")
        
        # ✅ MEJORA: Agregar contexto del agente y formatear profesionalmente
        agent_context = {
            "equity": "📊 Estado de Cambios en Patrimonio Neto",
            "cashflows": "💰 Estado de Flujos de Efectivo", 
            "balance": "⚖️ Balance de Situación",
            "income": "📈 Cuenta de Resultados Consolidada"
        }
        
        # Formatear respuesta con contexto profesional
        header = f"{agent_context.get(agent, f'📋 {agent.title()}')} - BBVA 2023\n{'='*60}"
        
        # ✅ MEJORA: Incluir metadata de calidad
        metadata = []
        if agent_response.get("files_generated", 0) > 0:
            metadata.append(f"📁 Archivos generados: {agent_response['files_generated']}")
        if agent_response.get("steps_taken", 0) > 0:
            metadata.append(f"🔄 Pasos ejecutados: {agent_response['steps_taken']}")
        
        metadata_str = " | ".join(metadata) if metadata else ""
        footer = f"\n{'─'*60}\n💡 **Análisis realizado por:** {agent.title()} Agent (Confianza: {confidence:.1%})"
        
        if metadata_str:
            footer += f"\n📋 **Procesamiento:** {metadata_str}"
        
        formatted_answer = f"{header}\n\n{answer}{footer}"

        return {
            "success": True,
            "question": question,
            "answer": formatted_answer,  # ✅ Respuesta formateada profesionalmente
            "agent_used": agent,
            "confidence": confidence,
            "steps_taken": agent_response.get("steps_taken", 0),
            "files_generated": agent_response.get("files_generated", 0),
            "session_id": agent_response.get("session_id", self.session_data["session_id"]),
            "raw_data": agent_response.get("extracted_data", {}),
            "timestamp": datetime.now().isoformat()
        }


    def _structure_data_for_predictor(self, specialized_results: Dict[str, Any]) -> Dict[str, Any]:
        """🔥 NUEVO: Estructurar datos de agentes especializados para el predictor"""
        
        def safe_extract_data(agent_key: str, data_key: str, default=None):
            """Extraer datos de forma segura de los resultados de agentes"""
            agent_result = specialized_results.get(agent_key, {})
            if not agent_result or not agent_result.get("success", False):
                return default
            
            agent_data = agent_result.get("data", {})
            return agent_data.get(data_key, default) if agent_data else default

        structured_data = {
            "balance_sheet": {
                "available": specialized_results.get('balance', {}).get('success', False),
                "key_metrics": safe_extract_data('balance', 'key_metrics', {}),
                "ratios": safe_extract_data('balance', 'ratios', {}),
                "trends": safe_extract_data('balance', 'trends', {}),
                "total_assets": safe_extract_data('balance', 'total_assets'),
                "total_liabilities": safe_extract_data('balance', 'total_liabilities'),
                "total_equity": safe_extract_data('balance', 'total_equity')
            },
            "income_statement": {
                "available": specialized_results.get('income', {}).get('success', False),
                "revenue_metrics": safe_extract_data('income', 'revenue', {}),
                "profitability": safe_extract_data('income', 'profitability', {}),
                "growth_rates": safe_extract_data('income', 'growth_rates', {}),
                "net_income": safe_extract_data('income', 'net_income'),
                "operating_income": safe_extract_data('income', 'operating_income')
            },
            "equity_changes": {
                "available": specialized_results.get('equity', {}).get('success', False),
                "equity_composition": safe_extract_data('equity', 'composition', {}),
                "retained_earnings": safe_extract_data('equity', 'retained_earnings', {}),
                "equity_ratios": safe_extract_data('equity', 'ratios', {}),
                "dividends_paid": safe_extract_data('equity', 'dividends_paid'),
                "capital_changes": safe_extract_data('equity', 'capital_changes', {})
            },
            "cash_flows": {
                "available": specialized_results.get('cashflows', {}).get('success', False),
                "operating_cashflow": safe_extract_data('cashflows', 'operating', {}),
                "investing_cashflow": safe_extract_data('cashflows', 'investing', {}),
                "financing_cashflow": safe_extract_data('cashflows', 'financing', {}),
                "free_cashflow": safe_extract_data('cashflows', 'free_cashflow', {}),
                "net_cash_change": safe_extract_data('cashflows', 'net_cash_change')
            },
            "data_quality": self._assess_data_quality(specialized_results),
            "extraction_metadata": {
                "pdf_source": "extracted_financial_statements",
                "pages_processed": "54-60",
                "extraction_timestamp": datetime.now().isoformat(),
                "data_completeness": self._calculate_data_completeness(specialized_results)
            },
            "timestamp": datetime.now().isoformat()
        }

        return structured_data

    def _assess_data_quality(self, specialized_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluar calidad de datos para predicciones"""
        total_agents = len(specialized_results)
        successful_agents = sum(1 for result in specialized_results.values() 
                              if result and result.get('success', False))
        
        quality_score = successful_agents / total_agents if total_agents > 0 else 0
        
        return {
            "completion_rate": quality_score,
            "successful_agents": successful_agents,
            "total_agents": total_agents,
            "quality_score": quality_score,
            "ready_for_prediction": successful_agents >= PREDICTOR_AGENT_CONFIG.get('required_successful_agents', 3),
            "data_reliability": "high" if quality_score >= 0.8 else "medium" if quality_score >= 0.5 else "low",
            "missing_agents": [name for name, result in specialized_results.items() 
                             if not result or not result.get('success', False)]
        }

    def _calculate_data_completeness(self, specialized_results: Dict[str, Any]) -> float:
        """Calcular completitud de datos extraídos"""
        total_expected_fields = 0
        completed_fields = 0
        
        expected_fields_by_agent = {
            'balance': ['total_assets', 'total_liabilities', 'total_equity', 'key_metrics'],
            'income': ['net_income', 'revenue', 'operating_income', 'profitability'],
            'cashflows': ['operating', 'investing', 'financing', 'free_cashflow'],
            'equity': ['composition', 'retained_earnings', 'dividends_paid']
        }
        
        for agent_name, expected_fields in expected_fields_by_agent.items():
            agent_result = specialized_results.get(agent_name, {})
            if agent_result and agent_result.get('success', False):
                agent_data = agent_result.get('data', {})
                for field in expected_fields:
                    total_expected_fields += 1
                    if field in agent_data and agent_data[field] is not None:
                        completed_fields += 1
            else:
                total_expected_fields += len(expected_fields)
        
        return completed_fields / total_expected_fields if total_expected_fields > 0 else 0

    def _determine_pdf_path(self, provided_path: Optional[str]) -> Optional[str]:
        """Determinar PDF a usar (priorizar PDF extraído)"""
        # 1. PDF extraído (prioridad)
        extracted_pdf = self._get_extracted_pdf_path()
        if extracted_pdf:
            return extracted_pdf
        
        # 2. PDF proporcionado
        if provided_path and Path(provided_path).exists():
            return provided_path

        # 3. Buscar PDFs en directorio de entrada
        pdf_files = list(DATA_INPUT_DIR.glob("*.pdf"))
        if pdf_files:
            return str(pdf_files[0])

        return None

    def _handle_agent_unavailable(self, agent_name: str, question: str) -> Dict[str, Any]:
        """Manejar error de agente no disponible"""
        return {
            "success": False,
            "question": question,
            "error": f"El agente {agent_name} no está disponible",
            "suggestion": "Verifica que el agente esté correctamente configurado y cargado",
            "available_agents": list(self.agents.keys()),
            "timestamp": datetime.now().isoformat()
        }

    def _handle_no_pdf_error(self) -> Dict[str, Any]:
        """Manejar error de PDF no encontrado"""
        pdf_paths = get_pdf_paths()
        
        return {
            "success": False,
            "error": "No se encontró ningún PDF para analizar",
            "suggestion": f"Asegúrate de que exista el PDF extraído en: {pdf_paths.get('output_pdf', 'Ruta no configurada')}",
            "expected_paths": {
                "extracted_pdf": pdf_paths.get('output_pdf'),
                "input_directory": str(DATA_INPUT_DIR)
            },
            "timestamp": datetime.now().isoformat()
        }

    def _update_session_stats(self, agent_used: str, question: str, response: Dict):
        """Actualizar estadísticas de la sesión"""
        self.session_data["questions_processed"] += 1
        self.session_data["agents_used"].add(agent_used)
        self.session_data["conversation_history"].append({
            "question": question[:100] + "..." if len(question) > 100 else question,
            "agent": agent_used,
            "success": response.get("success", False),
            "confidence": response.get("confidence", 0),
            "timestamp": datetime.now().isoformat()
        })

    # 🔥 MÉTODOS DE ESTADO ACTUALIZADOS

    def get_system_status(self) -> Dict[str, Any]:
        """Obtener estado completo del sistema"""
        pdf_paths = get_pdf_paths()
        
        return {
            "agents_loaded": len(self.agents),
            "total_agents": len(self.agent_keywords),
            "predictor_available": self.predictor_available,
            "pdf_extractor_enabled": PDF_EXTRACTOR_CONFIG.get('agent_enabled', False),
            "session_stats": {
                **self.session_data,
                "agents_used": list(self.session_data["agents_used"])  # Convertir set a list
            },
            "data_directories": {
                "input": str(DATA_INPUT_DIR),
                "output": str(DATA_OUTPUT_DIR),
                "pdf_input": PDF_EXTRACTOR_CONFIG.get('input_path'),
                "pdf_output": PDF_EXTRACTOR_CONFIG.get('output_path')
            },
            "pdf_status": {
                "extracted_pdf_exists": Path(pdf_paths.get('output_pdf', '')).exists() if pdf_paths.get('output_pdf') else False,
                "input_pdf_exists": Path(pdf_paths.get('input_pdf', '')).exists() if pdf_paths.get('input_pdf') else False,
                "extracted_pdf_path": pdf_paths.get('output_pdf'),
                "input_pdf_path": pdf_paths.get('input_pdf')
            },
            "available_agents": list(self.agents.keys()),
            "system_health": self._assess_system_health()
        }

    def _assess_system_health(self) -> Dict[str, Any]:
        """Evaluar salud general del sistema"""
        health_checks = {
            "agents_loaded": len(self.agents) >= 4,  # Al menos 4 agentes base
            "pdf_extracted": bool(self._get_extracted_pdf_path()),
            "predictor_available": self.predictor_available,
            "config_valid": True  # Se asumiría validación de config
        }
        
        overall_health = all(health_checks.values())
        health_score = sum(health_checks.values()) / len(health_checks)
        
        return {
            "overall_healthy": overall_health,
            "health_score": health_score,
            "checks": health_checks,
            "status": "healthy" if overall_health else "degraded" if health_score >= 0.5 else "unhealthy",
            "last_check": datetime.now().isoformat()
        }

    def get_available_agents(self) -> Dict[str, str]:
        """Obtener lista de agentes disponibles con descripciones"""
        agent_descriptions = {
            'balance': 'Especializado en balance de situación, activos, pasivos y patrimonio neto',
            'income': 'Especializado en cuenta de resultados, ingresos, gastos y rentabilidad',
            'cashflows': 'Especializado en flujos de efectivo, operaciones, inversión y financiación',
            'equity': 'Especializado en cambios en patrimonio, capital social y reservas'
        }

        if self.predictor_available:
            agent_descriptions['predictor'] = 'Especializado en predicciones financieras y análisis prospectivo basado en datos históricos'

        return agent_descriptions

    # 🔥 MÉTODO DE VALIDACIÓN COMPLETA
    async def is_ready(self) -> bool:
        """Verificar si el coordinador está listo para funcionar"""
        try:
            # Verificar agentes cargados
            if len(self.agents) < 4:  # Al menos los 4 agentes base
                return False
            
            # Verificar PDF extraído disponible
            if not self._validate_extracted_pdf():
                return False
            
            # Verificar configuración de predictor si está habilitado
            if self.predictor_available:
                if not PREDICTOR_AGENT_CONFIG.get('enabled', False):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error verificando readiness: {e}")
            return False
            
    def debug_routing(self, question: str):
        """Debug: Mostrar puntuación de todos los agentes"""
        question_lower = question.lower()
        print(f"\n🔍 DEBUGGING ROUTING PARA: '{question}'")
        
        scores = {}
        specific_weights = {
            'activos': ('balance', 4.0),
            'total activos': ('balance', 5.0),
            'ingresos': ('income', 3.5),
            'flujos': ('cashflows', 3.0),
            'capital': ('equity', 3.0),
        }
        
        for term, (agent, weight) in specific_weights.items():
            if term in question_lower:
                scores[agent] = scores.get(agent, 0) + weight
                print(f"  ✓ '{term}' encontrado → {agent} (+{weight})")
        
        print(f"\n📊 Puntuaciones finales:")
        for agent, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  {agent}: {score}")
        
        selected_agent, confidence = self.route_question_improved(question)
        print(f"\n🎯 Agente seleccionado: {selected_agent} (confianza: {confidence:.2f})\n")

