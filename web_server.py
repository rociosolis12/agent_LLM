from flask import Flask, jsonify, request
from flask_cors import CORS
import sys
import asyncio
import traceback
import logging
from pathlib import Path
import json  
import os

app = Flask(__name__)

# Ajusta BASE_DIR según la ubicación de este archivo
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "salida"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).parent))



try:
    from main_system import FinancialExtractionSystem
    SYSTEM_AVAILABLE = True
    logger.info(" FinancialExtractionSystem importado correctamente")
except ImportError as e:
    SYSTEM_AVAILABLE = False
    logger.error(f" Error importando FinancialExtractionSystem: {e}")

app = Flask(__name__)
CORS(app)
app.config['DEBUG'] = True

# ===== INSTANCIA GLOBAL DEL SISTEMA =====
system = None
if SYSTEM_AVAILABLE:
    try:
        system = FinancialExtractionSystem()
        logger.info(" Sistema multi-agente inicializado globalmente")
    except Exception as e:
        logger.error(f" Error inicializando sistema: {e}")
        SYSTEM_AVAILABLE = False


@app.route('/ask-question', methods=['POST'])
def ask_question():
    try:
        logger.info(" Procesando solicitud...")
        data = request.get_json()
        question = data.get('question', '')
        logger.info(f" Pregunta recibida: {question}")
        
        if not SYSTEM_AVAILABLE:
            logger.warning(" Sistema no disponible")
            return jsonify({
                'status': 'error',
                'message': 'Sistema no disponible'
            }), 503
        
        try:
            system = FinancialExtractionSystem()
            logger.info(" Sistema inicializado")
            
            result = asyncio.run(system.process_question(question))
            logger.info(f" Respuesta generada: {result[:100]}...")
            
            return jsonify({
                'status': 'success',
                'answer': result
            })
            
        except Exception as e:
            logger.error(f" Error en process_question: {e}")
            traceback.print_exc()
            return jsonify({
                'status': 'error',
                'message': f'Error procesando pregunta: {str(e)}'
            }), 500
            
    except Exception as e:
        logger.error(f" Error general: {e}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Error en el servidor: {str(e)}'
        }), 500

@app.route('/system-status', methods=['GET'])
def system_status():
    return jsonify({
        'status': 'online' if SYSTEM_AVAILABLE else 'offline',
        'system_available': SYSTEM_AVAILABLE
    })

# ===== ENDPOINTS PARA EL PREDICTOR =====

@app.route('/api/predictor/predictions/latest', methods=['GET'])
def get_latest_predictions():
    try:
        return jsonify({
            "status": "success",
            "predictions": []
        })
    except Exception as e:
        logger.error(f"Error en predictions/latest: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/predictor/pipeline/status', methods=['GET'])
def get_pipeline_status():
    try:
        if SYSTEM_AVAILABLE:
            return jsonify({
                "status": "active",
                "pipeline": "running",
                "system_available": True,
                "hybrid_predictor": {
                    "status": "active",
                    "agent_loaded": True,
                    "model": "hybrid_predictor_agent.py"
                },
                "main_predictor": {
                    "status": "active",
                    "module": "main_predictor.py"
                },
                "validation_module": {
                    "status": "active",
                    "module": "validation_module.py"
                },
                "regulatory_config": {
                    "status": "active",
                    "module": "regulatory_config_agent.py"
                },
                "update_predictor": {
                    "status": "active",
                    "module": "update_predictor_agent.py"
                }
            })
        else:
            return jsonify({
                "status": "inactive",
                "pipeline": "stopped",
                "system_available": False,
                "hybrid_predictor": {"status": "inactive"},
                "main_predictor": {"status": "inactive"},
                "validation_module": {"status": "inactive"},
                "regulatory_config": {"status": "inactive"},
                "update_predictor": {"status": "inactive"}
            })
    except Exception as e:
        logger.error(f"Error en pipeline/status: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/predictor/run-hybrid-analysis', methods=['POST', 'OPTIONS'])
def run_hybrid_analysis():
    # Manejar preflight CORS request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
    
    try:
        logger.info(" Ejecutando análisis híbrido completo con BBVA...")
        
        if not SYSTEM_AVAILABLE:
            logger.warning("⚠️ Sistema no disponible")
            return jsonify({
                'status': 'error',
                'message': 'Sistema no disponible'
            }), 503
        
        try:
            # Inicializar el sistema
            system = FinancialExtractionSystem()
            logger.info(" Sistema multi-agente inicializado")
            
            # Ejecutar el pipeline completo con predictor híbrido
            # Este método ejecuta:
            # 1. PDF Extractor (extrae páginas 54-60)
            # 2. Agentes especializados (Balance, Income, CashFlows, Equity)
            # 3. Predictor híbrido (Prophet + XGBoost + LLM + Validación)
            result = asyncio.run(
                system.run_complete_pipeline_with_hybrid_predictor(
                    question=None,  # Sin pregunta específica, análisis completo
                    generate_new_predictions=True  # SIEMPRE generar nuevas predicciones
                )
            )
            
            logger.info(f" Pipeline completado: {result.get('success', False)}")
            
            # Extraer métricas del resultado
            execution_summary = result.get('execution_summary', {})
            hybrid_predictions = result.get('hybrid_predictions', {})
            
            # Preparar respuesta estructurada para el frontend
            response_data = {
                'status': 'success' if result.get('success') else 'error',
                'analysis': {
                    'company': 'BBVA',
                    'timestamp': result.get('timestamp'),
                    'pipeline_completion': execution_summary.get('pipeline_completion', '0/0'),
                    'success_rate': execution_summary.get('success_rate', 0),
                    
                    # Métricas del predictor híbrido
                    'confidence_level': 0.85,  # Calcular del resultado real
                    'ml_predictions_count': execution_summary.get('ml_predictions_count', 0),
                    'validation_metrics_count': execution_summary.get('validation_metrics_count', 0),
                    'recommendations_count': execution_summary.get('recommendations_count', 0),
                    
                    # Datos detallados
                    'ml_predictions': hybrid_predictions.get('ml_predictions', []) if hybrid_predictions else [],
                    'validation_results': hybrid_predictions.get('validation_results', {}) if hybrid_predictions else {},
                    'recommendations': hybrid_predictions.get('recommendations', []) if hybrid_predictions else [],
                    
                    # Estadísticas del pipeline
                    'pdf_pages_processed': execution_summary.get('pdf_pages_processed', 0),
                    'agents_executed': execution_summary.get('agents_executed', 0),
                    'files_generated': execution_summary.get('files_generated', 0)
                }
            }
            
            logger.info(f" Predicciones ML: {response_data['analysis']['ml_predictions_count']}")
            logger.info(f" Métricas validadas: {response_data['analysis']['validation_metrics_count']}")
            logger.info(f" Recomendaciones: {response_data['analysis']['recommendations_count']}")
            
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f" Error ejecutando pipeline híbrido: {e}")
            traceback.print_exc()
            return jsonify({
                'status': 'error',
                'message': f'Error en pipeline: {str(e)}'
            }), 500
            
    except Exception as e:
        logger.error(f" Error general: {e}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Error en el servidor: {str(e)}'
        }), 500



@app.route('/api/predictor/recommendations', methods=['GET'])
def get_recommendations():
    try:
        return jsonify({
            "status": "success",
            "recommendations": []
        })
    except Exception as e:
        logger.error(f"Error en recommendations: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    

# ===== ENDPOINTS PARA LOS 4 AGENTES ESPECIALIZADOS (SIN HARDCODEAR) =====
def load_json_or_fail(filepath):
    logger.info(f"Intentando abrir archivo: {filepath}")
    if not filepath.exists():
        logger.error(f"No existe el archivo: {filepath}")
        return None
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)

def extract_agent_result(pipeline_result, agent_name):
    """
    Extrae el resultado de un agente específico del pipeline completo.
    VERSIÓN CORREGIDA: Soporta múltiples formatos de respuesta de agentes.
    """
    try:
        # Extraer resultados de agentes especializados
        financial_analysis = pipeline_result.get('financial_analysis', {})
        agents_results = financial_analysis.get('agents_results', {})
        agent_data = agents_results.get(agent_name, {})
        
        # Extraer información de extracción PDF
        pdf_extraction = pipeline_result.get('pdf_extraction', {})
        
        # ✅ CORRECCIÓN: Soportar múltiples campos de respuesta
        # Intentar diferentes nombres de campo que usan los agentes
        answer = (
            agent_data.get('final_answer') or           # Nombre genérico
            agent_data.get('specificanswer') or         # Balance agent usa esto
            agent_data.get('specific_answer') or        # Variación snake_case
            agent_data.get('answer') or                 # Nombre simple
            agent_data.get('finalresponse') or          # Otra variación
            agent_data.get('final_response') or         # Snake case
            'No se encontró análisis.'                  # Fallback
        )
        
        # Formato de respuesta estandarizado
        response = {
            "status": "success",
            "agent": agent_name.capitalize(),
            "financial_analysis": {
                "answer": answer,
                "confidence": agent_data.get('confidence', 0.75),
                "files_generated": agent_data.get('files_generated', agent_data.get('filesgenerated', 0)),
                "steps_taken": agent_data.get('total_steps', agent_data.get('steps_taken', agent_data.get('stepstaken', 0)))
            },
            "pdf_extraction": {
                "pages_extracted": pdf_extraction.get('pages_extracted', pdf_extraction.get('pagesextracted', [])),
                "total_pages_extracted": len(pdf_extraction.get('pages_extracted', pdf_extraction.get('pagesextracted', [])))
            },
            "cached": False,  # Datos generados en tiempo real
            "timestamp": pipeline_result.get('timestamp'),
            "raw_agent_data": agent_data  # ✅ AGREGADO: Para debugging
        }
        
        logger.info(f"✅ Respuesta extraída para {agent_name}: {answer[:100]}...")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error extrayendo resultado de {agent_name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Error procesando agente {agent_name}: {str(e)}",
            "agent": agent_name
        }



@app.route('/api/agents/balance-analysis', methods=['POST', 'OPTIONS'])
def balance_analysis():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'})
    
    try:
        logger.info(" Ejecutando análisis de Balance General...")
        
        if not SYSTEM_AVAILABLE or system is None:
            return jsonify({'status': 'error', 'message': 'Sistema no disponible'}), 503
        
        data = request.get_json() or {}
        question = data.get('question', 'Analiza el balance general y estructura de activos')
        
        # Ejecutar pipeline dinámicamente
        result = asyncio.run(
            system.run_complete_pipeline_with_hybrid_predictor(
                question=question,
                generate_new_predictions=False
            )
        )
        
        # Extraer resultado del agente de balance
        response = extract_agent_result(result, 'balance')
        
        logger.info(" Balance analysis completado")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f" Error en balance-analysis: {e}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/agents/income-analysis', methods=['POST', 'OPTIONS'])
def income_analysis():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'})
    
    try:
        logger.info(" Ejecutando análisis de Estado de Resultados...")
        
        if not SYSTEM_AVAILABLE or system is None:
            return jsonify({'status': 'error', 'message': 'Sistema no disponible'}), 503
        
        data = request.get_json() or {}
        question = data.get('question', 'Analiza el estado de resultados y rentabilidad')
        
        result = asyncio.run(
            system.run_complete_pipeline_with_hybrid_predictor(
                question=question,
                generate_new_predictions=False
            )
        )
        
        response = extract_agent_result(result, 'income')
        
        logger.info(" Income analysis completado")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f" Error en income-analysis: {e}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/agents/cashflow-analysis', methods=['POST', 'OPTIONS'])
def cashflow_analysis():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'})
    
    try:
        logger.info(" Ejecutando análisis de Flujos de Efectivo...")
        
        if not SYSTEM_AVAILABLE or system is None:
            return jsonify({'status': 'error', 'message': 'Sistema no disponible'}), 503
        
        data = request.get_json() or {}
        question = data.get('question', 'Analiza los flujos de efectivo operativos y de inversión')
        
        result = asyncio.run(
            system.run_complete_pipeline_with_hybrid_predictor(
                question=question,
                generate_new_predictions=False
            )
        )
        
        response = extract_agent_result(result, 'cashflows')
        
        logger.info(" Cashflow analysis completado")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f" Error en cashflow-analysis: {e}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/agents/equity-analysis', methods=['POST', 'OPTIONS'])
def equity_analysis():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'})
    
    try:
        logger.info(" Ejecutando análisis de Estado de Patrimonio...")
        
        if not SYSTEM_AVAILABLE or system is None:
            return jsonify({'status': 'error', 'message': 'Sistema no disponible'}), 503
        
        data = request.get_json() or {}
        question = data.get('question', 'Analiza el estado de patrimonio y cambios en el capital')
        
        result = asyncio.run(
            system.run_complete_pipeline_with_hybrid_predictor(
                question=question,
                generate_new_predictions=False
            )
        )
        
        response = extract_agent_result(result, 'equity')
        
        logger.info(" Equity analysis completado")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f" Error en equity-analysis: {e}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
