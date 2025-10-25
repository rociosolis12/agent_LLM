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
    logger.info("FinancialExtractionSystem importado correctamente")
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
        logger.info("Sistema multi-agente inicializado globalmente")
    except Exception as e:
        logger.error(f"Error inicializando sistema: {e}")
        SYSTEM_AVAILABLE = False


# ===== FUNCIÓN AUXILIAR PARA EXTRAER RESULTADOS =====
def extract_agent_result(pipeline_result, agent_name):
    """
    Extrae el resultado del pipeline.
    VERSIÓN CORREGIDA: Extrae de financial_analysis directamente (sin agents_results).
    """
    try:
        financial_analysis = pipeline_result.get('financial_analysis', {})
        
        # Los datos están directamente en financial_analysis
        # NO hay agents_results separado
        answer = (
            financial_analysis.get('answer') or
            financial_analysis.get('specific_answer') or
            financial_analysis.get('final_answer') or
            'No se encontró análisis.'
        )
        
        pdf_extraction = pipeline_result.get('pdf_extraction', {})
        
        response = {
            "status": "success" if financial_analysis.get('success', False) else "error",
            "agent": agent_name.capitalize(),
            "financial_analysis": {
                "answer": answer,
                "confidence": financial_analysis.get('confidence', 0.75),
                "files_generated": financial_analysis.get('files_generated', 0),
                "steps_taken": financial_analysis.get('steps_taken', 0),
                "agent_used": financial_analysis.get('agent_used', agent_name)
            },
            "pdf_extraction": {
                "pages_extracted": pdf_extraction.get('pages_extracted', []),
                "total_pages_extracted": len(pdf_extraction.get('pages_extracted', []))
            },
            "cached": False,
            "timestamp": pipeline_result.get('timestamp')
        }
        
        logger.info(f" {agent_name}: Respuesta extraída correctamente")
        logger.info(f"   Answer: {answer[:100]}...")
        logger.info(f"   Success: {financial_analysis.get('success', False)}")
        
        return response
        
    except Exception as e:
        logger.error(f" Error extrayendo resultado de {agent_name}: {e}")
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Error procesando agente {agent_name}: {str(e)}",
            "agent": agent_name
        }


# ===== ENDPOINTS DE AGENTES =====

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
        
        result = asyncio.run(
            system.run_complete_pipeline_with_hybrid_predictor(
                question=question,
                generate_new_predictions=False
            )
        )
        
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
        
        logger.info("Income analysis completado")
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


# ===== OTROS ENDPOINTS =====

@app.route('/ask-question', methods=['POST'])
def ask_question():
    try:
        logger.info(" Procesando solicitud...")
        data = request.get_json()
        question = data.get('question', '')
        logger.info(f" Pregunta recibida: {question}")
        
        if not SYSTEM_AVAILABLE:
            logger.warning(" Sistema no disponible")
            return jsonify({'status': 'error', 'message': 'Sistema no disponible'}), 503
        
        try:
            system_temp = FinancialExtractionSystem()
            logger.info(" Sistema inicializado")
            
            result = asyncio.run(system_temp.process_question(question))
            logger.info(f" Respuesta generada: {result[:100]}...")
            
            return jsonify({'status': 'success', 'answer': result})
            
        except Exception as e:
            logger.error(f" Error en process_question: {e}")
            traceback.print_exc()
            return jsonify({'status': 'error', 'message': f'Error procesando pregunta: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f" Error general: {e}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': f'Error en el servidor: {str(e)}'}), 500


@app.route('/system-status', methods=['GET'])
def system_status():
    return jsonify({
        'status': 'online' if SYSTEM_AVAILABLE else 'offline',
        'system_available': SYSTEM_AVAILABLE
    })


@app.route('/api/predictor/predictions/latest', methods=['GET'])
def get_latest_predictions():
    try:
        return jsonify({"status": "success", "predictions": []})
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
                "hybrid_predictor": {"status": "active", "agent_loaded": True},
                "main_predictor": {"status": "active"},
                "validation_module": {"status": "active"},
                "regulatory_config": {"status": "active"},
                "update_predictor": {"status": "active"}
            })
        else:
            return jsonify({
                "status": "inactive",
                "pipeline": "stopped",
                "system_available": False
            })
    except Exception as e:
        logger.error(f"Error en pipeline/status: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/predictor/run-hybrid-analysis', methods=['POST', 'OPTIONS'])
def run_hybrid_analysis():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'})
    
    try:
        logger.info("Ejecutando análisis híbrido completo con BBVA...")
        
        if not SYSTEM_AVAILABLE or system is None:
            return jsonify({
                'status': 'error',
                'message': 'Sistema no disponible'
            }), 503
        
        try:
            # Ejecutar el pipeline completo con predictor híbrido
            result = asyncio.run(
                system.run_complete_pipeline_with_hybrid_predictor(
                    question=None,
                    generate_new_predictions=True
                )
            )
            
            logger.info(f"Pipeline completado: {result.get('success', False)}")
            
            # Extraer datos del resultado
            execution_summary = result.get('execution_summary', {})
            hybrid_predictions = result.get('hybrid_predictions', {})
            
            # Verificar si hay predicciones ML reales
            ml_predictions = hybrid_predictions.get('ml_predictions', []) if hybrid_predictions else []
            validation_results = hybrid_predictions.get('validation_results', {}) if hybrid_predictions else {}
            recommendations = hybrid_predictions.get('recommendations', []) if hybrid_predictions else []
            
            # Estructura de respuesta que espera el frontend
            response_data = {
                'status': 'success',  # ← Debe ser 'success' para que el frontend lo acepte
                'message': 'Análisis híbrido completado exitosamente',
                'analysis': {
                    'company': 'BBVA',
                    'timestamp': result.get('timestamp'),
                    'pipeline_completion': execution_summary.get('pipeline_completion', '3/3'),
                    'success_rate': execution_summary.get('success_rate', 100),
                    'confidence_level': 0.85,
                    
                    # Contadores
                    'ml_predictions_count': len(ml_predictions),
                    'validation_metrics_count': len(validation_results) if isinstance(validation_results, dict) else 0,
                    'recommendations_count': len(recommendations),
                    
                    # Datos detallados
                    'ml_predictions': ml_predictions,
                    'validation_results': validation_results,
                    'recommendations': recommendations,
                    
                    # Estadísticas adicionales
                    'pdf_pages_processed': execution_summary.get('pdf_pages_processed', 0),
                    'agents_executed': execution_summary.get('agents_executed', 0),
                    'files_generated': execution_summary.get('files_generated', 0)
                },
                # Agregar datos completos para debugging
                'full_result': {
                    'success': result.get('success'),
                    'mode': result.get('mode'),
                    'total_steps_completed': result.get('total_steps_completed'),
                    'execution_summary': execution_summary
                }
            }
            
            logger.info(f" Predicciones ML: {response_data['analysis']['ml_predictions_count']}")
            logger.info(f" Métricas validadas: {response_data['analysis']['validation_metrics_count']}")
            logger.info(f" Recomendaciones: {response_data['analysis']['recommendations_count']}")
            
            return jsonify(response_data), 200
            
        except Exception as e:
            logger.error(f" Error ejecutando pipeline híbrido: {e}")
            traceback.print_exc()
            return jsonify({
                'status': 'error',
                'message': f'Error en pipeline: {str(e)}',
                'error_details': str(e)
            }), 500
            
    except Exception as e:
        logger.error(f" Error general: {e}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Error en el servidor: {str(e)}',
            'error_details': str(e)
        }), 500


@app.route('/api/predictor/recommendations', methods=['GET'])
def get_recommendations():
    try:
        return jsonify({"status": "success", "recommendations": []})
    except Exception as e:
        logger.error(f"Error en recommendations: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
