from flask import Flask, jsonify, request
from flask_cors import CORS
import sys
import asyncio
import traceback
import logging
from pathlib import Path

# Configurar logging detallado
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).parent))

try:
    from main_system import FinancialExtractionSystem  # ✅ CORRECTO: Usar la clase principal
    SYSTEM_AVAILABLE = True
    logger.info("✅ FinancialExtractionSystem importado correctamente")
except ImportError as e:
    SYSTEM_AVAILABLE = False
    logger.error(f"❌ Error importando FinancialExtractionSystem: {e}")

app = Flask(__name__)
CORS(app)
app.config['DEBUG'] = True

@app.route('/ask-question', methods=['POST'])
def ask_question():
    try:
        logger.info("🔄 Procesando solicitud...")
        
        data = request.get_json()
        question = data.get('question', '')
        
        logger.info(f"📨 Pregunta recibida: {question}")
        
        if not SYSTEM_AVAILABLE:
            logger.warning("⚠️ Sistema no disponible")
            return jsonify({
                'answer': f'Sistema procesando: {question}',
                'success': True
            })
        
        # ✅ CORRECTO: Usar FinancialExtractionSystem
        logger.info("🚀 Iniciando sistema financiero completo...")
        system = FinancialExtractionSystem()
        
        logger.info("⚡ Ejecutando pipeline completo...")
        
        # ✅ CORRECTO: Usar asyncio.run con run_complete_pipeline
        pipeline_result = asyncio.run(system.run_complete_pipeline(question))
        
        logger.info("✅ Pipeline ejecutado correctamente")
        logger.info(f"📊 Resultado recibido: {type(pipeline_result)}")
        
        # ✅ CORRECTO: Extraer respuesta según la estructura de main_system.py
        if pipeline_result.get('success', False):
            # La respuesta está en financial_analysis
            financial_analysis = pipeline_result.get('financial_analysis', {})
            answer = financial_analysis.get('answer', 'Análisis completado')
            
            # Información adicional del pipeline
            agent_used = financial_analysis.get('agent_used', 'pipeline_completo')
            confidence = financial_analysis.get('confidence', 0.0)
            
            # Resumen de ejecución
            summary = pipeline_result.get('execution_summary', {})
            
            logger.info(f"✅ Agente usado: {agent_used} (confianza: {confidence:.2f})")
            logger.info(f"✅ Pipeline: {summary.get('pipeline_completion', 'N/A')}")
            logger.info(f"✅ Respuesta: {answer[:100]}...")
            
            return jsonify({
                'answer': answer,
                'success': True,
                'agent_used': agent_used,
                'confidence': confidence,
                'pipeline_completion': summary.get('pipeline_completion', 'N/A'),
                'pages_processed': summary.get('pdf_pages_processed', 0),
                'agents_executed': summary.get('agents_executed', 0),
                'files_generated': summary.get('files_generated', 0)
            })
        else:
            error_msg = pipeline_result.get('error', 'Error desconocido en pipeline')
            logger.error(f"❌ Error en pipeline: {error_msg}")
            
            return jsonify({
                'error': error_msg,
                'success': False
            }), 500
        
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"❌ ERROR COMPLETO:\n{error_trace}")
        
        return jsonify({
            'error': str(e),
            'success': False,
            'debug_info': error_trace
        }), 500

@app.route('/system-status', methods=['GET'])
def system_status():
    if SYSTEM_AVAILABLE:
        try:
            system = FinancialExtractionSystem()
            
            # Usar la información del coordinator dentro del sistema
            coordinator_status = system.coordinator.get_system_status()
            
            return jsonify({
                'status': 'active',
                'system_available': True,
                'pipeline_enabled': True,
                'predictor_available': coordinator_status.get('predictor_available', False),
                'agents_loaded': coordinator_status.get('agents_loaded', 0),
                'pdf_status': coordinator_status.get('pdf_status', {}),
                **coordinator_status
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'system_available': False,
                'error': str(e)
            })
    else:
        return jsonify({
            'status': 'unavailable',
            'system_available': False,
            'message': 'FinancialExtractionSystem no disponible'
        })

if __name__ == '__main__':
    print("🚀 Servidor backend iniciado con arquitectura completa de main_system.py")
    print("📍 URL: http://127.0.0.1:8000")
    print("🔍 Pipeline: PDF Extractor → FinancialCoordinator → Question Router → Agentes Especializados → Predictor")
    
    app.run(debug=True, host='127.0.0.1', port=8000)
