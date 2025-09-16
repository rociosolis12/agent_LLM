"""
FastAPI Backend para el Sistema Multi-Agente Financiero
Expone el sistema existente como API REST
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import asyncio
import uuid
from pathlib import Path
import json
from datetime import datetime

# Importar tu sistema existente
import sys
sys.path.append(str(Path(__file__).parent.parent))

from main_system import FinancialExtractionSystem
from financial_coordinator import FinancialCoordinator
from config import get_pdf_paths, PREDICTOR_AGENT_CONFIG, DATA_OUTPUT_DIR

app = FastAPI(
    title="Sistema Multi-Agente Financiero API",
    description="API para análisis inteligente de estados financieros",
    version="4.0.0"
)

# CORS para desarrollo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sistema principal y coordinador
system = FinancialExtractionSystem()
coordinator = FinancialCoordinator()

# Almacenamiento de sesiones activas
active_sessions: Dict[str, Dict] = {}

# Modelos Pydantic
class QuestionRequest(BaseModel):
    question: str
    include_predictions: bool = True
    session_id: Optional[str] = None

class PipelineRequest(BaseModel):
    include_predictions: bool = True
    session_id: Optional[str] = None

class SessionResponse(BaseModel):
    session_id: str
    status: str
    timestamp: str

# ====== ENDPOINTS PRINCIPALES ======

@app.get("/")
async def root():
    """Endpoint de estado del sistema"""
    system_status = coordinator.get_system_status()
    return {
        "message": "Sistema Multi-Agente Financiero API v4.0",
        "status": "active",
        "system_health": system_status.get("system_health", {}),
        "available_agents": system_status.get("available_agents", []),
        "predictor_available": system_status.get("predictor_available", False),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Subir PDF para análisis"""
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")
        
        # Guardar archivo temporal
        pdf_paths = get_pdf_paths()
        input_path = Path(pdf_paths['input_pdf'])
        input_path.parent.mkdir(parents=True, exist_ok=True)
        
        content = await file.read()
        with open(input_path, 'wb') as f:
            f.write(content)
        
        # Ejecutar extractor PDF
        extraction_result = await system.pdf_extractor.extract_financial_statements()
        
        if extraction_result["success"]:
            return {
                "success": True,
                "message": "PDF procesado exitosamente",
                "pages_extracted": extraction_result.get("total_pages_extracted", 0),
                "output_file": extraction_result.get("output_file"),
                "ready_for_analysis": True
            }
        else:
            raise HTTPException(status_code=500, detail=f"Error procesando PDF: {extraction_result['error']}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-question")
async def ask_question(request: QuestionRequest):
    """Hacer pregunta al sistema"""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        # Procesar pregunta con el coordinador
        if request.include_predictions and coordinator.should_include_predictions(request.question):
            result = await coordinator.process_question_with_predictions(request.question)
        else:
            result = await coordinator.process_question(request.question)
        
        # Actualizar sesión
        active_sessions[session_id] = {
            "last_question": request.question,
            "last_result": result,
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "session_id": session_id,
            "success": result.get("success", False),
            "answer": result.get("answer", ""),
            "agent_used": result.get("agent_used", ""),
            "confidence": result.get("confidence", 0),
            "predictions_included": "prediction_answer" in result,
            "prediction_data": result.get("prediction_data", {}),
            "files_generated": result.get("files_generated", 0),
            "timestamp": result.get("timestamp", "")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run-pipeline")
async def run_complete_pipeline(request: PipelineRequest):
    """Ejecutar pipeline completo"""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        # Ejecutar pipeline completo
        pipeline_result = await system.run_complete_pipeline()
        
        # Actualizar sesión
        active_sessions[session_id] = {
            "pipeline_result": pipeline_result,
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "session_id": session_id,
            "success": pipeline_result.get("success", False),
            "pipeline_steps": pipeline_result.get("pipeline_steps", []),
            "execution_summary": pipeline_result.get("execution_summary", {}),
            "predictions": pipeline_result.get("predictions", {}),
            "total_steps": len(pipeline_result.get("pipeline_steps", [])),
            "timestamp": pipeline_result.get("timestamp", "")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system-status")
async def get_system_status():
    """Obtener estado completo del sistema"""
    status = coordinator.get_system_status()
    pdf_paths = get_pdf_paths()
    
    return {
        "agents_status": {
            "loaded": status.get("agents_loaded", 0),
            "available": status.get("available_agents", []),
            "predictor_available": status.get("predictor_available", False)
        },
        "pdf_status": {
            "input_exists": Path(pdf_paths['input_pdf']).exists() if pdf_paths.get('input_pdf') else False,
            "extracted_exists": Path(pdf_paths['output_pdf']).exists() if pdf_paths.get('output_pdf') else False,
            "input_path": pdf_paths.get('input_pdf'),
            "output_path": pdf_paths.get('output_pdf')
        },
        "session_stats": status.get("session_stats", {}),
        "system_health": status.get("system_health", {}),
        "configuration": {
            "predictor_enabled": PREDICTOR_AGENT_CONFIG.get('enabled', False),
            "prediction_horizon": PREDICTOR_AGENT_CONFIG.get('prediction_horizon', 12),
            "pipeline_mode": True
        }
    }

@app.get("/download-results/{session_id}")
async def download_results(session_id: str):
    """Descargar resultados de una sesión"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    
    # Buscar archivos generados
    output_files = list(DATA_OUTPUT_DIR.glob("*.csv"))
    if output_files:
        # Retornar el archivo más reciente
        latest_file = max(output_files, key=lambda f: f.stat().st_mtime)
        return FileResponse(
            latest_file,
            media_type='application/octet-stream',
            filename=latest_file.name
        )
    else:
        raise HTTPException(status_code=404, detail="No hay archivos de resultados disponibles")

@app.get("/sessions")
async def get_active_sessions():
    """Obtener sesiones activas"""
    return {
        "active_sessions": len(active_sessions),
        "sessions": {
            session_id: {
                "timestamp": data["timestamp"],
                "has_question": "last_question" in data,
                "has_pipeline": "pipeline_result" in data
            }
            for session_id, data in active_sessions.items()
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
