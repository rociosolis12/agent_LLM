🧠 Sistema Multi-Agente Financiero (agent_LLM)

Este proyecto implementa un pipeline completo de Ingest & Retrieve orientado al análisis de documentos financieros, en particular la Memoria Anual de los bancos.

El sistema:

Lee el PDF completo.

Extrae y organiza los cuatro estados financieros principales:

Balance

Cuenta de resultados

Estado de cambios en el patrimonio neto

Estado de flujos de efectivo

Permite generar predicciones sobre KPIs financieros relevantes.

✨ Características principales

Extracción de informes financieros
Procesamiento automático de la Memoria Anual, focalizado en los 4 estados financieros.

Coordinación multi-agente
Orquestación mediante un Financial Coordinator, que distribuye tareas de parsing, validación y análisis entre agentes.

Agentes especializados

Parsing de texto

Análisis contable

Validaciones regulatorias

Estructuración de datos

Predicción de KPIs
Estimación de indicadores clave como rentabilidad, solvencia o liquidez.

🔍 Modos de interacción

Preguntas libres → El usuario formula cuestiones abiertas y el sistema responde en base a la información extraída.

Análisis detallado → Ejecución de pipelines de análisis predefinidos sobre los estados financieros.

⚙️ Arquitectura

Backend: FastAPI
 (lógica de negocio + endpoints)

Frontend: React + Material UI (interfaz moderna y responsiva)

📂 Estructura del repositorio
agent_LLM/
├─ api/                      # Backend (FastAPI)
│  ├─ main_api.py            # Punto de entrada FastAPI
│  └─ ...                    # Routers, servicios, utils
│
├─ agents/                   # Lógica de agentes
├─ frontend/                 # Frontend (React)
├─ exports/                  # Resultados generados
├─ sessions/                 # Logs / historiales
│
├─ main_system.py            # Orquestación principal
├─ financial_coordinator.py  # Coordinador de agentes
├─ extractor_pdf_agent.py    # Extracción de PDFs
├─ predictor_agent.py        # Predicción de KPIs
├─ question_router.py        # Routing de preguntas
├─ web_server.py             # Utilidades web (opcional)
│
├─ config.py                 # Configuración central
├─ requirements.txt          # Dependencias Python
├─ .env                      # Variables de entorno (IGNORADO)
└─ README.md                 # Este archivo

✅ Requisitos

Python 3.10+ (recomendado 3.11)

Node.js 18+ y npm 9+ (para frontend)

Git

En Windows: PowerShell o Git Bash

🔐 Variables de entorno

Crea un archivo .env en la raíz con las claves necesarias.

Ejemplo mínimo:

# Backend
API_HOST=127.0.0.1
API_PORT=8000

# Modelos / APIs
OPENAI_API_KEY=...
GROQ_API_KEY=...


⚠️ No subas .env a GitHub (ya está en .gitignore).

🚀 Puesta en marcha
1) Backend (FastAPI)
# Crear entorno virtual
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Arrancar API
python -m uvicorn api.main_api:app --host 127.0.0.1 --port 8000 --reload


Accede a la documentación:

Swagger UI → http://127.0.0.1:8000/docs

ReDoc → http://127.0.0.1:8000/redoc

2) Frontend (React + MUI)
cd frontend
npm install


Configura el endpoint del backend:

Vite → frontend/.env

VITE_API_BASE_URL=http://127.0.0.1:8000


Create React App (CRA) → frontend/.env

REACT_APP_API_BASE_URL=http://127.0.0.1:8000


Inicia el servidor de desarrollo:

# Vite
npm run dev   # http://127.0.0.1:5173
# CRA
npm start     # http://127.0.0.1:3000

🔗 Flujo de ejecución (resumen)
flowchart TD
  A[Frontend (React)] --> B[Backend (FastAPI)]
  B --> C[main_system.py]
  C --> D[financial_coordinator.py]
  D --> E[extractor_pdf_agent.py]
  D --> F[Agentes especializados]
  D --> G[predictor_agent.py]
  G --> H[Respuesta JSON → UI]
