Sistema Multi-Agente Financiero (agent_LLM)

Pipeline completo de Ingest & Retrieve para documentos financieros con:

Extracción y pre-procesado de PDF

Coordinación por un Financial Coordinator

Agentes especializados (parsing, análisis contable, validaciones, etc.)

Módulo Predictor para KPIs

API (FastAPI) y Frontend (React + MUI)

Diagrama de alto nivel: tfm_solution_diagram.svg

📂 Estructura del repositorio
agent_LLM/
├─ api/                      # Backend (FastAPI)
│  ├─ main_api.py            # Punto de entrada FastAPI (app)
│  └─ ...                    # Routers, servicios, utils
├─ agents/                   # Lógica de agentes
├─ frontend/                 # Frontend (React)
├─ exports/                  # Salidas / resultados
├─ sessions/                 # Historias de ejecución / logs ligeros
├─ main_system.py            # Orquestación de alto nivel
├─ financial_coordinator.py  # Coordinador de agentes
├─ extractor_pdf_agent.py    # Extracción/parseo de PDFs
├─ predictor_agent.py        # Predicción de KPIs (ML/Stats)
├─ question_router.py        # Router de preguntas → agente
├─ web_server.py             # (opcional) utilidades web
├─ config.py                 # Configuración central
├─ requirements.txt          # Dependencias Python
├─ .env                      # Variables de entorno (NO subir)
└─ README.md

✅ Requisitos

Python 3.10+ (recomendado 3.11)

Node.js 18+ y npm 9+ (para el frontend)

Git

Windows: se recomienda PowerShell o Git Bash.

🔐 Variables de entorno

Crea un archivo .env en la raíz con las claves que uses (modelos, APIs, etc.). Ejemplo mínimo:

# Backend
API_HOST=127.0.0.1
API_PORT=8000

# Modelos / claves (ejemplos)
OPENAI_API_KEY=...
GROQ_API_KEY=...


No subas .env a GitHub. Está ignorado en .gitignore.

🚀 Puesta en marcha
1) Backend (FastAPI)

Ve a la raíz del proyecto y crea un entorno virtual:

# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate


Instala dependencias:

pip install -r requirements.txt


Arranca la API:

python -m uvicorn api.main_api:app --host 127.0.0.1 --port 8000 --reload


Comprueba la documentación interactiva:

Swagger UI: http://127.0.0.1:8000/docs

ReDoc: http://127.0.0.1:8000/redoc

Si cambias el puerto/host, recuerda alinear el frontend (ver variable API_BASE_URL).

2) Frontend (React + MUI)

Entra en la carpeta del frontend:

cd frontend


Instala dependencias:

npm install


Configura el endpoint del backend (según tu setup).
Si el proyecto usa Vite, crea frontend/.env o frontend/.env.local:

VITE_API_BASE_URL=http://127.0.0.1:8000


Si es Create React App:

REACT_APP_API_BASE_URL=http://127.0.0.1:8000


Arranca el desarrollo:

# Si es Vite
npm run dev

# Si es Create React App
npm start


Abre el navegador:

Vite (por defecto): http://127.0.0.1:5173

CRA (por defecto): http://127.0.0.1:3000

Si no estás seguro de si es Vite o CRA, mira el package.json (campo scripts):

Vite suele tener "dev": "vite".

CRA suele tener "start": "react-scripts start".

🔗 Flujo de ejecución (resumen)
Cliente (Frontend) ──► FastAPI (Backend)
         │                 │
         ▼                 ▼
  main_system.py   financial_coordinator.py
         │                 │
         ├─► extractor_pdf_agent.py
         ├─► agentes especializados (paralelo)
         └─► predictor_agent.py
         ▼
      Respuesta JSON → Frontend (UI)

🧪 Probar rápido (end-to-end)

Levanta backend (uvicorn ...) y después frontend (npm run dev).

Desde el frontend, lanza una consulta (sube un PDF o elige un caso de ejemplo, según tu UI).

Verifica resultados en:

UI del frontend

Respuestas JSON de la API (/docs)

Carpeta exports/ (si tu pipeline guarda salidas)

🛠️ Comandos útiles
# Backend (desde la raíz, con venv activo)
pip install -r requirements.txt
python -m uvicorn api.main_api:app --host 127.0.0.1 --port 8000 --reload

# Frontend
cd frontend
npm install
npm run dev     # Vite
# o
npm start       # CRA

❗ Problemas frecuentes

CORS: si el navegador bloquea peticiones, añade CORSMiddleware en FastAPI permitiendo el origen del frontend.

Puerto en uso: cambia --port en uvicorn o el puerto del frontend (--port 5174 en Vite).

.env no cargado: asegúrate de tenerlo en la raíz y que config.py lo lea (por ejemplo, con python-dotenv).

node_modules gigantes: no se suben al repo; si lo hiciste, bórralos del control de versiones con:

git rm -r --cached frontend/node_modules


Avisos LF/CRLF (Windows): opcionalmente:

git config --global core.autocrlf true

🧹 .gitignore recomendado
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.venv/
env/
venv/
.ipynb_checkpoints/

# Node / Frontend
frontend/node_modules/
node_modules/
dist/
build/
*.log
.cache/

# Entornos / claves
.env
*.env
*.env.*

# SO / editores
.DS_Store
Thumbs.db
.vscode/
.idea/

# Salidas del proyecto
exports/
sessions/

📄 Licencia

Este repositorio se distribuye con fines académicos. Añade aquí tu licencia (MIT, Apache-2.0, etc.) si procede.
