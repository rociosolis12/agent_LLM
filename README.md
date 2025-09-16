# 🧠 Sistema Multi-Agente Financiero Avanzado (agent_LLM)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-blue.svg)](https://reactjs.org/)
[![Azure OpenAI](https://img.shields.io/badge/Azure%20OpenAI-GPT--4o-orange.svg)](https://azure.microsoft.com/en-us/services/cognitive-services/openai-service/)
[![Groq](https://img.shields.io/badge/Groq-Llama--3.3--70b-purple.svg)](https://groq.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-blue.svg)](https://postgresql.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.0+-red.svg)](https://neo4j.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Descripción General

**Sistema de IA Multi-Modal Avanzado** que combina análisis financiero automatizado, procesamiento de lenguaje natural y bases de datos multimodales. Desarrollado con arquitectura multi-agente para el procesamiento inteligente de documentos financieros complejos, con capacidades de **Text-to-SQL**, **Text-to-Cypher**, y **Memory Vector Store**.

Utiliza **Azure OpenAI GPT-4o** para análisis financiero complejo y **Groq Llama-3.3-70b** para procesamiento de alta velocidad, garantizando tanto precisión como rendimiento empresarial.

### 🎯 Capacidades Principales

🔍 **Análisis Financiero Multi-Agente**
- Procesamiento automático de memorias anuales bancarias
- Extracción de 4 estados financieros principales
- Predicción de KPIs y análisis de tendencias

🗃️ **Text-to-SQL Inteligente**
- Conversión de lenguaje natural a consultas SQL
- Análisis de datos financieros estructurados
- Generación automática de reportes

🕸️ **Graph Database (Text-to-Cypher)**
- Modelado de relaciones financieras complejas
- Consultas en Neo4j mediante lenguaje natural
- Análisis de redes de riesgo y exposiciones

🧠 **Memory Vector Store con Embeddings**
- Almacenamiento semántico de documentos
- Búsqueda por similitud avanzada
- Memoria persistente para contexto histórico

## ✨ Arquitectura del Sistema

### 🏗️ Flujo del Sistema - Arquitectura Real

```mermaid
graph TB
    subgraph "🎯 INPUT LAYER"
        A[👤 User Query]
        B[📄 PDF Documents]
        C[📊 Financial Data]
    end
    
    subgraph "🔄 PROCESSING CORE"
        D[🎭 main_system.py]
        E[📄 PDF Extractor Agent]
        F[💾 Extracted PDF Content]
    end
    
    subgraph "🧠 COORDINATION LAYER"
        G[🤖 Financial Coordinator]
        H{🔍 Question Analysis}
    end
    
    subgraph "🔀 INTELLIGENT ROUTING"
        H -->|Financial Analysis| I[💼 Financial Agents Decision]
        H -->|SQL Query| J[🗃️ Text-to-SQL Decision]
        H -->|Graph Query| K[🕸️ Text-to-Cypher Decision]
        H -->|Semantic Search| L[🧠 Vector Store Decision]
    end
    
    subgraph "🏦 FINANCIAL AGENTS"
        I --> M[🏦 Balance Agent]
        I --> N[📈 Income Agent]
        I --> O[💰 Equity Agent]
        I --> P[💸 Cash Flow Agent]
    end
    
    subgraph "🗃️ DATA AGENTS"
        J --> Q[🗃️ Text-to-SQL Agent]
        K --> R[🕸️ Text-to-Cypher Agent]
        L --> S[🧠 Vector Store Agent]
    end
    
    subgraph "🛠️ SPECIALIZED PROCESSORS"
        T[📝 Parser Agent]
        U[🔍 Analysis Agent]
        V[✅ Validation Agent]
        W[🏗️ Structure Agent]
    end
    
    subgraph "🗄️ DATA LAYER"
        X[(🐘 PostgreSQL)]
        Y[(🕸️ Neo4j Graph)]
        Z[(🧠 Vector Store)]
        AA[📁 File Storage]
    end
    
    subgraph "🔮 AI LAYER"
        BB[🔮 Predictor Agent]
        CC[🤖 LLM Orchestrator]
        DD[📊 Analytics Engine]
    end
    
    subgraph "🌐 OUTPUT LAYER"
        EE[📤 FastAPI Backend]
        FF[⚛️ React Frontend]
        GG[📊 Interactive Dashboard]
    end
    
    %% Flujo principal
    A --> D
    B --> E
    D --> E
    E --> F
    F --> G
    A --> G
    G --> H
    
    %% Los agentes usan el PDF extraído como input
    F -.->|PDF Content Input| M
    F -.->|PDF Content Input| N
    F -.->|PDF Content Input| O
    F -.->|PDF Content Input| P
    F -.->|PDF Content Input| Q
    F -.->|PDF Content Input| R
    F -.->|PDF Content Input| S
    
    %% Procesamiento especializado
    M --> T
    N --> U
    O --> V
    P --> W
    Q --> X
    R --> Y
    S --> Z
    
    %% AI Layer
    T --> BB
    U --> CC
    V --> DD
    W --> BB
    
    %% Output
    BB --> EE
    CC --> FF
    DD --> GG
    
    style F fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px
    style G fill:#e1f5fe,stroke:#01579b,stroke-width:3px
    style H fill:#f3e5f5,stroke:#4a148c,stroke-width:3px
```

### 🔄 Secuencia de Ejecución Correcta

#### **Paso 1: Extracción de PDF (Proceso Inicial)**
```python
# 📄 main_system.py inicia el proceso
pdf_content = pdf_extractor_agent.extract_document(pdf_path)
# El PDF se procesa y estructura ANTES de cualquier consulta
```

#### **Paso 2: Financial Coordinator Recibe la Pregunta**
```python
# 🤖 Financial Coordinator analiza la consulta del usuario
user_question = "¿Cuál fue el ROE de GarantiBank?"
coordinator = FinancialCoordinator(pdf_content=pdf_content)
```

#### **Paso 3: Decisión de Agente Basada en la Pregunta**
```python
# 🔍 Question Analysis determina el agente apropiado
selected_agent = coordinator.analyze_question(user_question)
# Resultado: "income_agent" (porque ROE está en cuenta de resultados)
```

#### **Paso 4: Agente Ejecuta con PDF Pre-extraído**
```python
# 📈 El agente seleccionado usa el PDF ya procesado
income_agent = IncomeAgent(pdf_content=pdf_content)
result = income_agent.calculate_roe()
```

### 🎯 Sistema de Routing Inteligente

El **Question Router** analiza cada consulta y determina el mejor agente para procesarla:

## 🔧 Tecnologías y Arquitectura

### 🖥️ Stack Tecnológico Completo

**🧠 AI & Machine Learning**
- **Azure OpenAI GPT-4o**: Agente LLM principal (enterprise-grade)
- **Groq Llama-3.3-70b**: Agente LLM de alta velocidad  
- **OpenAI**: Backup LLM provider
- **Sentence Transformers**: Generación de embeddings
- **LangChain**: Orquestación de agentes
- **Pinecone/ChromaDB**: Vector database
- **HuggingFace Transformers**: Modelos de NLP

**💾 Bases de Datos Multi-Modal**
- **PostgreSQL**: Datos financieros estructurados
- **Neo4j**: Relaciones y grafos financieros
- **Vector Store**: Búsqueda semántica
- **Redis**: Cache y sesiones

**🚀 Backend & API**
- **FastAPI**: Framework web asíncrono
- **SQLAlchemy**: ORM para PostgreSQL
- **Neo4j Driver**: Conexión a graph database
- **Celery**: Procesamiento asíncrono
- **Docker**: Containerización

**🎨 Frontend & Visualización**
- **React 18**: Interface de usuario
- **Material-UI**: Componentes de diseño
- **D3.js**: Visualizaciones de grafos
- **Recharts**: Gráficos financieros
- **Cytoscape.js**: Visualización de redes

### 📁 Estructura del Proyecto Completa

```
agent_LLM/
├── 🚀 api/                              # Backend FastAPI
│   ├── main_api.py                      # Entry point principal
│   ├── services/
│   │   ├── sql_generator.py             # SQL generation logic
│   │   ├── cypher_generator.py          # Cypher query logic
│   │   ├── embedding_service.py         # Vector embeddings
│   │   └── financial_service.py         # Financial analysis
│   ├── models/
│   │   ├── database.py                  # PostgreSQL models
│   │   ├── graph_models.py              # Neo4j models
│   │   └── vector_models.py             # Vector store models
│   └── utils/
│       ├── db_connection.py             # Database connections
│       ├── graph_connection.py          # Neo4j connection
│       └── vector_connection.py         # Vector DB connection
│
├── 🤖 agents/                           # Sistema Multi-Agente
│   ├── 🧠 core/
│   │   ├── base_agent.py                # Agente base
│   │   ├── financial_coordinator.py     # Coordinador principal
│   │   ├── question_router.py           # Router inteligente
│   │   └── agent_factory.py             # Factory de agentes
│   │
│   ├── 💼 financial_agents/             # Agentes Financieros
│   │   ├── balance_agent.py             # Balance General
│   │   ├── income_agent.py              # Cuenta de Resultados
│   │   ├── equity_agent.py              # Patrimonio Neto
│   │   ├── cashflow_agent.py            # Flujos de Efectivo
│   │   └── predictor_agent.py           # Predicciones KPIs
│   │
│   ├── 🔧 processing_agents/            # Agentes de Procesamiento
│   │   ├── extractor_pdf_agent.py       # Extracción PDF
│   │   ├── parser_agent.py              # Parsing de texto
│   │   ├── analysis_agent.py            # Análisis avanzado
│   │   ├── validation_agent.py          # Validaciones
│   │   └── structure_agent.py           # Estructuración datos
│   │
│   └── 🔗 integration/                  # Integraciones
│       ├── database_agent.py            # Integración PostgreSQL
│       ├── graph_agent.py               # Integración Neo4j
│       └── vector_agent.py              # Integración Vector DB
│
├── 🎨 frontend/                         # React Frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── 📊 FinancialDashboard/   # Dashboard financiero
│   │   │   ├── 🗃️ SQLQueryBuilder/      # Constructor SQL
│   │   │   ├── 🕸️ GraphVisualizer/      # Visualizador grafos
│   │   │   ├── 🔍 VectorSearch/         # Búsqueda semántica
│   │   │   ├── 🤖 AgentMonitor/         # Monitor de agentes
│   │   │   └── 📁 FileUploader/         # Cargador archivos
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx            # Página principal
│   │   │   ├── SQLInterface.jsx         # Interface SQL
│   │   │   ├── GraphInterface.jsx       # Interface grafos
│   │   │   └── VectorInterface.jsx      # Interface vectores
│   │   ├── services/
│   │   │   ├── api.js                   # Cliente API principal
│   │   │   ├── sqlService.js            # Servicio SQL
│   │   │   ├── graphService.js          # Servicio grafos
│   │   │   └── vectorService.js         # Servicio vectores
│   │   └── utils/
│   │       ├── formatters.js            # Formateadores datos
│   │       ├── validators.js            # Validadores
│   │       └── constants.js             # Constantes
│   ├── package.json
│   └── public/
│
├── 🗄️ database/                        # Configuración BD
│   ├── postgresql/
│   │   ├── init.sql                     # Inicialización PostgreSQL
│   │   ├── schemas/                     # Esquemas de tablas
│   │   └── migrations/                  # Migraciones
│   ├── neo4j/
│   │   ├── init.cypher                  # Inicialización Neo4j
│   │   ├── constraints.cypher           # Restricciones
│   │   └── indexes.cypher               # Índices
│   └── vector_store/
│       ├── init_collections.py          # Inicialización colecciones
│       └── embedding_config.py          # Configuración embeddings
│
├── 📁 data/                             # Datos del Sistema
│   ├── uploads/                         # Archivos cargados
│   │   ├── financial_reports/           # Memorias anuales
│   │   ├── sql_examples/                # Ejemplos SQL
│   │   └── graph_data/                  # Datos para grafos
│   ├── processed/                       # Datos procesados
│   │   ├── extracted_tables/            # Tablas extraídas
│   │   ├── generated_sql/               # SQL generado
│   │   ├── cypher_queries/              # Consultas Cypher
│   │   └── embeddings/                  # Embeddings generados
│   ├── examples/                        # Casos de ejemplo
│   │   ├── garanti_bank/                # Caso GarantiBank
│   │   ├── sql_examples/                # Ejemplos SQL
│   │   ├── cypher_examples/             # Ejemplos Cypher
│   │   └── vector_examples/             # Ejemplos vectores
│   └── exports/                         # Resultados exportados
│
├── 📊 examples/                         # Ejemplos de Uso
│   ├── 📋 README_examples.md            # Ejemplos documentados
│   ├── financial_analysis/              # Análisis financiero
│   ├── sql_queries/                     # Consultas SQL
│   ├── graph_analysis/                  # Análisis de grafos
│   └── vector_search/                   # Búsqueda vectorial
│
├── 🔧 tests/                            # Tests Automatizados
│   ├── unit/                            # Tests unitarios
│   ├── integration/                     # Tests integración
│   ├── api/                             # Tests API
│   └── e2e/                             # Tests end-to-end
│
├── 📋 docs/                             # Documentación
│   ├── api_reference.md                 # Referencia API
│   ├── agents_guide.md                  # Guía de agentes
│   ├── sql_agent_guide.md               # Guía Text-to-SQL
│   ├── graph_agent_guide.md             # Guía Text-to-Cypher
│   ├── vector_store_guide.md            # Guía Vector Store
│   └── deployment.md                    # Guía despliegue
│
├── 🐳 docker/                           # Configuración Docker
│   ├── docker-compose.yml              # Compose principal
│   ├── Dockerfile.api                   # API container
│   ├── Dockerfile.frontend             # Frontend container
│   ├── postgres.dockerfile             # PostgreSQL container
│   └── neo4j.dockerfile                # Neo4j container
│
├── main_system.py                       # 🎯 Orquestador principal
├── config.py                            # ⚙️ Configuración
├── requirements.txt                     # Dependencias Python
├── .env.example                         # Variables entorno
├── docker-compose.yml                   # Docker compose
└── README.md                            # Este archivo
```

## 🚀 Instalación y Configuración

### ✅ Prerrequisitos

- **Python 3.10+** (recomendado 3.11)
- **Node.js 18+** y **npm 9+**
- **Docker & Docker Compose**
- **PostgreSQL 15+**
- **Neo4j 5.0+**
- **Redis 7+**
- **Git**
- **8GB RAM mínimo** (16GB recomendado)

### 🐳 Instalación con Docker (Recomendado)

```bash
# 1. Clonar el repositorio
git clone https://github.com/rociosolis12/agent_LLM.git
cd agent_LLM

# 2. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus configuraciones

# 3. Levantar todos los servicios
docker-compose up -d

# 4. Verificar que todos los servicios estén corriendo
docker-compose ps
```

### 🛠️ Instalación Manual

#### 1. **Configurar Bases de Datos**

```bash
# PostgreSQL
createdb financial_agent_db

# Neo4j (descargar e instalar desde neo4j.com)
# Iniciar Neo4j Desktop o usar Docker
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  neo4j:latest

# Redis
redis-server
```

#### 2. **Backend Setup**

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt

# Configurar base de datos
python -c "from api.database import create_tables; create_tables()"

# Inicializar Neo4j
python database/neo4j/init_graph.py

# Configurar Vector Store
python database/vector_store/init_collections.py
```

#### 3. **Frontend Setup**

```bash
cd frontend
npm install
npm run build
cd ..
```

### 📝 Configuración de Variables de Entorno

```bash
# .env file

# ===== AZURE OPENAI CONFIGURATION =====
AZURE_OPENAI_ENDPOINT="https://llmcoeiberia-openai.openai.azure.com/"
AZURE_OPENAI_API_KEY="your-azure-openai-key"
AZURE_OPENAI_DEPLOYMENT="gpt-4o"
AZURE_OPENAI_API_VERSION="2024-10-21"

# ===== GROQ CONFIGURATION =====
GROQ_API_KEY="your-groq-api-key"
GROQ_MODEL="llama-3.3-70b-versatile"

# ===== ALTERNATIVE LLM PROVIDERS =====
# OpenAI Direct (Backup)
OPENAI_API_KEY="sk-your-openai-key-backup"
OPENAI_MODEL="gpt-4-turbo-preview"

# HuggingFace for Embeddings
HUGGINGFACE_API_KEY="hf_your-huggingface-key"
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"

# ===== DATABASE CONFIGURATION =====
DATABASE_URL="postgresql://user:password@localhost:5432/financial_agent_db"
NEO4J_URI="bolt://localhost:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="password123"
REDIS_URL="redis://localhost:6379/0"

# ===== VECTOR STORE CONFIGURATION =====
VECTOR_STORE_TYPE="pinecone"  # pinecone|chroma|weaviate
PINECONE_API_KEY="your-pinecone-key"
PINECONE_ENVIRONMENT="us-west1-gcp"
PINECONE_INDEX_NAME="financial-embeddings"

# ===== APPLICATION SETTINGS =====
API_HOST="127.0.0.1"
API_PORT=8000
FRONTEND_URL="http://localhost:3000"
DEBUG=True
LOG_LEVEL="INFO"

# ===== AGENT CONFIGURATION =====
PRIMARY_LLM_PROVIDER="azure_openai"  # azure_openai|groq|openai
BACKUP_LLM_PROVIDER="groq"
MAX_CONCURRENT_AGENTS=10
AGENT_TIMEOUT=300
RETRY_ATTEMPTS=3
BATCH_SIZE=100

# ===== SECURITY =====
SECRET_KEY="your-secret-key"
JWT_EXPIRATION=3600
CORS_ORIGINS=["http://localhost:3000"]

# ===== FILE PROCESSING =====
MAX_FILE_SIZE="50MB"
SUPPORTED_FORMATS=["pdf", "docx", "xlsx", "csv"]
TEMP_DIR="/tmp/agent_uploads"

# ===== MONITORING =====
ENABLE_METRICS=true
PROMETHEUS_PORT=9090
GRAFANA_URL="http://localhost:3001"
```

## 🎯 Uso del Sistema

### 🖥️ Iniciar el Sistema Completo

```bash
# Opción 1: Docker Compose (Recomendado)
docker-compose up -d

# Opción 2: Manual
# Terminal 1: Backend
python -m uvicorn api.main_api:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Frontend
cd frontend && npm start

# Terminal 3: Workers (para procesamiento asíncrono)
celery -A api.tasks worker --loglevel=info
```

### 🌐 Acceder a las Interfaces

- **🎨 Frontend Principal**: http://localhost:3000
- **📊 API Documentation**: http://localhost:8000/docs
- **🕸️ Neo4j Browser**: http://localhost:7474
- **📈 Grafana Dashboard**: http://localhost:3001
- **📋 Admin Panel**: http://localhost:8000/admin

## 💼 Casos de Uso y Ejemplos

### 🏦 1. Análisis Financiero Multi-Agente

#### **🤖 Sistema Multi-Provider LLM**

```python
# El sistema selecciona automáticamente el mejor provider para cada tarea
from agents.financial_coordinator import FinancialCoordinator

# Azure OpenAI para análisis financiero complejo
coordinator = FinancialCoordinator(
    primary_provider="azure_openai",    # GPT-4o para razonamiento complejo
    speed_provider="groq",              # Llama-3.3-70b para respuestas rápidas
    backup_provider="openai"            # GPT-4 como fallback
)

# Ejemplos de uso optimizado por provider:
questions = [
    "¿Cuál fue el ROE de GarantiBank?",           # → Azure OpenAI (análisis complejo)
    "Genera SQL: bancos con ROE > 15%",          # → Groq (generación rápida)
    "¿Qué bancos están conectados a GarantiBank?" # → Groq (consulta directa)
]

for question in questions:
    result = coordinator.process_question(question, pdf_content)
    print(f"🤖 Provider usado: {result['provider_used']}")
    print(f"⚡ Tiempo: {result['response_time']}ms")
    print(f"💡 Respuesta: {result['answer']}")
    print("─" * 50)
```

**Resultado Esperado:**
```
🤖 Provider usado: azure_openai (GPT-4o)
⚡ Tiempo: 2,340ms
💡 Respuesta: El ROE de GarantiBank fue del 15.2%...

🤖 Provider usado: groq (Llama-3.3-70b)  
⚡ Tiempo: 450ms
💡 Respuesta: SELECT name, roe FROM banks WHERE roe > 15...

🤖 Provider usado: groq (Llama-3.3-70b)
⚡ Tiempo: 680ms  
💡 Respuesta: MATCH (g:Bank {name: 'GarantiBank'})-[r]-(connected)...
```

#### **Resultado del Procesamiento**

```
📄 EXTRACCIÓN COMPLETADA - GarantiBank Annual Report
═══════════════════════════════════════════════════
✅ Balance General → Páginas 45-47 (extraído)
✅ Cuenta de Resultados → Páginas 48-50 (extraído)  
✅ Patrimonio Neto → Páginas 51-52 (extraído)
✅ Flujos de Efectivo → Páginas 53-55 (extraído)

💬 PROCESANDO CONSULTAS...

Pregunta: ¿Cuál fue el ROE de GarantiBank?
Agente: Income Agent (usando contenido pre-extraído)
Respuesta: El ROE de GarantiBank fue del 15.2%, mostrando una mejora 
          del 2.1% respecto al año anterior...

Pregunta: ¿Hay problemas de liquidez?
Agente: Cash Flow Agent (usando contenido pre-extraído)
Respuesta: El ratio LCR es del 142%, superando ampliamente el mínimo
          regulatorio del 100%. La posición de liquidez es sólida...

Pregunta: ¿Cómo evolucionó el patrimonio?
Agente: Equity Agent (usando contenido pre-extraído)
Respuesta: El patrimonio neto creció un 12.8% hasta 48.2B TL, 
          principalmente por retención de beneficios...
```

```

### 🕸️ 3. Text-to-Cypher para Análisis de Grafos

#### **Modelado de Relaciones Financieras**

```python
from agents.data_agents.cypher_agent import CypherAgent

cypher_agent = CypherAgent()

# Consulta sobre relaciones bancarias
query = "¿Qué bancos tienen exposición al riesgo soberano turco?"

cypher_result = cypher_agent.generate_cypher(
    natural_query=query,
    graph_schema="financial_network"
)
```

**Cypher Generado:**
```cypher
MATCH (b:Bank)-[e:EXPOSED_TO]->(r:SovereignRisk {country: 'Turkey'})
RETURN b.name, e.exposure_amount, e.risk_rating
ORDER BY e.exposure_amount DESC
```

#### **Visualización de Redes Financieras**

```javascript
// Frontend: Visualización con Cytoscape.js
const networkData = {
  nodes: [
    { id: 'garanti', label: 'GarantiBank', type: 'bank' },
    { id: 'turkey_sovereign', label: 'Turkey Sovereign', type: 'risk' },
    { id: 'corporate_loans', label: 'Corporate Loans', type: 'asset' }
  ],
  edges: [
    { source: 'garanti', target: 'turkey_sovereign', relationship: 'EXPOSED_TO' },
    { source: 'garanti', target: 'corporate_loans', relationship: 'HOLDS' }
  ]
}
```

### 🧠 4. Memory Vector Store con Embeddings

#### **Búsqueda Semántica Avanzada**

```python
from agents.data_agents.vector_agent import VectorAgent

vector_agent = VectorAgent()

# Búsqueda por similitud semántica
query = "Riesgos de liquidez en crisis financieras"

similar_docs = vector_agent.semantic_search(
    query=query,
    collection="financial_documents",
    top_k=5,
    threshold=0.8
)

for doc in similar_docs:
    print(f"📄 {doc['title']}")
    print(f"🎯 Similitud: {doc['score']:.2f}")
    print(f"📝 Extracto: {doc['content'][:200]}...")
    print("─" * 50)
```

#### **Generación y Almacenamiento de Embeddings**

```python
from agents.data_agents.embedding_agent import EmbeddingAgent

embedding_agent = EmbeddingAgent()

# Procesar nuevos documentos
documents = [
    "Memoria Anual GarantiBank 2023",
    "Informe de Estabilidad Financiera TCMB",
    "Análisis Sectorial Bancario Turco"
]

embeddings = embedding_agent.generate_embeddings(
    documents=documents,
    model="sentence-transformers/all-MiniLM-L6-v2"
)

# Almacenar en vector store
vector_agent.store_embeddings(
    embeddings=embeddings,
    metadata=[
        {"type": "annual_report", "bank": "garanti", "year": 2023},
        {"type": "central_bank_report", "institution": "tcmb"},
        {"type": "sector_analysis", "sector": "banking", "country": "turkey"}
    ]
)
```

## 📊 API Reference Completa

### 🏦 Financial Analysis Endpoints

#### Análisis de Documentos
```http
POST /api/financial/analyze
Content-Type: multipart/form-data

{
  "file": "annual_report.pdf",
  "analysis_type": "comprehensive|quick|custom",
  "extract_tables": true,
  "generate_predictions": true
}
```

#### Obtener KPIs
```http
GET /api/financial/kpis/{document_id}
Response: {
  "roe": 15.2,
  "roa": 1.8,
  "tier1_ratio": 16.8,
  "liquidity_ratio": 142.0,
  "efficiency_ratio": 45.3
}
```
```

### 🕸️ Graph Database Endpoints

#### Generar Cypher desde Lenguaje Natural
```http
POST /api/graph/generate-cypher
Content-Type: application/json

{
  "query": "¿Qué bancos están conectados a GarantiBank?",
  "graph_type": "financial_network",
  "include_visualization": true
}

Response: {
  "cypher_query": "MATCH (g:Bank {name: 'GarantiBank'})-[r]-(connected:Bank) RETURN g, r, connected",
  "results": [...],
  "visualization_data": {...}
}
```

#### Análisis de Centralidad
```http
GET /api/graph/centrality/{node_type}
Response: {
  "betweenness": {...},
  "closeness": {...},
  "degree": {...},
  "pagerank": {...}
}
```

### 🧠 Vector Store Endpoints

#### Búsqueda Semántica
```http
POST /api/vector/search
Content-Type: application/json

{
  "query": "análisis de riesgo crediticio",
  "collection": "financial_documents",
  "top_k": 10,
  "threshold": 0.7,
  "filters": {
    "document_type": "risk_analysis",
    "year": 2023
  }
}
```

## 🔧 Configuración Avanzada

### ⚙️ Variables de Entorno 

```bash
# ===== AZURE OPENAI CONFIGURATION (PRIMARY) =====
AZURE_OPENAI_ENDPOINT="https://llmcoeiberia-openai.openai.azure.com/"
AZURE_OPENAI_API_KEY="your-azure-openai-key"
AZURE_OPENAI_DEPLOYMENT="gpt-4o"
AZURE_OPENAI_API_VERSION="2024-10-21"

# ===== GROQ CONFIGURATION (HIGH-SPEED) =====
GROQ_API_KEY="your-groq-api-key"
GROQ_MODEL="llama-3.3-70b-versatile"

# ===== BACKUP LLM PROVIDERS =====
OPENAI_API_KEY="sk-your-openai-key"
OPENAI_MODEL="gpt-4-turbo-preview"
ANTHROPIC_API_KEY="your-anthropic-key"

# ===== EMBEDDING MODELS =====
EMBEDDING_MODEL="text-embedding-ada-002"  # Azure OpenAI
EMBEDDING_MODEL_BACKUP="sentence-transformers/all-MiniLM-L6-v2"  # Local
HUGGINGFACE_API_KEY="hf_your-key"

# ===== DATABASE CONFIGURATION =====
DATABASE_URL="postgresql://user:pass@localhost:5432/financial_agent"
NEO4J_URI="bolt://localhost:7687"
NEO4J_USERNAME="neo4j"

### 🏗️ Configuración de Agentes

```python
# config.py - Agent Configuration
AGENT_CONFIG = {
    "financial_coordinator": {
        "max_retries": 3,
        "timeout": 120,
        "parallel_processing": True,
        "memory_limit": "2GB"
    },
    "sql_agent": {
        "max_query_complexity": 5,
        "allowed_operations": ["SELECT", "WITH", "MATCH"],
        "timeout": 30,
        "result_limit": 10000
    },
    "cypher_agent": {
        "max_traversal_depth": 3,
        "node_limit": 1000,
        "timeout": 45,
        "visualization_threshold": 100
    },
    "vector_agent": {
        "similarity_threshold": 0.7,
        "max_results": 100,
        "embedding_dimensions": 768,
        "index_refresh_interval": 3600
    },
    "predictor_agent": {
        "prediction_horizon": 12,  # months
        "confidence_threshold": 0.8,
        "model_retrain_interval": 30  # days
    }
}

# LLM Configuration with Multiple Providers
LLM_CONFIG = {
    "azure_openai": {
        "endpoint": "https://llmcoeiberia-openai.openai.azure.com/",
        "deployment": "gpt-4o",
        "api_version": "2024-10-21",
        "temperature": 0.1,
        "max_tokens": 4000,
        "top_p": 0.9,
        "priority": 1  # Primary provider
    },
    "groq": {
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.2,
        "max_tokens": 8000,
        "top_p": 0.9,
        "priority": 2  # High-speed provider
    },
    "openai": {
        "model": "gpt-4-turbo-preview",
        "temperature": 0.1,
        "max_tokens": 4000,
        "top_p": 0.9,
        "priority": 3  # Backup provider
    }
}

# Provider Selection Strategy
PROVIDER_STRATEGY = {
    "financial_analysis": "azure_openai",  # Complex financial reasoning
    "text_to_sql": "groq",                # Fast SQL generation
    "text_to_cypher": "groq",             # Fast Cypher generation  
    "vector_search": "azure_openai",      # Semantic understanding
    "prediction": "azure_openai",         # Advanced analytics
    "fallback": "openai"                  # If others fail
}
```


### 📊 Logs y Monitoring

```bash
# Logs específicos por componente
tail -f logs/agents/financial_coordinator.log
tail -f logs/api/sql_agent.log
tail -f logs/database/postgresql.log
tail -f logs/database/neo4j.log
tail -f logs/vector_store/pinecone.log

# Métricas en tiempo real
docker exec -it grafana grafana-cli admin reset-admin-password admin123
# Acceder a http://localhost:3001

# Alertas personalizadas
python scripts/setup_alerts.py
```

**🚀 Sistema Multi-Agente Financiero** | **🤖 IA para Finanzas** | **🔬 Open Source**

*Desarrollado con ❤️ por [Rocío Solís](https://github.com/rociosolis12)*

</div>
