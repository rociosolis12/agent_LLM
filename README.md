# 🧠 Sistema Multi-Agente Financiero Avanzado (agent_LLM)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-blue.svg)](https://reactjs.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-blue.svg)](https://postgresql.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.0+-red.svg)](https://neo4j.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Descripción General

**Sistema de IA Multi-Modal Avanzado** que combina análisis financiero automatizado, procesamiento de lenguaje natural y bases de datos multimodales. Desarrollado con arquitectura multi-agente para el procesamiento inteligente de documentos financieros complejos, con capacidades de **Text-to-SQL**, **Text-to-Cypher**, y **Memory Vector Store**.

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

### 🏗️ Flujo Multi-Modal Completo

```mermaid
graph TB
    subgraph "🎯 INPUT LAYER"
        A[👤 User Query]
        B[📄 PDF Documents]
        C[📊 Financial Data]
    end
    
    subgraph "🧠 PROCESSING CORE"
        D[🎭 main_system.py]
        E[📄 PDF Extractor Agent]
        F[🤖 Financial Coordinator]
    end
    
    subgraph "🔀 INTELLIGENT ROUTING"
        G{🔍 Question Router}
        G -->|Financial Analysis| H[💼 Financial Agents]
        G -->|SQL Query| I[🗃️ Text-to-SQL Agent]
        G -->|Graph Query| J[🕸️ Text-to-Cypher Agent]
        G -->|Semantic Search| K[🧠 Vector Store Agent]
    end
    
    subgraph "🏦 FINANCIAL AGENTS"
        H --> L[🏦 Balance Agent]
        H --> M[📈 Income Agent]
        H --> N[💰 Equity Agent]
        H --> O[💸 Cash Flow Agent]
    end
    
    subgraph "🛠️ SPECIALIZED PROCESSORS"
        P[📝 Parser Agent]
        Q[🔍 Analysis Agent]
        R[✅ Validation Agent]
        S[🏗️ Structure Agent]
    end
    
    subgraph "🗄️ DATA LAYER"
        T[(🐘 PostgreSQL)]
        U[(🕸️ Neo4j Graph)]
        V[(🧠 Vector Store)]
        W[📁 File Storage]
    end
    
    subgraph "🔮 AI LAYER"
        X[🔮 Predictor Agent]
        Y[🤖 LLM Orchestrator]
        Z[📊 Analytics Engine]
    end
    
    subgraph "🌐 OUTPUT LAYER"
        AA[📤 FastAPI Backend]
        BB[⚛️ React Frontend]
        CC[📊 Interactive Dashboard]
    end
    
    A --> D
    B --> E
    C --> F
    D --> G
    
    L --> P
    M --> Q
    N --> R
    O --> S
    I --> T
    J --> U
    K --> V
    
    P --> X
    Q --> Y
    R --> Z
    S --> AA
    
    X --> AA
    Y --> BB
    Z --> CC
    
    style G fill:#e1f5fe,stroke:#01579b
    style H fill:#f3e5f5,stroke:#4a148c
    style I fill:#e8f5e8,stroke:#1b5e20
    style J fill:#fff3e0,stroke:#e65100
    style K fill:#fce4ec,stroke:#880e4f
```

### 🎯 Sistema de Routing Inteligente

El **Question Router** analiza cada consulta y determina el mejor agente para procesarla:

| Tipo de Consulta | Agente Responsable | Tecnología | Ejemplo |
|------------------|-------------------|------------|---------|
| **📊 Análisis Financiero** | Financial Agents | Multi-Agent | "¿Cuál es el ROE del banco?" |
| **🗃️ Consultas SQL** | Text-to-SQL Agent | SQL Generation | "Muestra ingresos por trimestre" |
| **🕸️ Relaciones Complejas** | Text-to-Cypher Agent | Neo4j Graph | "¿Qué bancos están conectados?" |
| **🔍 Búsqueda Semántica** | Vector Store Agent | Embeddings | "Documentos similares a este informe" |

## 🔧 Tecnologías y Arquitectura

### 🖥️ Stack Tecnológico Completo

**🧠 AI & Machine Learning**
- **OpenAI GPT-4/3.5**: Agentes LLM principales
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
│   ├── routers/
│   │   ├── financial.py                 # Endpoints financieros
│   │   ├── sql_agent.py                 # Text-to-SQL endpoints
│   │   ├── graph_agent.py               # Text-to-Cypher endpoints
│   │   ├── vector_store.py              # Vector search endpoints
│   │   └── upload.py                    # File upload endpoints
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
│   ├── 🗃️ data_agents/                  # Agentes de Datos
│   │   ├── sql_agent.py                 # Text-to-SQL
│   │   ├── cypher_agent.py              # Text-to-Cypher
│   │   ├── vector_agent.py              # Vector Store Search
│   │   └── embedding_agent.py           # Generación Embeddings
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
# API Keys
OPENAI_API_KEY=sk-your-openai-key
HUGGINGFACE_API_KEY=hf_your-huggingface-key

# Database URLs
DATABASE_URL=postgresql://user:password@localhost:5432/financial_agent_db
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password123
REDIS_URL=redis://localhost:6379

# Vector Store
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENVIRONMENT=us-west1-gcp

# Application Settings
API_HOST=127.0.0.1
API_PORT=8000
FRONTEND_URL=http://localhost:3000
DEBUG=True
LOG_LEVEL=INFO

# Agent Configuration
MAX_AGENTS=10
AGENT_TIMEOUT=300
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=gpt-4-turbo-preview
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

#### **Caso Real: GarantiBank**

```python
# Cargar y analizar memoria anual
from agents.financial_coordinator import FinancialCoordinator

coordinator = FinancialCoordinator()

# Procesamiento automático
result = coordinator.analyze_annual_report(
    pdf_path="data/uploads/GarantiBank_Annual_Report.pdf",
    analysis_type="comprehensive"
)

print(f"ROE: {result['kpis']['roe']}%")
print(f"Tier 1 Ratio: {result['kpis']['tier1_ratio']}%")
print(f"Risk Assessment: {result['risk_analysis']['overall_score']}")
```

#### **Consultas en Lenguaje Natural**

```
Usuario: "¿Cuál fue la evolución de la cartera crediticia de GarantiBank en los últimos 3 años?"

🤖 Agente Respuesta:
┌─────────────────────────────────────────────┐
│ 📊 EVOLUCIÓN CARTERA CREDITICIA GARANTI     │
├─────────────────────────────────────────────┤
│ 2021: 187.2B TL (+12.5%)                   │
│ 2022: 218.9B TL (+16.9%)                   │
│ 2023: 245.8B TL (+12.3%)                   │
│                                             │
│ 📈 Crecimiento promedio: 14.2% anual       │
│ 🎯 Morosidad actual: 2.8%                  │
│ ⚠️  Alerta: Crecimiento acelerado          │
└─────────────────────────────────────────────┘
```

### 🗃️ 2. Text-to-SQL Inteligente

#### **Generación Automática de Consultas**

```python
from agents.data_agents.sql_agent import SQLAgent

sql_agent = SQLAgent()

# Consulta en lenguaje natural
query = "Muestra los ingresos trimestrales del último año por línea de negocio"

sql_result = sql_agent.generate_sql(
    natural_query=query,
    database_schema="financial_data"
)

print("SQL Generado:")
print(sql_result['sql_query'])
print("\nResultados:")
print(sql_result['data'])
```

**SQL Generado:**
```sql
SELECT 
    business_line,
    EXTRACT(QUARTER FROM date) as quarter,
    EXTRACT(YEAR FROM date) as year,
    SUM(revenue) as quarterly_revenue
FROM financial_statements 
WHERE date >= DATE_SUB(CURRENT_DATE, INTERVAL 1 YEAR)
GROUP BY business_line, quarter, year
ORDER BY year, quarter, business_line;
```

#### **Ejemplos de Consultas Soportadas**

| Consulta Natural | SQL Generado | Complejidad |
|------------------|--------------|-------------|
| "Bancos con ROE > 15%" | `SELECT * FROM banks WHERE roe > 15` | 🟢 Simple |
| "Top 5 bancos por rentabilidad último trimestre" | `SELECT name, roe FROM banks WHERE quarter = 'Q4' ORDER BY roe DESC LIMIT 5` | 🟡 Medio |
| "Análisis comparativo de solvencia vs liquidez por región" | `SELECT region, AVG(solvency_ratio), AVG(liquidity_ratio) FROM bank_metrics GROUP BY region` | 🔴 Complejo |

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

### 🗃️ Text-to-SQL Endpoints

#### Generar SQL desde Lenguaje Natural
```http
POST /api/sql/generate
Content-Type: application/json

{
  "query": "Muestra los 10 bancos más rentables",
  "database": "financial_db",
  "format": "json|csv|table"
}

Response: {
  "sql_query": "SELECT name, roe FROM banks ORDER BY roe DESC LIMIT 10",
  "data": [...],
  "execution_time": 0.045,
  "row_count": 10
}
```

#### Ejecutar SQL Personalizada
```http
POST /api/sql/execute
Content-Type: application/json

{
  "sql_query": "SELECT * FROM banks WHERE country = 'Turkey'",
  "parameters": {"country": "Turkey"}
}
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

#### Añadir Documentos
```http
POST /api/vector/add-documents
Content-Type: application/json

{
  "documents": [
    {
      "content": "...",
      "metadata": {"type": "report", "bank": "garanti"}
    }
  ],
  "collection": "financial_documents"
}
```

## 🧪 Testing y Validación

### 🔬 Suite de Tests Completa

```bash
# Tests unitarios
pytest tests/unit/ -v

# Tests de integración
pytest tests/integration/ -v

# Tests de API
pytest tests/api/ -v

# Tests end-to-end
pytest tests/e2e/ -v

# Tests con coverage
pytest --cov=. --cov-report=html tests/
```

### 📊 Performance Testing

```bash
# Load testing para API
locust -f tests/load/locustfile.py --host=http://localhost:8000

# Database performance
python tests/performance/db_benchmark.py

# Vector search performance
python tests/performance/vector_benchmark.py
```

### ✅ Validación de Agentes

```python
# Test individual de agentes
from tests.agent_tests import AgentValidator

validator = AgentValidator()

# Test Financial Coordinator
validator.test_financial_coordinator(
    test_case="garanti_annual_report",
    expected_kpis={"roe": 15.2, "tier1": 16.8}
)

# Test SQL Agent
validator.test_sql_agent(
    natural_query="Top 5 banks by ROE",
    expected_sql_pattern="SELECT.*ORDER BY roe DESC LIMIT 5"
)

# Test Cypher Agent
validator.test_cypher_agent(
    natural_query="Banks connected to GarantiBank",
    expected_cypher_pattern="MATCH.*GarantiBank.*connected"
)
```

## 🔧 Configuración Avanzada

### ⚙️ Variables de Entorno Completas

```bash
# AI Models Configuration
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-4-turbo-preview
EMBEDDING_MODEL=text-embedding-ada-002
HUGGINGFACE_API_KEY=hf_your-key

# Database Configuration
DATABASE_URL=postgresql://user:pass@localhost:5432/financial_agent
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password123
REDIS_URL=redis://localhost:6379/0

# Vector Store Configuration
VECTOR_STORE_TYPE=pinecone|chroma|weaviate
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENVIRONMENT=us-west1-gcp
PINECONE_INDEX_NAME=financial-embeddings

# Application Settings
API_HOST=127.0.0.1
API_PORT=8000
FRONTEND_URL=http://localhost:3000
DEBUG=True
LOG_LEVEL=INFO

# Agent Configuration
MAX_CONCURRENT_AGENTS=10
AGENT_TIMEOUT=300
RETRY_ATTEMPTS=3
BATCH_SIZE=100

# Security
SECRET_KEY=your-secret-key
JWT_EXPIRATION=3600
CORS_ORIGINS=["http://localhost:3000"]

# File Processing
MAX_FILE_SIZE=50MB
SUPPORTED_FORMATS=["pdf", "docx", "xlsx", "csv"]
TEMP_DIR=/tmp/agent_uploads

# Monitoring
ENABLE_METRICS=true
PROMETHEUS_PORT=9090
GRAFANA_URL=http://localhost:3001
```

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

# LLM Configuration
LLM_CONFIG = {
    "openai": {
        "model": "gpt-4-turbo-preview",
        "temperature": 0.1,
        "max_tokens": 4000,
        "top_p": 0.9
    },
    "backup_model": {
        "model": "gpt-3.5-turbo-16k",
        "temperature": 0.2,
        "max_tokens": 8000
    }
}
```

## 🚨 Troubleshooting Avanzado

### 🔍 Diagnóstico del Sistema

```bash
# Health check completo
curl http://localhost:8000/health/detailed

# Status de todas las bases de datos
python scripts/check_databases.py

# Verificar conectividad de agentes
python scripts/agent_connectivity_test.py

# Monitor de performance en tiempo real
python scripts/system_monitor.py
```

### 🐛 Problemas Comunes y Soluciones

#### **Error: "Database connection failed"**
```bash
# Verificar PostgreSQL
pg_isready -h localhost -p 5432

# Verificar Neo4j
cypher-shell -a bolt://localhost:7687 -u neo4j -p password123

# Reinicializar conexiones
python scripts/reset_connections.py
```

#### **Error: "Vector store not accessible"**
```bash
# Verificar Pinecone
python -c "import pinecone; pinecone.init(api_key='your-key'); print('OK')"

# Verificar índices
python scripts/check_vector_indices.py

# Recrear colecciones
python scripts/recreate_vector_collections.py
```

#### **Error: "Agent timeout"**
```bash
# Verificar carga del sistema
htop

# Aumentar timeouts
export AGENT_TIMEOUT=600

# Monitorear agentes
python scripts/agent_monitor.py
```

#### **Error: "LLM API rate limit"**
```bash
# Verificar cuotas de API
python scripts/check_api_quotas.py

# Configurar rate limiting
export OPENAI_REQUESTS_PER_MINUTE=60

# Usar modelo de backup
export USE_BACKUP_MODEL=true
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

## 🚀 Despliegue en Producción

### 🐳 Docker Compose para Producción

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  api:
    build: 
      context: .
      dockerfile: docker/Dockerfile.api
    environment:
      - NODE_ENV=production
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
          cpus: '1'
    
  frontend:
    build:
      context: ./frontend
      dockerfile: ../docker/Dockerfile.frontend
    environment:
      - NODE_ENV=production
    
  postgresql:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=financial_agent
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
  
  neo4j:
    image: neo4j:5.0-enterprise
    environment:
      - NEO4J_AUTH=neo4j/${NEO4J_PASSWORD}
      - NEO4J_ACCEPT_LICENSE_AGREEMENT=yes
    volumes:
      - neo4j_data:/data
    
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl/certs

volumes:
  postgres_data:
  neo4j_data:
  redis_data:
```

### ☁️ Despliegue en AWS

```bash
# Usar Terraform para infraestructura
cd infrastructure/terraform
terraform init
terraform plan
terraform apply

# Desplegar con ECS
aws ecs create-service --cli-input-json file://ecs-service.json

# Configurar RDS y ElastiCache
aws rds create-db-instance --db-instance-identifier financial-agent-db
aws elasticache create-cache-cluster --cache-cluster-id financial-agent-redis
```

### 📈 Escalado y Performance

```python
# Configuración de escalado automático
AUTO_SCALING_CONFIG = {
    "api_instances": {
        "min": 2,
        "max": 10,
        "target_cpu": 70,
        "scale_up_cooldown": 300,
        "scale_down_cooldown": 600
    },
    "agent_workers": {
        "min": 5,
        "max": 50,
        "queue_threshold": 100,
        "memory_threshold": 80
    }
}

# Load balancing configuration
LOAD_BALANCER_CONFIG = {
    "algorithm": "round_robin",
    "health_check": "/health",
    "timeout": 30,
    "max_retries": 3
}
```

## 🤝 Contribución y Desarrollo

### 🔄 Workflow de Contribución

1. **Fork** el repositorio
2. **Crea** una rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. **Desarrolla** siguiendo los estándares de código
4. **Ejecuta** todos los tests (`pytest tests/`)
5. **Commit** con mensaje descriptivo (`git commit -am 'Add: nueva funcionalidad'`)
6. **Push** a tu fork (`git push origin feature/nueva-funcionalidad`)
7. **Abre** un Pull Request con descripción detallada

### 📋 Estándares de Código

```python
# Pre-commit hooks
repos:
  - repo: https://github.com/psf/black
    hooks:
      - id: black
        language_version: python3.10
  
  - repo: https://github.com/pycqa/flake8
    hooks:
      - id: flake8
        args: [--max-line-length=88]
  
  - repo: https://github.com/pycqa/isort
    hooks:
      - id: isort
        args: [--profile=black]
```

### 🧪 Guía de Testing

```python
# Estructura de tests
tests/
├── unit/
│   ├── test_agents/
│   ├── test_api/
│   └── test_services/
├── integration/
│   ├── test_database/
│   ├── test_llm_integration/
│   └── test_agent_coordination/
└── e2e/
    ├── test_complete_workflows/
    └── test_user_scenarios/

# Ejemplo de test de agente
def test_financial_coordinator():
    coordinator = FinancialCoordinator()
    result = coordinator.analyze_document("test_data/sample_report.pdf")
    
    assert result['status'] == 'success'
    assert 'roe' in result['kpis']
    assert result['kpis']['roe'] > 0
```

### 🚀 Roadmap de Desarrollo

#### **Q1 2025: Funcionalidades Core**
- [x] ✅ Sistema multi-agente financiero
- [x] ✅ Text-to-SQL básico
- [x] ✅ Text-to-Cypher básico
- [x] ✅ Vector store con embeddings
- [ ] 🔄 Optimización de performance
- [ ] 📱 API móvil

#### **Q2 2025: IA Avanzada**
- [ ] 🤖 Fine-tuning de modelos específicos
- [ ] 🧠 Agentes con memoria a largo plazo
- [ ] 📊 Predicciones avanzadas con ML
- [ ] 🔍 Análisis de sentimientos en documentos

#### **Q3 2025: Integración y Escalado**
- [ ] 🔗 Integración con Bloomberg API
- [ ] 📈 Dashboard en tiempo real
- [ ] ☁️ Despliegue multi-cloud
- [ ] 🌍 Soporte multi-idioma

#### **Q4 2025: Funcionalidades Empresariales**
- [ ] 👥 Sistema de usuarios y roles
- [ ] 🔒 Auditoría y compliance
- [ ] 📊 Reportes regulatorios automáticos
- [ ] 🤝 Integraciones con sistemas bancarios

## 📄 Licencia

Este proyecto está licenciado bajo la **Licencia MIT** - ver el archivo [LICENSE](LICENSE) para más detalles.

```
MIT License

Copyright (c) 2024 Rocío Solís

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

## 🙏 Agradecimientos

### 🏆 Tecnologías y Frameworks

- **[OpenAI](https://openai.com/)** - Por las APIs de GPT que potencian nuestros agentes
- **[FastAPI](https://fastapi.tiangolo.com/)** - Por el excelente framework web asíncrono
- **[React](https://reactjs.org/)** - Por la biblioteca de UI moderna y eficiente
- **[PostgreSQL](https://postgresql.org/)** - Por la robusta base de datos relacional
- **[Neo4j](https://neo4j.com/)** - Por la potente base de datos de grafos
- **[LangChain](https://langchain.com/)** - Por las herramientas de orquestación de LLMs

### 👥 Comunidad Open Source

- **[Sentence Transformers](https://www.sbert.net/)** - Por los modelos de embeddings
- **[Hugging Face](https://huggingface.co/)** - Por el ecosistema de ML/NLP
- **[Material-UI](https://mui.com/)** - Por los componentes de diseño
- **[Docker](https://docker.com/)** - Por la plataforma de containerización

### 🎓 Instituciones Académicas

- **Universidad Complutense de Madrid** - Por el apoyo en la investigación
- **Comunidad FinTech** - Por la retroalimentación y casos de uso reales

### 💡 Inspiración

Este proyecto se inspiró en la necesidad de democratizar el análisis financiero avanzado y hacer que las herramientas de IA sean accesibles para analistas financieros, investigadores y desarrolladores.

## 📞 Soporte y Contacto

### 🆘 Obtener Ayuda

- **📚 Documentación**: [Wiki del Proyecto](https://github.com/rociosolis12/agent_LLM/wiki)
- **🐛 Issues**: [GitHub Issues](https://github.com/rociosolis12/agent_LLM/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/rociosolis12/agent_LLM/discussions)
- **📺 Tutoriales**: [Canal de YouTube](https://youtube.com/@agent-llm-tutorials)

### 📧 Contacto Directo

- **Email Principal**: rocio.solis@financialagent.com
- **LinkedIn**: [Rocío Solís](https://linkedin.com/in/rociosolis12)
- **Twitter**: [@rociosolis_dev](https://twitter.com/rociosolis_dev)

### 🏢 Uso Empresarial

Para consultas sobre licencias empresariales, integración personalizada o soporte premium:
- **Email Empresarial**: enterprise@financialagent.com
- **Teléfono**: +34 xxx xxx xxx
- **Calendly**: [Agendar Reunión](https://calendly.com/rociosolis12)

---

<div align="center">

### 🌟 **¿Te gusta el proyecto? ¡Dale una ⭐ en GitHub!**

[🐛 Reportar Bug](https://github.com/rociosolis12/agent_LLM/issues/new?template=bug_report.md) · [✨ Solicitar Feature](https://github.com/rociosolis12/agent_LLM/issues/new?template=feature_request.md) · [📚 Documentación](https://github.com/rociosolis12/agent_LLM/wiki) · [💬 Discussions](https://github.com/rociosolis12/agent_LLM/discussions)

---

**🚀 Sistema Multi-Agente Financiero** | **🤖 IA para Finanzas** | **🔬 Open Source**

*Desarrollado con ❤️ por [Rocío Solís](https://github.com/rociosolis12)*

</div>
