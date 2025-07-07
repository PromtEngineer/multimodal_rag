# LocalGPT Architecture Diagrams

**Document Type**: Technical Architecture Diagrams  
**Version**: 1.0  
**Last Updated**: 2025-07-06

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Data Flow Diagrams](#data-flow-diagrams)
4. [Processing Pipelines](#processing-pipelines)
5. [API Interaction Flows](#api-interaction-flows)
6. [Database Relationships](#database-relationships)
7. [Deployment Architecture](#deployment-architecture)

---

## System Overview

### High-Level System Architecture

```mermaid
graph TB
    subgraph "External Layer"
        User[👤 User]
        Admin[👨‍💼 Administrator]
        API_Consumer[🔌 API Consumer]
    end
    
    subgraph "Presentation Layer"
        WebUI[🌐 Web Interface<br/>React/Next.js<br/>Port 3000]
        MobileApp[📱 Mobile App<br/>Future Enhancement]
    end
    
    subgraph "API Gateway Layer"
        Backend[🚪 Backend Server<br/>Python HTTP<br/>Port 8000]
        Auth[🔐 Authentication<br/>Future Enhancement]
        RateLimit[⏱️ Rate Limiting<br/>Future Enhancement]
        LoadBalancer[⚖️ Load Balancer<br/>Production]
    end
    
    subgraph "Business Logic Layer"
        RAG_API[🧠 RAG API Server<br/>Python<br/>Port 8001]
        Agent[🤖 RAG Agent<br/>Core Intelligence]
        Triage[🎯 Query Triage<br/>Smart Routing]
        Verifier[✅ Response Verifier<br/>Quality Control]
    end
    
    subgraph "AI Services Layer"
        Ollama[🦙 Ollama Server<br/>Local LLMs<br/>Port 11434]
        HuggingFace[🤗 Hugging Face<br/>External Models]
        EmbeddingService[🔢 Embedding Service<br/>Vector Generation]
        RerankerService[📊 Reranker Service<br/>Result Refinement]
    end
    
    subgraph "Data Persistence Layer"
        SQLite[(📊 SQLite<br/>Metadata & Sessions)]
        LanceDB[(🎯 LanceDB<br/>Vector Database)]
        BM25Index[(🔍 BM25 Index<br/>Keyword Search)]
        FileStorage[📁 File Storage<br/>Document Repository)]
    end
    
    subgraph "Infrastructure Layer"
        Docker[🐳 Docker Containers]
        Monitoring[📈 Monitoring<br/>Logs & Metrics]
        Backup[💾 Backup System<br/>Data Protection]
    end
    
    User --> WebUI
    Admin --> WebUI
    API_Consumer --> Backend
    
    WebUI --> Backend
    MobileApp --> Backend
    
    Backend --> Auth
    Backend --> RateLimit
    Backend --> RAG_API
    
    RAG_API --> Agent
    Agent --> Triage
    Agent --> Verifier
    
    Agent --> Ollama
    Agent --> HuggingFace
    Agent --> EmbeddingService
    Agent --> RerankerService
    
    Agent --> SQLite
    Agent --> LanceDB
    Agent --> BM25Index
    Agent --> FileStorage
    
    Backend --> SQLite
    
    Docker --> Monitoring
    Docker --> Backup
```

### Technology Stack Overview

```mermaid
graph TB
    subgraph "Frontend Technologies"
        React[React 19]
        NextJS[Next.js 15]
        TypeScript[TypeScript]
        TailwindCSS[Tailwind CSS]
    end
    
    subgraph "Backend Technologies"
        Python[Python 3.11]
        HTTPServer[HTTP Server]
        SQLite3[SQLite3]
        AsyncIO[AsyncIO]
    end
    
    subgraph "AI/ML Technologies"
        Transformers[🤗 Transformers]
        PyTorch[PyTorch]
        SentenceTransformers[Sentence Transformers]
        FAISS[FAISS - Future]
    end
    
    subgraph "Database Technologies"
        LanceDB[LanceDB Vector DB]
        SQLiteDB[SQLite RDBMS]
        BM25[BM25 Search]
        FTS[Full-Text Search]
    end
    
    subgraph "Infrastructure"
        DockerEngine[Docker Engine]
        DockerCompose[Docker Compose]
        Nginx[Nginx - Production]
        SystemD[SystemD - Linux]
    end
    
    React --> NextJS
    NextJS --> TypeScript
    TypeScript --> TailwindCSS
    
    Python --> HTTPServer
    Python --> SQLite3
    Python --> AsyncIO
    
    Transformers --> PyTorch
    PyTorch --> SentenceTransformers
    
    LanceDB --> SQLiteDB
    SQLiteDB --> BM25
    BM25 --> FTS
    
    DockerEngine --> DockerCompose
    DockerCompose --> Nginx
```

---

## Component Architecture

### Frontend Component Hierarchy

```mermaid
graph TB
    App[App.tsx<br/>Root Application]
    
    App --> Layout[Layout.tsx<br/>Page Layout]
    App --> Demo[Demo.tsx<br/>Main Container]
    
    Demo --> Sidebar[Sidebar<br/>Navigation]
    Demo --> MainContent[Main Content Area]
    
    Sidebar --> SessionList[Session List]
    Sidebar --> IndexList[Index List]
    Sidebar --> Settings[Settings Panel]
    
    MainContent --> LandingMenu[Landing Menu<br/>HOME Mode]
    MainContent --> ChatInterface[Chat Interface<br/>CHAT Mode]
    MainContent --> IndexWizard[Index Wizard<br/>INDEX Mode]
    
    ChatInterface --> SessionChat[Session Chat]
    ChatInterface --> QuickChat[Quick Chat]
    ChatInterface --> SessionIndexInfo[Index Info Display]
    
    SessionChat --> ChatBubble[Chat Bubble]
    SessionChat --> ChatInput[Chat Input]
    SessionChat --> MessageLoading[Loading States]
    
    IndexWizard --> IndexForm[Index Form]
    IndexWizard --> IndexPicker[Index Picker]
    IndexWizard --> ModelSelect[Model Selection]
    
    subgraph "Shared UI Components"
        Button[Button]
        Input[Input]
        Modal[Modal]
        Tooltip[Tooltip]
        Avatar[Avatar]
        Badge[Badge]
    end
```

### Backend Service Architecture

```mermaid
graph TB
    subgraph "Backend Server (Port 8000)"
        HTTPHandler[HTTP Request Handler]
        Router[Request Router]
        
        Router --> SessionHandler[Session Handler]
        Router --> IndexHandler[Index Handler]
        Router --> ChatHandler[Chat Handler]
        Router --> UploadHandler[Upload Handler]
        Router --> HealthHandler[Health Handler]
        
        SessionHandler --> SessionService[Session Service]
        IndexHandler --> IndexService[Index Service]
        ChatHandler --> ChatService[Chat Service]
        UploadHandler --> FileService[File Service]
        
        SessionService --> Database[Database Layer]
        IndexService --> Database
        ChatService --> RAGClient[RAG API Client]
        FileService --> FileStorage[File Storage]
    end
    
    subgraph "RAG API Server (Port 8001)"
        RAGHandler[RAG Request Handler]
        RAGRouter[RAG Router]
        
        RAGRouter --> ChatEndpoint[Chat Endpoint]
        RAGRouter --> StreamEndpoint[Stream Endpoint]
        RAGRouter --> IndexEndpoint[Index Endpoint]
        RAGRouter --> ModelsEndpoint[Models Endpoint]
        
        ChatEndpoint --> AgentService[Agent Service]
        StreamEndpoint --> AgentService
        IndexEndpoint --> IndexingService[Indexing Service]
        ModelsEndpoint --> ModelService[Model Service]
        
        AgentService --> RAGAgent[RAG Agent Core]
        IndexingService --> IndexingPipeline[Indexing Pipeline]
        ModelService --> ModelManager[Model Manager]
    end
    
    Database --> SQLiteConn[SQLite Connection]
    RAGAgent --> VectorDB[Vector Database]
    RAGAgent --> LLMService[LLM Service]
    IndexingPipeline --> VectorDB
    IndexingPipeline --> FileStorage
```

### RAG Agent Internal Architecture

```mermaid
graph TB
    subgraph "RAG Agent Core"
        AgentController[Agent Controller<br/>Main Orchestrator]
        
        AgentController --> TriageSystem[Query Triage System]
        AgentController --> RetrievalPipeline[Retrieval Pipeline]
        AgentController --> GenerationPipeline[Generation Pipeline]
        AgentController --> VerificationSystem[Verification System]
    end
    
    subgraph "Triage System"
        QueryClassifier[Query Classifier]
        RouteDecision[Route Decision Engine]
        
        QueryClassifier --> GreetingDetector[Greeting Detector]
        QueryClassifier --> DocumentQueryDetector[Document Query Detector]
        QueryClassifier --> ComplexityAnalyzer[Complexity Analyzer]
        
        RouteDecision --> DirectLLMRoute[Direct LLM Route]
        RouteDecision --> RAGRoute[RAG Route]
        RouteDecision --> AdvancedRAGRoute[Advanced RAG Route]
    end
    
    subgraph "Retrieval Pipeline"
        QueryProcessor[Query Processor]
        MultiRetriever[Multi-Modal Retriever]
        ResultFusion[Result Fusion]
        Reranker[AI Reranker]
        
        QueryProcessor --> QueryTransformer[Query Transformer]
        QueryProcessor --> QueryDecomposer[Query Decomposer]
        
        MultiRetriever --> VectorRetriever[Vector Retriever]
        MultiRetriever --> BM25Retriever[BM25 Retriever]
        MultiRetriever --> HybridRetriever[Hybrid Retriever]
        
        ResultFusion --> ScoreFusion[Score Fusion]
        ResultFusion --> RankFusion[Rank Fusion]
        
        Reranker --> CrossEncoder[Cross-Encoder]
        Reranker --> SentencePruner[Sentence Pruner]
    end
    
    subgraph "Generation Pipeline"
        ContextAssembler[Context Assembler]
        PromptBuilder[Prompt Builder]
        LLMInterface[LLM Interface]
        ResponseProcessor[Response Processor]
        
        ContextAssembler --> ContextWindow[Context Window Manager]
        ContextAssembler --> SourceAttribution[Source Attribution]
        
        PromptBuilder --> TemplateEngine[Template Engine]
        PromptBuilder --> ContextInjector[Context Injector]
        
        LLMInterface --> OllamaClient[Ollama Client]
        LLMInterface --> HuggingFaceClient[HuggingFace Client]
        
        ResponseProcessor --> SourceLinker[Source Linker]
        ResponseProcessor --> QualityFilter[Quality Filter]
    end
    
    subgraph "Verification System"
        FactChecker[Fact Checker]
        GroundingVerifier[Grounding Verifier]
        ConfidenceScorer[Confidence Scorer]
        
        FactChecker --> SourceValidator[Source Validator]
        GroundingVerifier --> ClaimExtractor[Claim Extractor]
        ConfidenceScorer --> UncertaintyDetector[Uncertainty Detector]
    end
```

---

## Data Flow Diagrams

### Document Indexing Flow

```mermaid
flowchart TD
    Start([User Uploads Document]) --> ValidateFile{Validate File Type}
    ValidateFile -->|Valid| StoreFile[Store File in FileSystem]
    ValidateFile -->|Invalid| Error1[Return Error: Unsupported Format]
    
    StoreFile --> CreateIndex[Create Index Record in DB]
    CreateIndex --> ConvertDocument[Convert Document to Text]
    
    ConvertDocument --> ChunkDocument[Chunk Document into Segments]
    ChunkDocument --> EnrichChunks{Enrichment Enabled?}
    
    EnrichChunks -->|Yes| ContextualEnrich[Add Contextual Information]
    EnrichChunks -->|No| GenerateEmbeddings[Generate Vector Embeddings]
    ContextualEnrich --> GenerateEmbeddings
    
    GenerateEmbeddings --> StoreVectors[Store Vectors in LanceDB]
    StoreVectors --> CreateBM25[Create BM25 Index]
    CreateBM25 --> CreateFTS[Create Full-Text Search Index]
    
    CreateFTS --> UpdateMetadata[Update Index Metadata]
    UpdateMetadata --> IndexingComplete([Indexing Complete])
    
    Error1 --> End([End])
    IndexingComplete --> End
    
    subgraph "Parallel Processing"
        StoreVectors
        CreateBM25
        CreateFTS
    end
```

### Query Processing Flow

```mermaid
flowchart TD
    UserQuery([User Submits Query]) --> ValidateInput{Validate Input}
    ValidateInput -->|Invalid| ErrorResponse[Return Validation Error]
    ValidateInput -->|Valid| LoadSession[Load Session Context]
    
    LoadSession --> QueryTriage[Query Triage Analysis]
    
    QueryTriage --> DirectLLM{Route to Direct LLM?}
    DirectLLM -->|Yes| CallLLM[Call LLM Directly]
    DirectLLM -->|No| RAGPipeline[Enter RAG Pipeline]
    
    RAGPipeline --> DecomposeQuery{Decompose Query?}
    DecomposeQuery -->|Yes| QueryDecomposition[Break into Sub-queries]
    DecomposeQuery -->|No| VectorSearch[Vector Search]
    
    QueryDecomposition --> ParallelRetrieval[Parallel Retrieval for Sub-queries]
    ParallelRetrieval --> CombineResults[Combine Sub-query Results]
    CombineResults --> VectorSearch
    
    VectorSearch --> BM25Search[BM25 Keyword Search]
    BM25Search --> FuseResults[Fuse Search Results]
    
    FuseResults --> AIRerank{AI Reranking Enabled?}
    AIRerank -->|Yes| RerankerModel[Apply Reranker Model]
    AIRerank -->|No| AssembleContext[Assemble Context]
    RerankerModel --> AssembleContext
    
    AssembleContext --> GenerateResponse[Generate LLM Response]
    GenerateResponse --> VerifyResponse{Verification Enabled?}
    
    VerifyResponse -->|Yes| FactCheck[Fact Check Against Sources]
    VerifyResponse -->|No| FormatResponse[Format Final Response]
    FactCheck --> ConfidenceScore[Calculate Confidence Score]
    ConfidenceScore --> FormatResponse
    
    CallLLM --> FormatResponse
    FormatResponse --> SaveToHistory[Save to Chat History]
    SaveToHistory --> ReturnResponse([Return Response to User])
    
    ErrorResponse --> End([End])
    ReturnResponse --> End
```

### Real-time Chat Flow

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant Backend
    participant RAG_API
    participant Agent
    participant LLM
    participant VectorDB
    
    User->>Frontend: Type message
    Frontend->>Backend: POST /chat
    Backend->>RAG_API: Forward request
    
    RAG_API->>Agent: Process query
    Agent->>Agent: Query triage
    
    alt Direct LLM Path
        Agent->>LLM: Direct query
        LLM-->>Agent: Response
    else RAG Path
        Agent->>VectorDB: Vector search
        VectorDB-->>Agent: Relevant chunks
        Agent->>Agent: Fuse results
        Agent->>LLM: Generate with context
        LLM-->>Agent: Contextual response
        Agent->>Agent: Verify response
    end
    
    Agent-->>RAG_API: Structured response
    RAG_API-->>Backend: JSON response
    Backend-->>Frontend: HTTP response
    Frontend->>Frontend: Update chat UI
    Frontend-->>User: Display response
```

### Streaming Response Flow

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant Backend
    participant RAG_API
    participant Agent
    participant LLM
    
    User->>Frontend: Submit query
    Frontend->>Backend: POST /chat (stream=true)
    Backend->>RAG_API: POST /chat/stream
    
    RAG_API->>Agent: Initialize streaming
    Agent->>Agent: Start processing
    
    Agent-->>RAG_API: Event: "processing"
    RAG_API-->>Backend: SSE: processing
    Backend-->>Frontend: SSE: processing
    Frontend->>Frontend: Show "thinking" indicator
    
    Agent->>LLM: Start generation
    
    loop Streaming tokens
        LLM-->>Agent: Token chunk
        Agent-->>RAG_API: Event: "chunk"
        RAG_API-->>Backend: SSE: chunk
        Backend-->>Frontend: SSE: chunk
        Frontend->>Frontend: Append to response
    end
    
    Agent-->>RAG_API: Event: "complete"
    RAG_API-->>Backend: SSE: complete
    Backend-->>Frontend: SSE: complete
    Frontend->>Frontend: Finalize response
    Frontend-->>User: Complete response visible
```

---

## Processing Pipelines

### Document Processing Pipeline

```mermaid
graph TB
    subgraph "Input Stage"
        FileUpload[📄 File Upload]
        FileValidation[✅ File Validation]
        FileStorage[💾 File Storage]
    end
    
    subgraph "Conversion Stage"
        FormatDetection[🔍 Format Detection]
        PDFConverter[📄 PDF Converter]
        DOCXConverter[📝 DOCX Converter]
        TXTConverter[📃 TXT Converter]
        MarkdownConverter[📋 Markdown Converter]
    end
    
    subgraph "Text Processing Stage"
        TextExtraction[📝 Text Extraction]
        TextCleaning[🧹 Text Cleaning]
        LanguageDetection[🌐 Language Detection]
        TextNormalization[⚖️ Text Normalization]
    end
    
    subgraph "Chunking Stage"
        ChunkingStrategy[🎯 Chunking Strategy]
        SemanticChunking[🧠 Semantic Chunking]
        FixedSizeChunking[📏 Fixed Size Chunking]
        OverlapManagement[🔄 Overlap Management]
    end
    
    subgraph "Enrichment Stage"
        ContextualEnrichment[🎨 Contextual Enrichment]
        MetadataExtraction[📊 Metadata Extraction]
        EntityRecognition[🏷️ Entity Recognition]
        TopicModeling[📈 Topic Modeling]
    end
    
    subgraph "Embedding Stage"
        EmbeddingGeneration[🔢 Embedding Generation]
        VectorNormalization[⚖️ Vector Normalization]
        DimensionReduction[📉 Dimension Reduction]
        QualityValidation[✅ Quality Validation]
    end
    
    subgraph "Indexing Stage"
        VectorIndexing[🎯 Vector Indexing]
        BM25Indexing[🔍 BM25 Indexing]
        FTSIndexing[📝 Full-Text Search]
        MetadataIndexing[📊 Metadata Indexing]
    end
    
    FileUpload --> FileValidation
    FileValidation --> FileStorage
    FileStorage --> FormatDetection
    
    FormatDetection --> PDFConverter
    FormatDetection --> DOCXConverter
    FormatDetection --> TXTConverter
    FormatDetection --> MarkdownConverter
    
    PDFConverter --> TextExtraction
    DOCXConverter --> TextExtraction
    TXTConverter --> TextExtraction
    MarkdownConverter --> TextExtraction
    
    TextExtraction --> TextCleaning
    TextCleaning --> LanguageDetection
    LanguageDetection --> TextNormalization
    
    TextNormalization --> ChunkingStrategy
    ChunkingStrategy --> SemanticChunking
    ChunkingStrategy --> FixedSizeChunking
    SemanticChunking --> OverlapManagement
    FixedSizeChunking --> OverlapManagement
    
    OverlapManagement --> ContextualEnrichment
    ContextualEnrichment --> MetadataExtraction
    MetadataExtraction --> EntityRecognition
    EntityRecognition --> TopicModeling
    
    TopicModeling --> EmbeddingGeneration
    EmbeddingGeneration --> VectorNormalization
    VectorNormalization --> DimensionReduction
    DimensionReduction --> QualityValidation
    
    QualityValidation --> VectorIndexing
    QualityValidation --> BM25Indexing
    QualityValidation --> FTSIndexing
    QualityValidation --> MetadataIndexing
```

### Query Understanding Pipeline

```mermaid
graph TB
    subgraph "Input Processing"
        QueryInput[❓ User Query]
        InputValidation[✅ Input Validation]
        QueryNormalization[⚖️ Query Normalization]
        LanguageDetection[🌐 Language Detection]
    end
    
    subgraph "Query Analysis"
        IntentClassification[🎯 Intent Classification]
        EntityExtraction[🏷️ Entity Extraction]
        ComplexityAnalysis[📊 Complexity Analysis]
        ContextAnalysis[🔍 Context Analysis]
    end
    
    subgraph "Query Enhancement"
        QueryExpansion[📈 Query Expansion]
        SynonymReplacement[🔄 Synonym Replacement]
        ContextInjection[💉 Context Injection]
        HistoryIntegration[📚 History Integration]
    end
    
    subgraph "Query Decomposition"
        SubQueryExtraction[🧩 Sub-query Extraction]
        DependencyAnalysis[🔗 Dependency Analysis]
        PriorityRanking[📊 Priority Ranking]
        ParallelizationPlan[⚡ Parallelization Plan]
    end
    
    subgraph "Retrieval Strategy"
        SearchStrategySelection[🎯 Search Strategy Selection]
        VectorSearchConfig[🔢 Vector Search Config]
        KeywordSearchConfig[🔍 Keyword Search Config]
        HybridSearchConfig[🔄 Hybrid Search Config]
    end
    
    QueryInput --> InputValidation
    InputValidation --> QueryNormalization
    QueryNormalization --> LanguageDetection
    
    LanguageDetection --> IntentClassification
    IntentClassification --> EntityExtraction
    EntityExtraction --> ComplexityAnalysis
    ComplexityAnalysis --> ContextAnalysis
    
    ContextAnalysis --> QueryExpansion
    QueryExpansion --> SynonymReplacement
    SynonymReplacement --> ContextInjection
    ContextInjection --> HistoryIntegration
    
    HistoryIntegration --> SubQueryExtraction
    SubQueryExtraction --> DependencyAnalysis
    DependencyAnalysis --> PriorityRanking
    PriorityRanking --> ParallelizationPlan
    
    ParallelizationPlan --> SearchStrategySelection
    SearchStrategySelection --> VectorSearchConfig
    SearchStrategySelection --> KeywordSearchConfig
    SearchStrategySelection --> HybridSearchConfig
```

---

## API Interaction Flows

### Session Management Flow

```mermaid
sequenceDiagram
    participant Client
    participant Backend
    participant Database
    
    Note over Client,Database: Create Session
    Client->>Backend: POST /sessions
    Backend->>Backend: Validate request
    Backend->>Database: INSERT session
    Database-->>Backend: Session ID
    Backend-->>Client: Session created
    
    Note over Client,Database: List Sessions
    Client->>Backend: GET /sessions
    Backend->>Database: SELECT sessions
    Database-->>Backend: Session list
    Backend-->>Client: Sessions with metadata
    
    Note over Client,Database: Update Session
    Client->>Backend: PUT /sessions/{id}
    Backend->>Database: UPDATE session
    Database-->>Backend: Update confirmation
    Backend-->>Client: Session updated
    
    Note over Client,Database: Delete Session
    Client->>Backend: DELETE /sessions/{id}
    Backend->>Database: DELETE session
    Backend->>Database: DELETE related messages
    Database-->>Backend: Deletion confirmation
    Backend-->>Client: Session deleted
```

### Index Lifecycle Flow

```mermaid
sequenceDiagram
    participant Client
    participant Backend
    participant RAG_API
    participant FileSystem
    participant VectorDB
    
    Note over Client,VectorDB: Create Index
    Client->>Backend: POST /indexes
    Backend->>Backend: Generate index ID
    Backend->>Backend: Store index metadata
    Backend-->>Client: Index ID
    
    Note over Client,VectorDB: Upload Documents
    Client->>Backend: POST /indexes/{id}/upload
    Backend->>FileSystem: Store documents
    FileSystem-->>Backend: File paths
    Backend-->>Client: Upload confirmation
    
    Note over Client,VectorDB: Build Index
    Client->>Backend: POST /indexes/{id}/build
    Backend->>RAG_API: Trigger indexing
    RAG_API->>RAG_API: Process documents
    RAG_API->>VectorDB: Store embeddings
    RAG_API->>RAG_API: Create search indexes
    RAG_API-->>Backend: Indexing complete
    Backend-->>Client: Build confirmation
    
    Note over Client,VectorDB: Query Index
    Client->>Backend: POST /chat
    Backend->>RAG_API: Forward query
    RAG_API->>VectorDB: Search vectors
    VectorDB-->>RAG_API: Relevant chunks
    RAG_API-->>Backend: Response with sources
    Backend-->>Client: Final response
```

### Error Handling Flow

```mermaid
flowchart TD
    Request[Incoming Request] --> Validate{Validate Request}
    Validate -->|Valid| Process[Process Request]
    Validate -->|Invalid| ValidationError[Return 400 Validation Error]
    
    Process --> CheckAuth{Authentication Required?}
    CheckAuth -->|Yes| AuthCheck{User Authenticated?}
    CheckAuth -->|No| ExecuteLogic[Execute Business Logic]
    
    AuthCheck -->|Yes| ExecuteLogic
    AuthCheck -->|No| AuthError[Return 401 Unauthorized]
    
    ExecuteLogic --> BusinessLogic{Business Logic Success?}
    BusinessLogic -->|Success| SuccessResponse[Return Success Response]
    BusinessLogic -->|Failure| HandleError[Handle Business Error]
    
    HandleError --> ErrorType{Error Type}
    ErrorType -->|Not Found| NotFoundError[Return 404 Not Found]
    ErrorType -->|Conflict| ConflictError[Return 409 Conflict]
    ErrorType -->|Server Error| ServerError[Return 500 Internal Error]
    ErrorType -->|Timeout| TimeoutError[Return 408 Timeout]
    
    ValidationError --> LogError[Log Error Details]
    AuthError --> LogError
    NotFoundError --> LogError
    ConflictError --> LogError
    ServerError --> LogError
    TimeoutError --> LogError
    
    LogError --> ErrorResponse[Format Error Response]
    ErrorResponse --> ReturnError[Return Error to Client]
    
    SuccessResponse --> LogSuccess[Log Success]
    LogSuccess --> ReturnSuccess[Return Success to Client]
```

---

## Database Relationships

### Entity Relationship Diagram

```mermaid
erDiagram
    SESSIONS {
        string id PK
        string title
        string model
        string embedding_model
        timestamp created_at
        timestamp updated_at
    }
    
    MESSAGES {
        integer id PK
        string session_id FK
        string role
        text content
        text metadata
        timestamp created_at
    }
    
    INDEXES {
        string id PK
        string name
        text description
        string status
        string vector_table_name
        text metadata
        timestamp created_at
        timestamp updated_at
    }
    
    SESSION_INDEXES {
        string session_id FK
        string index_id FK
        timestamp linked_at
    }
    
    DOCUMENTS {
        string id PK
        string index_id FK
        string filename
        string file_path
        string content_type
        integer file_size
        text metadata
        timestamp uploaded_at
    }
    
    CHUNKS {
        string id PK
        string document_id FK
        integer chunk_index
        text content
        text metadata
        timestamp created_at
    }
    
    SESSIONS ||--o{ MESSAGES : "has"
    SESSIONS ||--o{ SESSION_INDEXES : "links to"
    INDEXES ||--o{ SESSION_INDEXES : "linked by"
    INDEXES ||--o{ DOCUMENTS : "contains"
    DOCUMENTS ||--o{ CHUNKS : "split into"
```

### Vector Database Schema

```mermaid
graph TB
    subgraph "LanceDB Tables"
        VectorTable[Vector Table<br/>text_pages_{index_id}]
        MetadataTable[Metadata Table<br/>chunk_metadata]
        IndexTable[Index Configuration<br/>index_config]
    end
    
    subgraph "Vector Table Schema"
        VectorField[vector: List[Float32]]
        TextField[text: String]
        ChunkIdField[chunk_id: String]
        DocumentIdField[document_id: String]
        ChunkIndexField[chunk_index: Int32]
        MetadataField[metadata: String JSON]
    end
    
    subgraph "Search Indexes"
        VectorIndex[Vector Index<br/>HNSW/IVF]
        FTSIndex[Full-Text Search<br/>Tantivy]
        MetadataIndex[Metadata Index<br/>B-Tree]
    end
    
    VectorTable --> VectorField
    VectorTable --> TextField
    VectorTable --> ChunkIdField
    VectorTable --> DocumentIdField
    VectorTable --> ChunkIndexField
    VectorTable --> MetadataField
    
    VectorField --> VectorIndex
    TextField --> FTSIndex
    MetadataField --> MetadataIndex
```

---

## Deployment Architecture

### Docker Container Architecture

```mermaid
graph TB
    subgraph "Docker Host"
        subgraph "Frontend Container"
            NextJS[Next.js Application<br/>Port 3000]
            StaticFiles[Static Assets]
        end
        
        subgraph "Backend Container"
            PythonApp[Python HTTP Server<br/>Port 8000]
            SQLiteDB[SQLite Database]
            UploadStorage[Upload Storage]
        end
        
        subgraph "RAG API Container"
            RAGServer[RAG API Server<br/>Port 8001]
            VectorDB[LanceDB Storage]
            ModelCache[Model Cache]
        end
        
        subgraph "Ollama Container"
            OllamaServer[Ollama Server<br/>Port 11434]
            ModelStorage[Model Storage<br/>~10GB]
        end
        
        subgraph "Shared Volumes"
            DocumentVolume[Document Storage]
            DatabaseVolume[Database Storage]
            ModelVolume[Model Storage]
            LogVolume[Log Storage]
        end
    end
    
    NextJS --> PythonApp
    PythonApp --> RAGServer
    RAGServer --> OllamaServer
    
    PythonApp --> SQLiteDB
    PythonApp --> UploadStorage
    RAGServer --> VectorDB
    RAGServer --> ModelCache
    OllamaServer --> ModelStorage
    
    UploadStorage --> DocumentVolume
    SQLiteDB --> DatabaseVolume
    VectorDB --> DatabaseVolume
    ModelStorage --> ModelVolume
    ModelCache --> ModelVolume
```

### Production Deployment

```mermaid
graph TB
    subgraph "Load Balancer Layer"
        LB[Load Balancer<br/>Nginx/HAProxy]
        SSL[SSL Termination]
    end
    
    subgraph "Application Layer"
        subgraph "Frontend Cluster"
            FE1[Frontend Instance 1]
            FE2[Frontend Instance 2]
            FE3[Frontend Instance 3]
        end
        
        subgraph "Backend Cluster"
            BE1[Backend Instance 1]
            BE2[Backend Instance 2]
            BE3[Backend Instance 3]
        end
        
        subgraph "RAG Cluster"
            RAG1[RAG Instance 1]
            RAG2[RAG Instance 2]
        end
    end
    
    subgraph "AI Services Layer"
        subgraph "Ollama Cluster"
            OL1[Ollama Instance 1<br/>GPU Node]
            OL2[Ollama Instance 2<br/>GPU Node]
        end
    end
    
    subgraph "Data Layer"
        subgraph "Database Cluster"
            DB1[(Primary Database)]
            DB2[(Replica Database)]
        end
        
        subgraph "Vector Storage"
            VDB1[(Vector DB Primary)]
            VDB2[(Vector DB Replica)]
        end
        
        subgraph "File Storage"
            FS1[File Storage<br/>NFS/S3]
        end
    end
    
    subgraph "Infrastructure"
        Monitor[Monitoring<br/>Prometheus/Grafana]
        Logs[Centralized Logging<br/>ELK Stack]
        Backup[Backup System]
    end
    
    Internet --> LB
    LB --> SSL
    SSL --> FE1
    SSL --> FE2
    SSL --> FE3
    
    FE1 --> BE1
    FE2 --> BE2
    FE3 --> BE3
    
    BE1 --> RAG1
    BE2 --> RAG2
    BE3 --> RAG1
    
    RAG1 --> OL1
    RAG2 --> OL2
    
    BE1 --> DB1
    BE2 --> DB1
    BE3 --> DB2
    
    RAG1 --> VDB1
    RAG2 --> VDB2
    
    DB1 --> FS1
    VDB1 --> FS1
    
    Monitor --> FE1
    Monitor --> BE1
    Monitor --> RAG1
    
    Logs --> FE1
    Logs --> BE1
    Logs --> RAG1
    
    Backup --> DB1
    Backup --> VDB1
    Backup --> FS1
```

---

This comprehensive architectural documentation provides detailed visual representations of LocalGPT's system design, component interactions, data flows, and deployment strategies. These diagrams serve as the foundation for understanding the system's structure and can be used for development, maintenance, and scaling decisions. 