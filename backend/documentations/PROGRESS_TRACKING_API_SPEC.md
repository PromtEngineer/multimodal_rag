# Progress Tracking API Specification

## Overview

This document outlines the API changes required to integrate real-time progress tracking for document upload and indexing operations in the multimodal RAG system.

## Backend API Changes

### 1. Enhanced Indexing Endpoint with Progress Support

#### `POST /sessions/{session_id}/index_with_progress`

**Purpose**: Start document indexing with real-time progress tracking

**Request Body**:
```json
{
  "batch_size": 50,
  "chunk_batch_size": 10,
  "enable_contextual_enrichment": true,
  "progress_session_id": "uuid-string" // Optional, auto-generated if not provided
}
```

**Response**:
```json
{
  "progress_session_id": "uuid-string",
  "status": "started",
  "message": "Indexing started with progress tracking",
  "estimated_duration": "2-5 minutes",
  "total_files": 5
}
```

### 2. Server-Sent Events (SSE) Endpoint

#### `GET /sessions/{session_id}/progress/stream?progress_session_id={progress_session_id}`

**Purpose**: Real-time progress updates via Server-Sent Events

**Response Stream Format**:
```
event: progress
data: {"step": "chunking", "current": 25, "total": 100, "percentage": 25.0, "eta": "2m 30s", "message": "Processing chunks..."}

event: step_complete
data: {"step": "chunking", "duration": "45s", "items_processed": 100, "message": "Chunking completed"}

event: error
data: {"step": "embedding", "error": "Connection timeout", "retry_possible": true}

event: complete
data: {"total_duration": "3m 15s", "total_items": 100, "message": "Indexing completed successfully"}
```

### 3. Progress Status Endpoint

#### `GET /sessions/{session_id}/progress/{progress_session_id}`

**Purpose**: Get current progress status (for polling fallback)

**Response**:
```json
{
  "progress_session_id": "uuid-string",
  "status": "in_progress", // "not_started" | "in_progress" | "completed" | "error"
  "current_step": "embedding",
  "overall_progress": {
    "current": 75,
    "total": 100,
    "percentage": 75.0,
    "eta": "1m 15s"
  },
  "step_progress": {
    "chunking": {"status": "completed", "duration": "45s", "items": 100},
    "contextual_enrichment": {"status": "completed", "duration": "2m 10s", "items": 100},
    "embedding": {"status": "in_progress", "current": 75, "total": 100, "eta": "1m 15s"},
    "indexing": {"status": "pending"}
  },
  "error": null,
  "started_at": "2024-01-15T10:30:00Z",
  "estimated_completion": "2024-01-15T10:35:00Z"
}
```

### 4. Progress Cancellation Endpoint

#### `POST /sessions/{session_id}/progress/{progress_session_id}/cancel`

**Purpose**: Cancel ongoing indexing operation

**Response**:
```json
{
  "message": "Indexing operation cancelled",
  "progress_session_id": "uuid-string",
  "status": "cancelled",
  "partial_results": {
    "indexed_chunks": 45,
    "completed_steps": ["chunking", "contextual_enrichment"]
  }
}
```

### 5. Enhanced Health Check

#### `GET /health` (Enhanced)

**Additional fields in existing response**:
```json
{
  "status": "healthy",
  "ollama_running": true,
  "available_models": ["llama3.2:latest"],
  "database_stats": {...},
  "progress_tracking": {
    "active_sessions": 2,
    "sse_connections": 3,
    "batch_processor_status": "ready"
  }
}
```

## Frontend API Changes

### 1. Enhanced ChatAPI Class Methods

#### New Methods to Add to `src/lib/api.ts`

```typescript
// New interfaces
interface IndexingStartResponse {
  progress_session_id: string;
  status: string;
  message: string;
  estimated_duration: string;
  total_files: number;
}

interface ProgressData {
  step: string;
  current: number;
  total: number;
  percentage: number;
  eta: string;
  message: string;
}

interface StepCompleteData {
  step: string;
  duration: string;
  items_processed: number;
  message: string;
}

interface ProgressStatus {
  progress_session_id: string;
  status: 'not_started' | 'in_progress' | 'completed' | 'error' | 'cancelled';
  current_step: string;
  overall_progress: {
    current: number;
    total: number;
    percentage: number;
    eta: string;
  };
  step_progress: Record<string, {
    status: string;
    duration?: string;
    items?: number;
    current?: number;
    total?: number;
    eta?: string;
  }>;
  error: string | null;
  started_at: string;
  estimated_completion: string;
}

interface ProgressCallbacks {
  onProgress?: (data: ProgressData) => void;
  onStepComplete?: (data: StepCompleteData) => void;
  onError?: (error: string) => void;
  onComplete?: (data: any) => void;
  onConnectionError?: () => void;
}

// New methods for ChatAPI class
class ChatAPI {
  // ... existing methods ...

  /**
   * Start indexing with progress tracking
   */
  async startIndexingWithProgress(
    sessionId: string, 
    options?: {
      batch_size?: number;
      chunk_batch_size?: number;
      enable_contextual_enrichment?: boolean;
    }
  ): Promise<IndexingStartResponse> {
    const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/index_with_progress`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(options || {})
    });
    
    if (!response.ok) {
      throw new Error(`Failed to start indexing: ${response.statusText}`);
    }
    
    return await response.json();
  }

  /**
   * Get current progress status
   */
  async getProgressStatus(sessionId: string, progressSessionId: string): Promise<ProgressStatus> {
    const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/progress/${progressSessionId}`);
    
    if (!response.ok) {
      throw new Error(`Failed to get progress: ${response.statusText}`);
    }
    
    return await response.json();
  }

  /**
   * Connect to progress stream via Server-Sent Events
   */
  connectProgressStream(
    sessionId: string, 
    progressSessionId: string, 
    callbacks: ProgressCallbacks
  ): () => void {
    const eventSource = new EventSource(
      `${API_BASE_URL}/sessions/${sessionId}/progress/stream?progress_session_id=${progressSessionId}`
    );

    eventSource.addEventListener('progress', (event) => {
      const data: ProgressData = JSON.parse(event.data);
      callbacks.onProgress?.(data);
    });

    eventSource.addEventListener('step_complete', (event) => {
      const data: StepCompleteData = JSON.parse(event.data);
      callbacks.onStepComplete?.(data);
    });

    eventSource.addEventListener('error', (event) => {
      const data = JSON.parse(event.data);
      callbacks.onError?.(data.error);
    });

    eventSource.addEventListener('complete', (event) => {
      const data = JSON.parse(event.data);
      callbacks.onComplete?.(data);
      eventSource.close();
    });

    eventSource.onerror = () => {
      callbacks.onConnectionError?.();
    };

    // Return cleanup function
    return () => {
      eventSource.close();
    };
  }

  /**
   * Cancel ongoing indexing operation
   */
  async cancelIndexing(sessionId: string, progressSessionId: string): Promise<any> {
    const response = await fetch(
      `${API_BASE_URL}/sessions/${sessionId}/progress/${progressSessionId}/cancel`, 
      { method: 'POST' }
    );
    
    if (!response.ok) {
      throw new Error(`Failed to cancel indexing: ${response.statusText}`);
    }
    
    return await response.json();
  }
}
```

### 2. React Hook for Progress Management

#### New Hook: `useIndexingProgress`

```typescript
// src/hooks/useIndexingProgress.ts
interface UseIndexingProgressReturn {
  // State
  isIndexing: boolean;
  progressData: ProgressData | null;
  progressStatus: ProgressStatus | null;
  error: string | null;
  
  // Actions
  startIndexing: (sessionId: string, options?: any) => Promise<void>;
  cancelIndexing: () => Promise<void>;
  
  // Progress details
  currentStep: string;
  overallProgress: number;
  stepProgress: Record<string, any>;
  eta: string;
  completedSteps: string[];
}

export function useIndexingProgress(): UseIndexingProgressReturn;
```

## UI Components API

### 1. Enhanced Progress Components

#### `ProgressBar` Component
```typescript
interface ProgressBarProps {
  current: number;
  total: number;
  className?: string;
  showPercentage?: boolean;
  animated?: boolean;
}
```

#### `IndexingProgressPanel` Component
```typescript
interface IndexingProgressPanelProps {
  sessionId: string;
  isVisible: boolean;
  onComplete?: () => void;
  onCancel?: () => void;
  onError?: (error: string) => void;
}
```

#### `ProgressStepIndicator` Component
```typescript
interface ProgressStepIndicatorProps {
  steps: Array<{
    id: string;
    label: string;
    status: 'pending' | 'in_progress' | 'completed' | 'error';
    duration?: string;
    items?: number;
  }>;
  currentStep: string;
}
```

### 2. Enhanced DocumentUpload Component

#### Modified `EmptyChatState` Props
```typescript
interface EmptyChatStateProps {
  onSendMessage: (message: string, attachedFiles?: AttachedFile[]) => void;
  disabled?: boolean;
  placeholder?: string;
  // New props for progress tracking
  onStartIndexing?: (sessionId: string) => void;
  showProgressPanel?: boolean;
  progressSessionId?: string;
}
```

## Data Flow Architecture

### 1. Upload & Indexing Flow with Progress

```
User selects files → EmptyChatState
    ↓
SessionChat.sendMessage() with files
    ↓
chatAPI.uploadFiles() (existing)
    ↓
Show "Index Documents" button
    ↓
User clicks "Index Documents"
    ↓
chatAPI.startIndexingWithProgress()
    ↓
IndexingProgressPanel appears
    ↓
SSE connection established
    ↓
Real-time progress updates
    ↓
Completion or error handling
```

### 2. Progress State Management

```
useIndexingProgress hook
    ↓
Manages: progressData, status, error
    ↓
Connects to: SSE stream, polling fallback
    ↓
Updates: ProgressBar, StepIndicator, Panel
    ↓
Handles: completion, errors, cancellation
```

## Error Handling & Fallbacks

### 1. Connection Failures
- **SSE unavailable**: Automatic fallback to polling
- **Network timeout**: Exponential backoff retry
- **Server restart**: Graceful reconnection

### 2. Partial Failures
- **Step-level errors**: Continue with remaining steps
- **Batch failures**: Retry individual batches
- **Critical errors**: Clean cancellation with state preservation

### 3. User Experience
- **Cancellation**: Immediate UI feedback, background cleanup
- **Offline mode**: Queue operations for retry
- **Mobile support**: Optimized progress display

## Implementation Phases

### Phase 1: Core Infrastructure
- Backend SSE endpoint
- Enhanced indexing API
- Basic progress tracking

### Phase 2: Frontend Integration
- ChatAPI enhancements
- useIndexingProgress hook
- Basic progress components

### Phase 3: Advanced UI
- Detailed progress panels
- Step-by-step indicators
- Cancellation support

### Phase 4: Polish & Optimization
- Mobile responsiveness
- Performance optimization
- Comprehensive error handling

## Testing Strategy

### Backend Tests
- SSE connection handling
- Progress state management
- Cancellation scenarios
- Error recovery

### Frontend Tests
- Hook state transitions
- Component rendering
- SSE integration
- Error boundaries

### Integration Tests
- End-to-end indexing flow
- Real-time updates
- Network failure scenarios
- Performance under load 