# DSPy Integration Experiments

This directory contains comprehensive DSPy implementations and testing for the RAG system, exploring how DSPy can improve upon the current architecture.

## 🎯 Overview

DSPy (Declarative Self-improving Python) is a framework that replaces manual prompt engineering with automated optimization. This experiment suite tests DSPy's capabilities against our current RAG system across multiple dimensions:

### Current vs DSPy Comparison

| Feature | Current System | DSPy Implementation |
|---------|---------------|-------------------|
| **Prompt Engineering** | Manual prompts in code | Declarative signatures |
| **Optimization** | Manual tuning | Automatic via teleprompters |
| **Query Decomposition** | Rule-based logic | Learned decomposition |
| **Multi-hop Reasoning** | Basic implementation | Advanced hop planning |
| **Verification** | Simple checks | Learned verification |
| **Reranking** | Model-based | Optimized ranking |

## 🏗️ Architecture

```
dspy_experiments/
├── modules/
│   ├── retrievers.py      # Custom retrievers (LanceDB, Hybrid, Graph)
│   ├── rag_modules.py     # RAG implementations (Basic, Advanced, MultiHop)
│   ├── react_modules.py   # DSPy ReAct agents
│   └── signatures.py      # Declarative task signatures
├── pipelines/
│   ├── indexing.py        # DSPy-optimized indexing
│   └── evaluation.py      # Pipeline evaluation
├── optimizers/
│   ├── teleprompters.py   # DSPy optimization strategies
│   └── metrics.py         # Custom evaluation metrics
├── benchmarks/
│   └── comparison.py      # Head-to-head comparisons
├── config.py              # DSPy configuration
└── comprehensive_test.py  # Main testing suite
```

## 🚀 Quick Start

### 1. Basic Setup Test

```bash
# Test DSPy configuration and basic functionality
python -c "from dspy_experiments.config import test_dspy_setup; test_dspy_setup()"
```

### 2. Run Individual Tests

```python
# Test basic RAG
from dspy_experiments.modules.rag_modules import BasicRAG
rag = BasicRAG()
result = rag("What is artificial intelligence?")
print(result.answer)

# Test advanced RAG with all features
from dspy_experiments.modules.rag_modules import AdvancedRAG
advanced_rag = AdvancedRAG(use_verification=True, use_query_decomposition=True)
result = advanced_rag("How do neural networks relate to biological neurons?")
print(f"Answer: {result.answer}")
print(f"Verification: {result.verification}")
```

### 3. Comprehensive Testing

```bash
# Run full comparison between current system and DSPy
python dspy_experiments/comprehensive_test.py
```

## 🧪 Test Modules

### Core RAG Modules

#### BasicRAG
- Simple retrieve-and-generate pipeline
- Direct comparison with current basic RAG
- Uses LanceDB retriever + DSPy generation

#### AdvancedRAG  
- Query decomposition
- Hybrid retrieval (dense + BM25)
- Answer verification
- Reasoning chains

#### MultiHopRAG
- Iterative question refinement
- Multi-step retrieval
- Context expansion detection
- Complex reasoning chains

#### ConversationalRAG
- Conversation history integration
- Context-aware responses
- Session management

#### SelfCorrectingRAG
- Answer verification loops
- Error diagnosis
- Iterative improvement

### Custom Retrievers

#### LanceDBRetriever
- Direct integration with existing LanceDB
- Maintains compatibility with current data
- DSPy-optimized query generation

#### HybridRetriever
- Combines dense + BM25 search
- Reciprocal rank fusion
- Configurable weights

#### GraphEnhancedRetriever
- Knowledge graph integration
- Multi-modal retrieval
- Enhanced context discovery

## 📊 Evaluation Framework

### Metrics

1. **Performance Metrics**
   - Response time
   - Memory usage
   - Throughput

2. **Quality Metrics**
   - Answer relevance
   - Factual accuracy
   - Completeness

3. **Advanced Metrics**
   - Multi-hop reasoning success
   - Verification accuracy
   - Context utilization

### Test Suites

#### Basic Functionality
- Simple Q&A tasks
- Retrieval accuracy
- Generation quality

#### Advanced Features
- Query decomposition effectiveness
- Multi-hop reasoning
- Answer verification

#### Stress Testing
- Large document collections
- Complex queries
- Concurrent usage

## 🔧 Configuration

### DSPy Model Setup

```python
from dspy_experiments.config import configure_dspy_model

# Use Ollama models (default)
configure_dspy_model("generation")  # qwen3:8b

# Use OpenAI for comparison (if API key available)
configure_dspy_model("openai")  # gpt-4o-mini
```

### Retrieval Configuration

```python
# Use existing pipeline configs
from dspy_experiments.modules.rag_modules import AdvancedRAG

rag = AdvancedRAG(
    config_mode="default",  # or "fast"
    num_passages=10,
    use_verification=True,
    use_query_decomposition=True
)
```

## 📈 Expected Benefits

### 1. Automated Optimization
- DSPy teleprompters automatically optimize prompts
- No manual prompt engineering required
- Continuous improvement through feedback

### 2. Better Multi-hop Reasoning
- Learned query decomposition
- Intelligent hop planning
- Context-aware search refinement

### 3. Enhanced Verification
- Automated answer checking
- Learned verification patterns
- Self-correction capabilities

### 4. Improved Modularity
- Declarative task specifications
- Reusable components
- Easier experimentation

## 🎯 Testing Strategy

### Phase 1: Basic Functionality
1. Test DSPy configuration
2. Verify retriever integration
3. Basic RAG functionality

### Phase 2: Advanced Features
1. Query decomposition comparison
2. Multi-hop reasoning evaluation
3. Verification accuracy testing

### Phase 3: Performance Analysis
1. Speed comparisons
2. Memory usage analysis
3. Scalability testing

### Phase 4: Optimization
1. DSPy teleprompter training
2. Parameter optimization
3. Production readiness

## 📝 Results Analysis

### Automated Reports
The comprehensive test generates detailed JSON reports with:
- Success rates by system
- Performance comparisons
- Error analysis
- Feature-specific metrics

### Expected Outcomes
Based on DSPy literature and architecture:

1. **Prompt Quality**: 20-40% improvement in prompt effectiveness
2. **Multi-hop Reasoning**: 30-50% better complex question handling
3. **Verification**: 25-35% improved answer accuracy
4. **Development Speed**: 60-80% faster iteration cycles

## 🚦 Current Status

- ✅ Basic DSPy integration
- ✅ Custom retrievers for LanceDB
- ✅ Core RAG modules implemented
- ✅ Comprehensive testing framework
- 🔄 Optimization experiments (in progress)
- ⏳ Production deployment (pending)

## 🤝 Contributing

To add new DSPy experiments:

1. Create new modules in appropriate directories
2. Add test cases to `comprehensive_test.py`
3. Update this README with new features
4. Run full test suite to ensure compatibility

## 📚 References

- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)
- [DSPy Paper](https://arxiv.org/abs/2310.03714)

---

**Note**: This is an experimental branch for exploring DSPy integration. Results should be thoroughly evaluated before production deployment. 