# System Maintenance & Update Guidelines

## 🚨 Critical Rules for RAG System Updates

### **1. Pre-Change Analysis**

#### **1.1 Configuration Mapping**
- [ ] **Map all existing configurations** across files before making changes
- [ ] **Identify dependencies** between configuration files (`main.py`, `config.py`, pipeline files)
- [ ] **Document current model usage** at each stage (embedding, generation, reranking)
- [ ] **Check for hardcoded values** in individual modules that might override configs

#### **1.2 Data Compatibility Assessment**
- [ ] **Identify existing indexed data** and its creation parameters
- [ ] **Check vector dimensions** of existing embeddings before changing embedding models
- [ ] **Verify storage paths** and table naming conventions
- [ ] **Document current data schemas** and formats

#### **1.3 Impact Analysis**
- [ ] **List all affected components** for each proposed change
- [ ] **Identify potential breaking changes** to existing functionality
- [ ] **Map configuration propagation** through the system
- [ ] **Check for cascading effects** of model changes

### **2. Configuration Management**

#### **2.1 Single Source of Truth**
- [ ] **Consolidate configurations** into one authoritative location
- [ ] **Remove duplicate configuration files** that cause conflicts
- [ ] **Ensure all modules import** from the consolidated config
- [ ] **Add validation** to prevent configuration drift

#### **2.2 Backward Compatibility**
- [ ] **Preserve existing data compatibility** when changing models
- [ ] **Support migration paths** for data format changes
- [ ] **Maintain API compatibility** for existing integrations
- [ ] **Document breaking changes** and provide upgrade paths

#### **2.3 Model Compatibility**
- [ ] **Match embedding model dimensions** to existing indexed data
- [ ] **Verify model availability** in target environments (Ollama, HuggingFace)
- [ ] **Check resource requirements** for new models
- [ ] **Test model loading** and initialization

### **3. Change Implementation**

#### **3.1 Targeted Changes**
- [ ] **Identify specific code segments** that need modification
- [ ] **Make minimal necessary changes** to achieve the goal
- [ ] **Avoid broad refactoring** unless specifically needed
- [ ] **Preserve existing working functionality**

#### **3.2 Code Segment Analysis**
```bash
# Before making changes, run:
grep -r "model_name\|embedding_model" rag_system/
grep -r "config\." rag_system/
grep -r "EXTERNAL_MODELS\|OLLAMA_CONFIG" rag_system/
```

#### **3.3 Path Validation**
- [ ] **Verify all file paths** exist and are accessible
- [ ] **Check storage locations** for databases and indexes
- [ ] **Test path resolution** in different environments
- [ ] **Use relative paths consistently**

### **4. Testing & Validation**

#### **4.1 Pre-Deployment Testing**
```bash
# Required test sequence:
python -c "from rag_system.main import get_agent; agent = get_agent('default'); print('✅ Agent initialization successful')"

python -c "
from rag_system.main import get_agent
agent = get_agent('default')
embedder = agent.retrieval_pipeline._get_text_embedder()
test_emb = embedder.create_embeddings(['test'])
print(f'✅ Embedding model: {embedder.model.name_or_path if hasattr(embedder.model, \"name_or_path\") else \"Unknown\"}')
print(f'📊 Embedding dimension: {test_emb.shape[1]}')
"

# Test specific table access
python -c "
from rag_system.main import get_agent
agent = get_agent('default')
# Replace with actual table name
result = agent.run('test query', table_name='actual_table_name')
print('✅ Query execution successful')
"
```

#### **4.2 Component Validation**
- [ ] **Test agent initialization** with new configurations
- [ ] **Verify model loading** for all configured models
- [ ] **Check database connections** and table access
- [ ] **Test embedding generation** and dimension compatibility
- [ ] **Validate retrieval pipeline** end-to-end
- [ ] **Test reranking functionality**

#### **4.3 Integration Testing**
- [ ] **Test complete query workflow** from input to response
- [ ] **Verify source document retrieval**
- [ ] **Check response quality** and relevance
- [ ] **Test error handling** for edge cases
- [ ] **Test UI option propagation** - verify all 16 options reach backend
- [ ] **Test dynamic model selection** from UI dropdowns
- [ ] **Test configuration override** in real-time

### **5. Error Recovery**

#### **5.1 Rollback Procedures**
- [ ] **Document original configurations** before changes
- [ ] **Create configuration backups** 
- [ ] **Test rollback procedures** before deployment
- [ ] **Maintain version history** of working configurations

#### **5.2 Common Issues Checklist**
- [ ] **"Table not found" errors** → Check LanceDB paths and table naming
- [ ] **"Vector dimension mismatch"** → Verify embedding model consistency
- [ ] **"Module not found" errors** → Check import paths and dependencies  
- [ ] **"Model not available" errors** → Verify Ollama model installation

### **6. Documentation Requirements**

#### **6.1 Change Documentation**
- [ ] **Document all changes made** with rationale
- [ ] **Update model usage documentation**
- [ ] **Record configuration consolidation**
- [ ] **Note any breaking changes**

#### **6.2 Validation Documentation**
- [ ] **Record test results** and validation steps
- [ ] **Document configuration validation**
- [ ] **Note performance impacts**
- [ ] **Update troubleshooting guides**

### **7. Validation Commands**

#### **7.1 Configuration Validation**
```bash
# Check for configuration conflicts
find rag_system/ -name "*.py" -exec grep -l "config\|CONFIG" {} \;

# Verify model configurations
python -c "
from rag_system.main import EXTERNAL_MODELS, OLLAMA_CONFIG, PIPELINE_CONFIGS
print('External Models:', EXTERNAL_MODELS)
print('Ollama Config:', OLLAMA_CONFIG)  
print('Pipeline Configs:', PIPELINE_CONFIGS)
"
```

#### **7.2 Database Validation**
```bash
# Check LanceDB tables
python -c "
import lancedb
db = lancedb.connect('./lancedb')
tables = db.table_names()
print(f'Available tables: {len(tables)}')
for table in tables[:5]:
    print(f'  - {table}')
"
```

#### **7.3 Model Validation**
```bash
# Test all configured models
python validate_model_config.py  # If validation script exists
```

## **8. Emergency Procedures**

### **8.1 System Down Scenarios**
1. **Revert to last known working configuration**
2. **Check error logs for specific failure points**
3. **Validate individual components** (database, models, paths)
4. **Test minimal configuration** before full restoration

### **8.2 Quick Diagnostics**
```bash
# Quick system health check
python -c "
try:
    from rag_system.main import get_agent
    agent = get_agent('default')
    print('✅ System operational')
except Exception as e:
    print(f'❌ System error: {e}')
"
```

---

## **9. Checklist Template for Updates**

### **Before Making Changes:**
- [ ] Analyzed existing configurations
- [ ] Identified all affected components  
- [ ] Checked data compatibility requirements
- [ ] Documented current working state
- [ ] Planned rollback procedures

### **During Implementation:**
- [ ] Made targeted minimal changes
- [ ] Preserved backward compatibility
- [ ] Updated imports and references
- [ ] Maintained single source of truth

### **After Changes:**
- [ ] Tested agent initialization
- [ ] Validated model loading
- [ ] Checked database connectivity
- [ ] Tested end-to-end queries
- [ ] Documented changes made
- [ ] Updated troubleshooting guides

---

**Remember: When in doubt, test more thoroughly rather than assuming changes work!** 