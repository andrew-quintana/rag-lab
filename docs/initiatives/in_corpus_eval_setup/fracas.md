# FRACAS.md - Failure Reporting, Analysis, and Corrective Actions System

**Initiative:** In-Corpus Evaluation Dataset Generation System  
**Status:** Active  
**Date Started:** 2025-01-XX  
**Last Updated:** 2025-01-XX  
**Maintainer:** Development Team

## 📋 **How to Use This Document**

This document serves as a comprehensive failure tracking system for the In-Corpus Evaluation Dataset Generation System. Use it to:

1. **Document new failures** as they occur during development/testing
2. **Track investigation progress** and findings
3. **Record root cause analysis** and solutions
4. **Maintain a knowledge base** of known issues and fixes

### **Documentation Guidelines:**
- **Be specific** about symptoms, timing, and context
- **Include evidence** (logs, error messages, screenshots)
- **Update status** as investigation progresses
- **Link related failures** when applicable
- **Record both successful and failed solutions**

---

## 🚨 **Active Failure Modes**

*No active failures at this time. This section will be updated as issues are discovered during development.*

---

## 🔧 **Resolved Failure Modes**

*No resolved failures at this time. This section will be updated as issues are fixed.*

---

## 📝 **New Failure Documentation Template**

Use this template when documenting new failures:

```markdown
### **FM-XXX: [Failure Name]**
- **Severity**: [Low/Medium/High/Critical]
- **Status**: [🔍 Under Investigation | ⚠️ Known issue | 🔧 Fix in progress]
- **First Observed**: [YYYY-MM-DD]
- **Last Updated**: [YYYY-MM-DD]

**Symptoms:**
- [Specific error messages or behaviors]
- [When the failure occurs]
- [What functionality is affected]

**Observations:**
- [What you noticed during testing]
- [Patterns or timing of the failure]
- [Any error messages or logs]

**Investigation Notes:**
- [Steps taken to investigate]
- [Hypotheses about the cause]
- [Tests performed or attempted]
- [Files or components involved]

**Root Cause:**
[The actual cause once identified, or "Under investigation" if unknown]

**Solution:**
[How the issue was fixed, or "Pending" if not yet resolved]

**Evidence:**
- [Code changes made]
- [Log entries or error messages]
- [Test results or screenshots]

**Related Issues:**
- [Links to related failures or issues]
```

---

## 🧪 **Testing Scenarios**

### **Scenario 1: Query Generator - Basic Query Generation**
- **Steps**: Run query generator with sample chunks from Azure AI Search index
- **Expected**: Generate diverse, in-corpus queries with metadata
- **Current Status**: ⏳ Not yet tested
- **Last Tested**: N/A
- **Known Issues**: None

### **Scenario 2: Dataset Generator - Complete Dataset Creation**
- **Steps**: Load eval_inputs.json and generate complete evaluation dataset
- **Expected**: Dataset includes all required fields (input, retrieval_query, context, prompts, BEIR metrics)
- **Current Status**: ⏳ Not yet tested
- **Last Tested**: N/A
- **Known Issues**: None

### **Scenario 3: Output Generator - RAG Output Generation**
- **Steps**: Load eval_dataset.json and generate RAG outputs
- **Expected**: All entries have generated outputs using RAG pipeline
- **Current Status**: ⏳ Not yet tested
- **Last Tested**: N/A
- **Known Issues**: None

---

## 🔍 **Failure Tracking Guidelines**

### **When to Document a Failure:**
- Any unexpected behavior or error during development/testing
- Performance issues or slow responses
- Service unavailability or crashes
- Data inconsistencies or corruption
- Security concerns or vulnerabilities

### **What to Include:**
1. **Immediate Documentation**: Record symptoms and context as soon as possible
2. **Evidence Collection**: Screenshots, logs, error messages, stack traces
3. **Reproduction Steps**: Detailed steps to reproduce the issue
4. **Environment Details**: OS, browser, service versions, configuration
5. **Impact Assessment**: What functionality is affected and severity

### **Investigation Process:**
1. **Initial Assessment**: Determine severity and impact
2. **Data Gathering**: Collect logs, error messages, and context
3. **Hypothesis Formation**: Develop theories about the root cause
4. **Testing**: Attempt to reproduce and isolate the issue
5. **Root Cause Analysis**: Identify the actual cause
6. **Solution Development**: Implement and test fixes
7. **Documentation**: Update the failure record with findings

### **Status Updates:**
- **🔍 Under Investigation**: Issue is being analyzed and tested
- **⚠️ Known issue**: Issue understood, workaround available
- **🔧 Fix in progress**: Solution being implemented
- **✅ Fixed**: Issue has been resolved and verified
- **Won't Fix**: Issue is known but not planned to be addressed

## 📈 **System Health Metrics**

### **Current Performance:**
- Phase 0 (Context Harvest): ✅ Complete
- Phase 1 (Query Generator): ⏳ Pending
- Phase 2 (Dataset Generator): ⏳ Pending
- Phase 3 (Output Generator): ⏳ Pending
- Phase 4 (System Prompt Management): ⏳ Pending
- Phase 5 (Terminology Update): ⏳ Pending

### **Known Limitations:**
- System is in early development phase
- No production deployment yet
- Testing infrastructure being validated

## 🔍 **Investigation Areas**

### **High Priority:**
1. Azure AI Search index accessibility and query performance
2. Supabase prompt retrieval and system prompt management
3. RAG pipeline integration and output generation

### **Medium Priority:**
1. BEIR metrics computation accuracy
2. Query generation diversity and quality
3. Error handling and retry logic

### **Low Priority:**
1. Performance optimization
2. Logging and observability improvements
3. Documentation updates

## 📝 **Testing Notes**

### **Recent Tests (2025-01-XX):**
- ✅ Virtual environment validated (Python 3.13.7)
- ✅ pytest installed and can discover tests
- ✅ Backend dependencies installed
- ⏳ Azure AI Search validation pending
- ⏳ Supabase validation pending

### **Next Test Session:**
- [ ] Validate Azure AI Search index exists and is accessible
- [ ] Validate Azure AI Search credentials are configured
- [ ] Validate Supabase database schema and connection
- [ ] Validate prompts table structure supports system prompts
- [ ] Validate RAG pipeline components are accessible
- [ ] Validate evaluation system components are accessible

---

**Last Updated**: 2025-01-XX  
**Next Review**: After Phase 1 completion  
**Maintainer**: Development Team

