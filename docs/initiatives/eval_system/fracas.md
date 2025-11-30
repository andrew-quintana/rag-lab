# FRACAS.md - Failure Reporting, Analysis, and Corrective Actions System

**Initiative:** RAG Evaluation MVP (LLM-as-Judge + BEIR + Meta-Eval)  
**Status:** Active  
**Date Started:** 2024-12-19  
**Last Updated:** 2024-12-19  
**Maintainer:** Development Team

## 📋 **How to Use This Document**

This document serves as a comprehensive failure tracking system for the RAG Evaluation MVP initiative. Use it to:

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

*No active failure modes at this time. This section will be populated as issues are encountered during development.*

---

## 🔧 **Resolved Failure Modes**

*No resolved failure modes at this time. This section will be populated as issues are resolved during development.*

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

### **Scenario 1: Phase 0 Environment Validation**
- **Steps**: 
  1. Activate backend venv: `cd backend && source venv/bin/activate`
  2. Verify pytest installation: `pytest --version`
  3. Verify pytest-cov installation: `pip list | grep pytest-cov`
  4. Verify test discovery: `pytest tests/ --collect-only`
- **Expected**: All commands succeed, pytest discovers tests
- **Current Status**: ✅ Working
- **Last Tested**: 2024-12-19
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
- **Phase 0 Status**: ✅ Complete
- **Environment Setup**: ✅ Validated
- **Test Discovery**: ✅ Working (249 tests discovered)
- **Dependencies**: ✅ Installed

### **Known Limitations:**
- Evaluation system not yet implemented (Phase 1+ pending)
- Existing evaluator components are stubs (need full implementation per RFC001)

## 🔍 **Investigation Areas**

### **High Priority:**
1. None at this time

### **Medium Priority:**
1. None at this time

### **Low Priority:**
1. None at this time

## 📝 **Testing Notes**

### **Recent Tests (2024-12-19):**
- ✅ Virtual environment exists and is accessible
- ✅ pytest 9.0.1 installed and working
- ✅ pytest-cov 7.0.0 installed
- ✅ Test discovery working (249 tests collected)
- ✅ All Phase 0 validation requirements met

### **Next Test Session:**
- [ ] Phase 1: Evaluation Dataset Construction validation
- [ ] Phase 2: Correctness LLM-Node tests
- [ ] Phase 3: Hallucination LLM-Node tests

---

**Last Updated**: 2024-12-19  
**Next Review**: After Phase 1 completion  
**Maintainer**: Development Team

