# RAGLab AI Agent Guidelines

This document provides context, conventions, and requirements for AI coding agents working on RAGLab, a lean RAG evaluation framework.

## Project Overview

RAGLab is a notebook-driven RAG evaluation framework that provides comprehensive evaluation of retrieval-augmented generation systems using local file storage and FAISS indexing. Key components:

- **LLM-as-Judge** evaluation with insurance risk semantics  
- **BEIR-style retrieval metrics** (Recall@K, Precision@K, nDCG@K)
- **Meta-evaluation** for judge reliability assessment
- **Component registry** for comparative evaluation of multiple implementations
- **Local-first architecture** with no cloud dependencies

## Dev Environment Tips

- **Python Environment**: Project uses Python with dependencies managed via pip
- **Key Dependencies**: `faiss-cpu pandas numpy matplotlib seaborn pyyaml`
- **Optional Providers**: `openai anthropic sentence-transformers` for specific implementations
- **Project Structure**: Follows modular design with `src/core/`, `src/evaluation/`, `src/indexing/` 
- **Component Registry**: Use `@register_component` decorators for new implementations
- **Notebooks**: Interactive workflow in `notebooks/` (00-04 sequence)
- **Configuration**: Store artifacts in `artifacts/`, results in `runs/`, data in `data/`

## Code Conventions

- **Imports**: Use relative imports within src/ (e.g., `from ..core.interfaces import`)
- **Type Hints**: Include comprehensive type annotations for all functions
- **Documentation**: Add docstrings to all classes and public methods
- **Error Handling**: Implement graceful error handling with meaningful messages
- **Security**: Never expose or log secrets/keys; follow security best practices
- **Comments**: Avoid code comments unless explicitly requested by user
- **File Organization**: Prefer editing existing files over creating new ones

## Component Development

- **Base Classes**: Inherit from appropriate base class in `src/evaluation/base/`
- **Registry**: Use `@register_[component_type]` decorators with descriptive metadata
- **Interface Compliance**: Implement all abstract methods from base classes
- **Configuration**: Accept configuration via `**config` parameter in `__init__`
- **Naming**: Use descriptive names that indicate implementation approach
- **Testing**: Test new components with small datasets before full evaluation

## Testing Instructions

- **Unit Tests**: No specific test framework specified - check README or search codebase
- **Component Testing**: Use small evaluation datasets to validate implementations
- **Integration Testing**: Run complete evaluation pipeline with new components
- **Performance Testing**: Benchmark component performance with timing measurements
- **Validation**: Verify components implement required abstract methods
- **Error Testing**: Test error handling with invalid inputs

## Evaluation Standards

- **BEIR Metrics**: Ensure retrieval components support standard BEIR evaluation
- **Judge Evaluation**: LLM-as-Judge must support 4-stage evaluation (correctness, hallucination, risk direction, risk impact)
- **Insurance Risk Semantics**: Maintain care avoidance vs unexpected cost analysis
- **Meta-Evaluation**: Judge outputs should be validated for reliability
- **Reproducibility**: Save exact configurations and component versions used
- **Comparative Analysis**: Support side-by-side evaluation of multiple implementations

## Quality Standards

- **Code Quality**: Write idiomatic, clean code following existing patterns
- **Documentation**: Comprehensive docstrings and type hints required
- **Error Handling**: Robust error handling with informative messages
- **Performance**: Optimize for evaluation pipeline efficiency
- **Modularity**: Components should be isolated and interchangeable
- **Backward Compatibility**: Maintain compatibility with existing interfaces

## Security Constraints

- **Defensive Only**: Only assist with defensive security tasks
- **No Malicious Code**: Refuse to create/modify code for malicious purposes
- **No Credential Harvesting**: Do not assist with bulk credential discovery
- **Safe Analysis**: Allow security analysis, detection rules, vulnerability explanations
- **Documentation**: Permit defensive security documentation and tools

## File Management

- **Read Before Edit**: Always use Read tool before editing files
- **Prefer Editing**: Edit existing files rather than creating new ones
- **Absolute Paths**: Use absolute file paths for tool operations
- **No Proactive Documentation**: Don't create markdown/README files unless requested
- **Import Updates**: Update all import references when reorganizing files

## Evaluation Workflow

- **Pipeline Order**: Setup → Ingest/Index → Retrieval Eval → Agent Eval → Analysis
- **Results Storage**: Each run creates timestamped directory in `runs/`
- **Configuration**: Save evaluation config as `config.yaml` in run directory
- **Outputs**: Store detailed results as `outputs.jsonl`, metrics as `metrics.json`
- **Tracing**: Maintain execution traces in `traces.jsonl`
- **Prompts**: Load evaluation prompts from `prompts/evaluation/` directory

## PR Instructions

- **Branch Naming**: Use descriptive branch names indicating feature/change
- **Commit Messages**: Follow format: "Brief description\n\n🤖 Generated with [Claude Code](https://claude.ai/code)\n\nCo-Authored-By: Claude <noreply@anthropic.com>"
- **Pre-commit**: Run lint and typecheck commands if available (check README)
- **Testing**: Ensure all tests pass before creating PR
- **Documentation**: Update relevant documentation for significant changes
- **Component Registry**: Verify new components are properly registered
- **Import Verification**: Ensure all imports work after structural changes

## Change Management

- **AGENTS.md Updates**: Follow `.agents/rules/agents-md-update-guide.md` for updates
- **Version Control**: Tag component implementations with versions
- **Breaking Changes**: Document breaking changes and migration paths
- **Configuration**: Maintain backward compatibility for configurations
- **Registry**: Avoid conflicting component names in registry

## Reference Links

- **agents.md specification**: https://github.com/agentsmd/agents.md
- **Component Registry Guide**: `COMPONENT_REGISTRY.md`
- **Repository Structure**: `README.md`
- **Agent Rules**: `.agents/rules/` directory