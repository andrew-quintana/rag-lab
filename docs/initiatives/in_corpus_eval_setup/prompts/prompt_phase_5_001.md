# Phase 5 Prompt — Terminology Update: risk_impact → risk_magnitude

## Context

This prompt guides the implementation of **Phase 5: Terminology Update** for the In-Corpus Evaluation Dataset Generation System. This phase updates all `risk_impact` references to `risk_magnitude` throughout the codebase for consistent terminology.

**Related Documents:**
- @docs/initiatives/in_corpus_eval_setup/scoping/PRD001.md - Product requirements and functional specifications
- @docs/initiatives/in_corpus_eval_setup/scoping/RFC001.md - Technical architecture and design decisions
- @docs/initiatives/in_corpus_eval_setup/scoping/TODO001.md - Implementation breakdown (check off tasks as completed)
- @docs/initiatives/in_corpus_eval_setup/scoping/context.md - Project context and scope
- @docs/initiatives/in_corpus_eval_setup/intermediate/phase_4_handoff.md - Phase 4 handoff document

## Objectives

1. **Find All References**: Identify all `risk_impact` references in codebase
2. **Update Interfaces**: Update dataclass fields in interfaces.py
3. **Update Evaluator Modules**: Update all references in evaluator modules
4. **Update Database Schemas**: Update database schemas if applicable
5. **Update Documentation**: Update documentation references
6. **Verify No Regressions**: Ensure all tests still pass

## Execution Requirements

### Checklist Modifications
- **REQUIRED**: Check off completed tasks in @docs/initiatives/in_corpus_eval_setup/scoping/TODO001.md Phase 5 section as you complete them
- Update phase status when all implementation tasks are complete

### Validation
- **REQUIRED**: Run all existing tests to ensure no regressions
- **REQUIRED**: Verify all `risk_impact` references updated
- **REQUIRED**: Verify no broken imports or references
- **REQUIRED**: Document any blockers or issues in fracas.md (root directory)

### Documentation
- **REQUIRED**: Create `intermediate/phase_5_decisions.md` if any decisions are made that aren't in PRD/RFC
- **REQUIRED**: Create `intermediate/phase_5_testing.md` documenting test results
- **REQUIRED**: Create `intermediate/phase_5_handoff.md` summarizing completion

## Key References

### Files to Update
- @backend/rag_eval/core/interfaces.py - Dataclass definitions
- @backend/rag_eval/services/evaluator/meta_eval.py - Meta-evaluator module
- @backend/rag_eval/services/evaluator/risk_impact.py - Risk impact module
- All other files with `risk_impact` references (use grep to find)

### Database Schemas
- @infra/supabase/migrations/ - Database migration files that may reference risk_impact

## Phase 5 Tasks

### Implementation
1. Find all `risk_impact` references using grep:
   - Search entire codebase for "risk_impact"
   - Create list of all files that need updating
   - Categorize by type (interfaces, modules, tests, migrations, docs)
2. Update `backend/rag_eval/core/interfaces.py`:
   - `JudgeEvaluationResult.risk_impact` → `risk_magnitude`
   - `MetaEvaluationResult.ground_truth_risk_impact` → `ground_truth_risk_magnitude`
   - `JudgePerformanceMetrics.risk_impact` → `risk_magnitude`
   - Update all docstrings and comments
3. Update `backend/rag_eval/services/evaluator/meta_eval.py`:
   - All variable names containing `risk_impact`
   - All function parameters
   - All references in code
   - All docstrings and comments
4. Update `backend/rag_eval/services/evaluator/risk_impact.py`:
   - Function names if needed (consider if module should be renamed)
   - All variable names
   - All references
   - All docstrings and comments
5. Update all other files with `risk_impact` references:
   - Test files
   - Other service modules
   - API routes
   - Any other files found in grep search
6. Update database schemas if applicable:
   - Check migration files for risk_impact references
   - Create new migration if needed to update column names
   - Consider backward compatibility
7. Update documentation references:
   - Update any documentation that references risk_impact
   - Update docstrings
   - Update comments

### Testing
1. **REQUIRED**: Run all existing tests:
   - Run full test suite
   - Verify all tests pass
   - Fix any broken tests
2. **REQUIRED**: Verify all `risk_impact` references updated:
   - Run grep again to verify no remaining references
   - Check for case variations (risk_impact, RiskImpact, etc.)
3. **REQUIRED**: Verify no broken imports or references:
   - Check for import errors
   - Check for undefined references
   - Verify all modules can be imported
4. **REQUIRED**: Test coverage maintained:
   - Verify test coverage for updated modules
   - Ensure no coverage regressions

## Success Criteria

- [ ] All `risk_impact` references updated to `risk_magnitude`
- [ ] All interfaces updated correctly
- [ ] All evaluator modules updated correctly
- [ ] Database schemas updated (if applicable)
- [ ] Documentation updated
- [ ] All tests passing
- [ ] No regressions introduced
- [ ] No broken imports or references
- [ ] Test coverage maintained
- [ ] Phase 5 tasks completed and checked off in TODO001.md
- [ ] Phase 5 deliverables created in intermediate/ directory

## Completion

After completing Phase 5, the In-Corpus Evaluation Dataset Generation System implementation is complete. All phases have been finished and the system is ready for use.

---
**Document Status**: Draft  
**Last Updated**: 2025-01-XX  
**Author**: Documentation Agent  
**Reviewers**: TBD

