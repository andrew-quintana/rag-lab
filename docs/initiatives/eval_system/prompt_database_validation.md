# Prompt Database Validation — Database-Based Prompt Versioning

## Overview

This document validates the refactor to database-based prompt versioning and ensures all subsequent phases are scoped to use this approach.

**Date**: 2024-12-19  
**Status**: ✅ Validated

## Refactor Summary

### Changes Made

1. **Removed All Prompt Files from Repository**
   - Deleted `backend/rag_eval/prompts/evaluation/` directory and all evaluation prompt files
   - Removed `_DEFAULT_PROMPT_PATH` fallback from all evaluator classes
   - Prompts are now exclusively stored in `public.prompts` database table

2. **Moved Test Prompts**
   - Moved `prompt_v1.md` and `prompt_v2.md` to `backend/tests/fixtures/prompts/`
   - These are kept for testing purposes only

3. **Updated Evaluator Classes**
   - All evaluators now require either `query_executor` (for database) or `prompt_path` (for testing)
   - Removed default file path fallback logic
   - Updated docstrings to reflect database-first approach

4. **Updated Tests**
   - All tests updated to use mock `query_executor` or explicit `prompt_path` pointing to test fixtures
   - Integration tests updated to query correct database table (`prompts` not `prompt_versions`)
   - Fixed version references (use `0.1` instead of `v1`)

## Database Schema

### Table: `public.prompts`

```sql
CREATE TABLE prompts (
    id UUID PRIMARY KEY,
    version VARCHAR(50) NOT NULL,           -- Semantic version (e.g., '0.1', '0.2')
    prompt_type VARCHAR(50) NOT NULL,       -- 'rag', 'evaluation', etc.
    name VARCHAR(50),                        -- For evaluation prompts (e.g., 'correctness_evaluator')
    prompt_text TEXT NOT NULL,
    live BOOLEAN NOT NULL DEFAULT false,    -- Active version flag
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT prompts_prompt_type_name_version_unique UNIQUE (prompt_type, name, version)
);
```

### Evaluation Prompt Identification

Evaluation prompts are identified by:
- `prompt_type='evaluation'`
- `name` (e.g., 'correctness_evaluator', 'hallucination_evaluator', 'risk_direction_evaluator', 'cost_extraction_evaluator')
- `version` (e.g., '0.1', '0.2')
- `live=true` (for active version)

## Validation Results

### Test Results

**Prompt-Related Tests**: 51/51 passed ✅
- All prompt loading tests pass
- Database integration tests pass
- File-based tests (for testing) pass

**Test Coverage**:
- All evaluator classes properly use database loading
- Tests properly mock database or use test fixtures
- Integration tests verify database connectivity

### Code Validation

1. **Evaluator Classes** ✅
   - `CostExtractionEvaluator`: Uses database via `query_executor`
   - `CorrectnessEvaluator`: Uses database via `query_executor`
   - `HallucinationEvaluator`: Uses database via `query_executor`
   - `RiskDirectionEvaluator`: Uses database via `query_executor`
   - All classes removed `_DEFAULT_PROMPT_PATH` fallback

2. **Base Evaluator** ✅
   - `BaseEvaluatorNode._load_prompt_template()` properly handles:
     - Database loading via `query_executor`
     - File loading via `prompt_path` (for testing only)
     - Error handling when neither is provided

3. **Prompt Loading Function** ✅
   - `load_prompt_template()` in `rag_eval/services/rag/generation.py`:
     - Uses `name` parameter (not `evaluator_type`)
     - Queries `prompts` table (not `prompt_versions`)
     - Supports version and live flag

## Implementation Pattern for Future Phases

### Required Pattern for All New Evaluators

```python
class NewEvaluator(BaseEvaluatorNode):
    """New evaluator using database-based prompts"""
    
    def __init__(
        self,
        prompt_version: Optional[str] = None,
        live: bool = True,
        query_executor: Optional[QueryExecutor] = None,  # REQUIRED for production
        config: Optional[Config] = None,
        llm_provider: Optional[LLMProvider] = None,
        prompt_path: Optional[Path] = None  # For testing only
    ):
        """
        Initialize evaluator.
        
        Args:
            query_executor: QueryExecutor instance (required for database prompt loading)
            prompt_path: Path to prompt template file (optional, for testing only).
                        Either query_executor or prompt_path must be provided.
        """
        super().__init__(
            prompt_version=prompt_version,
            prompt_type="evaluation",
            name="new_evaluator",  # Must match database name
            live=live,
            query_executor=query_executor,
            config=config,
            llm_provider=llm_provider,
            prompt_path=prompt_path
        )
```

### Database Setup Requirements

For each new evaluator, the prompt must be stored in the database:

```sql
INSERT INTO prompts (version, prompt_type, name, prompt_text, live)
VALUES (
    '0.1',                    -- Version
    'evaluation',            -- Prompt type
    'new_evaluator',         -- Name (must match evaluator name)
    '... prompt text ...',   -- Prompt template
    true                     -- Live flag
);
```

### Testing Pattern

```python
def test_evaluator_with_database(self):
    """Test with mock database"""
    mock_query_executor = Mock(spec=QueryExecutor)
    mock_query_executor.execute_query.return_value = [
        {"prompt_text": "# Prompt Template\n\n{placeholder}\n"}
    ]
    evaluator = NewEvaluator(query_executor=mock_query_executor)
    # ... test code ...

def test_evaluator_with_file(self):
    """Test with file path (for testing)"""
    test_prompt_path = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "prompts" / "prompt_v1.md"
    evaluator = NewEvaluator(prompt_path=test_prompt_path)
    # ... test code ...
```

## Phase 6 and Beyond Requirements

### Phase 6: Risk Impact LLM-Node

**Prompt Requirements**:
- Store prompt in database with:
  - `prompt_type='evaluation'`
  - `name='risk_impact_evaluator'`
  - `version='0.1'`
  - `live=true`

**Implementation Requirements**:
- Use `query_executor` parameter (required for production)
- Support `prompt_path` parameter (for testing only)
- Do NOT create prompt file in repository
- Do NOT use `_DEFAULT_PROMPT_PATH` fallback

**Testing Requirements**:
- Use mock `query_executor` for unit tests
- Use test fixture path for file-based tests
- Integration tests verify database connectivity

### All Future Phases

1. **No Prompt Files in Repository**: All prompts must be in database
2. **Database-First Approach**: Evaluators must use `query_executor` for production
3. **Test Fixtures Only**: Use `tests/fixtures/prompts/` for test files if needed
4. **Version Management**: Use semantic versioning in database (`0.1`, `0.2`, etc.)
5. **Live Flag**: Mark active versions with `live=true`

## Migration Checklist

- [x] Removed all evaluation prompt files from repository
- [x] Moved test prompts to `tests/fixtures/prompts/`
- [x] Updated all evaluator classes to remove default path fallback
- [x] Updated all tests to use mock query_executor or test fixtures
- [x] Fixed integration tests to use correct database table and columns
- [x] Updated Phase 5 documentation to reflect database-based prompts
- [x] Updated Phase 6 documentation to reflect database-based prompts
- [x] Validated all prompt-related tests pass
- [x] Verified database schema matches implementation

## Known Issues

None. All tests pass and implementation is validated.

## Next Steps

1. **Phase 6 Implementation**: Follow database-based prompt pattern
2. **Future Phases**: All phases must use database-based prompts
3. **Documentation**: Update any remaining documentation references to file-based prompts

---

**Document Status**: Complete  
**Last Updated**: 2024-12-19  
**Author**: Implementation Agent

