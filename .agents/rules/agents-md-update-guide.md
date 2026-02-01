# AGENTS.md Update Guide

This document defines the workflow and change discipline for maintaining `AGENTS.md` according to the canonical agents.md specification.

## Canonical Reference

The authoritative specification for `AGENTS.md` format and content is maintained at:
**https://github.com/agentsmd/agents.md**

Always consult this reference when making updates to ensure compliance with the standard.

## Update Workflow

### 1. Before Making Changes

- **Review Current State**: Read the existing `AGENTS.md` to understand current structure
- **Consult Specification**: Check https://github.com/agentsmd/agents.md for latest requirements
- **Identify Scope**: Determine if changes are:
  - Content updates (new rules, updated conventions)
  - Structural changes (new sections, reorganization)  
  - Corrections (fixing inaccuracies or outdated information)

### 2. Change Process

1. **Document Rationale**: Clearly explain why the change is needed
2. **Preserve Stability**: Maintain existing section structure when possible
3. **Update Comprehensively**: Ensure related sections remain consistent
4. **Validate Against Spec**: Confirm changes align with agents.md standard
5. **Test Impact**: Consider how changes affect existing agent workflows

### 3. Content Guidelines

- **Be Specific**: Provide actionable, project-specific guidance
- **Stay Current**: Keep information up-to-date with project evolution
- **Avoid Duplication**: Don't repeat rules that belong in per-agent files
- **Use Examples**: Include concrete examples when helpful
- **Maintain Clarity**: Write for both human developers and AI agents

### 4. Review Checklist

Before finalizing updates to `AGENTS.md`:

- [ ] Changes align with https://github.com/agentsmd/agents.md specification
- [ ] Required sections are present and properly structured
- [ ] Content is specific to RAGLab project needs
- [ ] No duplication of rules better suited for `.agents/configs/` or `.agents/rules/`
- [ ] Clear, actionable language throughout
- [ ] Examples and code blocks are accurate
- [ ] Cross-references to other project files are correct
- [ ] Security constraints and quality standards are maintained

## Change Discipline

### Section Stability

Maintain these core sections as defined in the agents.md spec:
- **Project Overview** - High-level context about RAGLab
- **Dev Environment Tips** - Setup and configuration guidance  
- **Code Conventions** - Project-specific coding standards
- **Testing Instructions** - How to validate changes
- **PR Instructions** - Contribution workflow

### Content Ownership

- **AGENTS.md**: Repository-wide rules, conventions, and high-level guidance
- **`.agents/rules/`**: Specific workflow rules and detailed guidelines
- **`.agents/configs/`**: Agent-specific constraints and configurations
- **`.agents/templates/`**: Standardized code and configuration templates

### Rationale Documentation

When making significant changes, document:
- **What changed**: Specific sections or rules modified
- **Why it changed**: Business need, project evolution, or spec compliance
- **Impact**: How it affects existing workflows or agent behavior
- **Migration**: Any steps needed to adapt to changes

### Version Considerations

- **Breaking Changes**: Clearly mark changes that modify existing agent expectations
- **Backward Compatibility**: Preserve compatibility when possible
- **Deprecation**: Provide transition guidance for deprecated practices
- **Evolution**: Allow for natural project growth while maintaining stability

## Common Update Scenarios

### Adding New Component Types
- Update **Component Development** section
- Add registration patterns and examples
- Update **Testing Instructions** for new component validation
- Consider creating component-specific rules in `.agents/rules/`

### Changing Evaluation Standards  
- Update **Evaluation Standards** section
- Modify **Quality Standards** if metrics change
- Update **Testing Instructions** for new validation requirements
- Document migration path in rationale

### Security Policy Changes
- Update **Security Constraints** section
- Ensure alignment with organizational security policies
- Document any new restrictions or clarifications
- Consider impact on existing agent behaviors

### Tool or Framework Changes
- Update **Dev Environment Tips** for new dependencies
- Modify **Code Conventions** for new patterns
- Update **File Management** rules if structure changes
- Refresh examples and commands throughout

## Reference Integration

When updating `AGENTS.md`, consider referencing:
- **Component Registry Guide**: `COMPONENT_REGISTRY.md`
- **Project README**: `README.md` 
- **Agent Rules**: Specific files in `.agents/rules/`
- **External Standards**: https://github.com/agentsmd/agents.md

This ensures `AGENTS.md` serves as a comprehensive entry point while avoiding duplication of detailed rules maintained elsewhere.