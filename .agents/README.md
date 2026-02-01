# .agents Directory Structure

This directory contains agent-specific configurations and rules for RAGLab.

## Structure

```
.agents/
├── README.md                     # This file
├── rules/                        # Agent workflow rules and guidelines
│   ├── agents-md-update-guide.md # AGENTS.md maintenance guidelines
│   └── component-dev-guide.md    # Component development rules
├── configs/                      # Agent-specific configurations
│   ├── evaluation-agent.md       # Rules for evaluation pipeline agents
│   └── component-agent.md        # Rules for component development agents
└── templates/                    # Template files for consistent outputs
    ├── component-template.py      # Template for new component implementations
    └── evaluation-config.yaml    # Template for evaluation configurations
```

## Usage

- **Rules**: Path-based reference system for agent guidelines (e.g., `@raglab/.agents/rules/component-dev-guide.md`)
- **Configs**: Agent-specific configurations and constraints
- **Templates**: Standardized templates for consistent code generation

## Path-Based References

Agents can reference specific rules using path notation:
- `@raglab/.agents/rules/agents-md-update-guide.md` - AGENTS.md update workflow
- `@raglab/.agents/configs/evaluation-agent.md` - Evaluation pipeline rules
- `@raglab/.agents/templates/component-template.py` - Component implementation template