# Testing Conventions Skill - Changelog

## Version 2.0.0 (2026-01-14)

### Major Additions

#### Factory-Boy Integration
- Added comprehensive factory-boy patterns for SQLAlchemy/SQLModel integration tests
- Created two-factory pattern: DB factories (SQLAlchemyModelFactory) vs DTO factories (factory.Factory)
- Added decision trees for when to use `.create()` vs `.build()`
- Documented session management and configuration patterns

#### Repository Testing Patterns
- Added repository integration test patterns (Create, Read, Update, Delete, Query)
- Documented what to test vs what not to test in repository tests
- Created test organization guidelines (one class per repository method)
- Added edge case testing patterns (None returns, empty results, filtering)

#### Quick Reference Decision Trees
- "Which Factory to Use?" - Decision tree for selecting factory type
- "When to Call session.flush()?" - Guidance on session management
- "Factory Method Choice" - Table mapping scenarios to factory methods

#### New Reference Documents
1. **`references/factory-boy.md`** - Comprehensive factory-boy guide
   - Factory class anatomy
   - Session management
   - Common patterns (Sequence, Faker, Iterator, LazyAttribute)
   - Complex models (foreign keys, ENUMs, JSON)
   - Troubleshooting guide

2. **`references/repository-tests.md`** - Repository testing guide
   - Testing philosophy (what to test vs what not to test)
   - Test organization patterns
   - Factory selection decision trees
   - CRUD operation patterns
   - Query testing patterns
   - Common pitfalls

#### Updated Workflows
- Added "Workflow 6: Creating Repository Integration Tests with Factory-Boy"
- Step-by-step guide from domain model discovery to running tests
- Real-world example with call_classification domain
- Common mistakes to avoid section

#### Auto-Generation Templates
- Created `templates/factories_template.py` for generating factories.py files
- Created `templates/conftest_template.py` for generating conftest.py files
- Added field generation rules documentation in templates

### Updates to Existing Content

#### SKILL.md Enhancements
- Added "Factory-Boy Patterns for Repositories" section
- Added "Repository Integration Test Patterns" section (3 patterns)
- Added "Quick Decision Trees" section (3 decision trees)
- Added "Auto-Generating Repository Tests" section
- Updated "Detailed Patterns" section with new reference links

### Key Features

1. **Automatic Factory Generation**
   - Discovers domain models automatically
   - Generates both DB and DTO factories
   - Handles various field types (strings, ENUMs, sequences, etc.)

2. **Automatic Conftest Configuration**
   - Generates autouse fixture for session configuration
   - Imports and configures all DB factories automatically

3. **Clear Factory Selection Guidance**
   - Decision trees make factory choice obvious
   - Examples for every scenario
   - Common pitfalls documented

4. **Comprehensive Documentation**
   - 200+ lines of factory-boy reference documentation
   - 500+ lines of repository testing patterns
   - Real-world workflows and examples

### Breaking Changes

None - This is a backward-compatible addition to the skill.

### Migration Guide for Existing Users

If you have existing repository tests without factories:

1. **Read the new documentation**:
   - Start with SKILL.md for quick reference
   - Read `references/factory-boy.md` for factory patterns
   - Read `references/repository-tests.md` for test patterns

2. **Create factories.py**:
   - Use the pattern from Workflow 6 in `references/workflows.md`
   - Or ask: "Generate factories for <domain> domain"

3. **Create conftest.py**:
   - Add autouse fixture to configure factories
   - Use the template from `templates/conftest_template.py`

4. **Update existing tests**:
   - Replace manual object creation with factory calls
   - Add `session.flush()` after `DBFactory.create()`
   - Use DTO factories for create tests, DB factories for get tests

### Files Changed/Added

**Modified:**
- `.claude/skills/testing-conventions.skill/SKILL.md` - Added factory-boy and repository testing sections
- `.claude/skills/testing-conventions.skill/references/workflows.md` - Added Workflow 6

**Added:**
- `.claude/skills/testing-conventions.skill/references/factory-boy.md`
- `.claude/skills/testing-conventions.skill/references/repository-tests.md`
- `.claude/skills/testing-conventions.skill/templates/factories_template.py`
- `.claude/skills/testing-conventions.skill/templates/conftest_template.py`
- `.claude/skills/testing-conventions.skill/CHANGELOG.md`

### Verification

Tested with:
- call_acknowledge domain (existing tests refactored)
- call_classification domain (documented in Workflow 6)

All patterns match actual working test code in:
- `src/tests/integration/domains/call_acknowledge/infrastructure/repositories/`

### Next Steps for Users

1. **For new domains**: Ask skill to generate tests - everything will be created automatically
2. **For existing tests**: Gradually migrate to factory pattern using the workflows
3. **When debugging**: Check the troubleshooting section in `references/factory-boy.md`

### Known Limitations

- Auto-generation works best with standard SQLModel/SQLAlchemy models
- Complex relationship configurations may need manual adjustment
- Enum types must be imported manually in generated factories (documented in template)

### Contributors

- Enhanced by Claude Sonnet 4.5 based on transcriptions-system project patterns
- Patterns derived from FastPaip testing conventions
