# Documentation Directory

**Last Updated**: 2025-11-12
**Organization**: Topic-based structure

---

## Overview

All 01_pricing documentation organized by topic for easy navigation. This replaces the previous flat structure with 17+ markdown files in the root directory.

---

## Directory Structure

```
docs/
├── architecture/      # System design and data structures
├── optimization/      # Hyperparameter tuning and optimization
├── methodology/       # Research methodology and validation
├── usage/             # How-to guides and quick references
├── visualization/     # Plotting and visualization guides
└── misc/              # Additional documentation
```

---

## Architecture (architecture/)

System design, data structures, and implementation details.

| Document | Purpose |
|----------|---------|
| `MULTI_HORIZON_IMPLEMENTATION.md` | Multi-horizon architecture design |
| `DATA_SPLIT_STRATEGY.md` | Temporal and cross-validation strategy |

**Best for**: Understanding system architecture and data flow

---

## Optimization (optimization/)

Hyperparameter tuning, search strategies, and optimization roadmaps.

| Document | Purpose |
|----------|---------|
| `GRID_SEARCH_README.md` | Grid search methodology and usage |
| `OPTUNA_OPTIMIZATION_EXPLAINED.md` | Bayesian optimization with Optuna |
| `OPTIMIZATION_ROADMAP.md` | Hyperparameter tuning roadmap |

**Best for**: Tuning models and improving performance

---

## Methodology (methodology/)

Research methodology, validation strategies, and implementation summaries.

| Document | Purpose |
|----------|---------|
| `DATA_LEAKAGE_FIXES.md` | Preventing look-ahead bias |
| `WALK_FORWARD_VALIDATION.md` | Walk-forward CV implementation |
| `IMPLEMENTATION_SUMMARY.md` | Overall methodology summary |

**Best for**: Understanding research rigor and validation approaches

---

## Usage (usage/)

How-to guides, quick references, and operational documentation.

| Document | Purpose |
|----------|---------|
| `PIPELINE_USAGE.md` | How to run the pipeline |
| `CPU_OPTIMIZATION_QUICK_REFERENCE.md` | CPU optimization guide (32 vCPUs) |
| `FUNCTIONALITY_COMPARISON.md` | Feature and function comparisons |

**Best for**: Running pipelines and optimizing performance

---

## Visualization (visualization/)

Plotting standards, visualization strategies, and quick guides.

| Document | Purpose |
|----------|---------|
| `PLOT_GUIDE.md` | Comprehensive plotting standards and examples |
| `VISUALIZATION_PLAN.md` | Visualization strategy and roadmap |
| `VISUALIZATION_QUICKSTART.md` | Quick plotting guide |

**Best for**: Creating publication-quality plots and visualizations

---

## Miscellaneous (misc/)

Additional documentation that doesn't fit into other categories.

| Document | Purpose |
|----------|---------|
| `CPU_OPTIMIZATION_32vCPU.md` | Detailed 32 vCPU optimization analysis |
| `V3_MIGRATION_SUMMARY.md` | V3 to V4 migration notes |
| `WANDB_COMPARISON.md` | Weights & Biases tracking comparison |

**Best for**: Migration notes, platform-specific optimizations

---

## Navigation Guide

### "I want to..."

| Goal | Documentation |
|------|---------------|
| **Understand the V4 architecture** | `architecture/MULTI_HORIZON_IMPLEMENTATION.md` |
| **Run the pipeline** | `usage/PIPELINE_USAGE.md` |
| **Tune hyperparameters** | `optimization/OPTUNA_OPTIMIZATION_EXPLAINED.md` |
| **Prevent data leakage** | `methodology/DATA_LEAKAGE_FIXES.md` |
| **Create plots** | `visualization/VISUALIZATION_QUICKSTART.md` |
| **Optimize CPU usage** | `usage/CPU_OPTIMIZATION_QUICK_REFERENCE.md` |
| **Migrate from V3 to V4** | `misc/V3_MIGRATION_SUMMARY.md` |

---

## Documentation Standards

### File Naming

```
TOPIC_NAME_DESCRIPTION.md
```

**Examples:**
- `MULTI_HORIZON_IMPLEMENTATION.md`
- `WALK_FORWARD_VALIDATION.md`
- `PLOT_GUIDE.md`

### Content Structure

All documentation should include:
1. **Title and metadata** (last updated, version)
2. **Overview** (1-2 paragraphs)
3. **Sections** (organized with headers)
4. **Code examples** (when applicable)
5. **References** (links to related docs)

### Markdown Standards

- Use `#` for main title (once per document)
- Use `##` for major sections
- Use `###` for subsections
- Use code blocks with language tags: \`\`\`python
- Use tables for structured data
- Include TOC for documents >100 lines

---

## Updating Documentation

### Adding New Documentation

1. Determine the appropriate category (architecture, optimization, etc.)
2. Create the file in the correct subdirectory
3. Follow naming conventions
4. Update this README.md to list the new document
5. Cross-reference from related documents

### Archiving Old Documentation

Historical documentation should be moved to:
```
/home/ubuntu/Polymarket/research/model/archive/docs_historical/
```

---

## Related Documentation

### Project-Level Documentation
- `/home/ubuntu/Polymarket/research/model/README.md` - Project overview
- `/home/ubuntu/Polymarket/research/model/IMPLEMENTATION_PLAN_V4.md` - V4 technical spec
- `/home/ubuntu/Polymarket/research/model/NEXT_STEPS.md` - Action items

### Code Documentation
- `/home/ubuntu/Polymarket/research/model/01_pricing/README.md` - 01_pricing overview
- `/home/ubuntu/Polymarket/research/model/00_data_prep/README.md` - Feature engineering
- `/home/ubuntu/Polymarket/research/model/02_analysis/README.md` - Analysis scripts

### Historical Documentation
- `/home/ubuntu/Polymarket/research/model/archive/docs_historical/` - V3-era docs

---

## Quick Access

**Most commonly referenced documents:**

1. `usage/PIPELINE_USAGE.md` - How to run pipelines
2. `optimization/OPTUNA_OPTIMIZATION_EXPLAINED.md` - Hyperparameter tuning
3. `visualization/PLOT_GUIDE.md` - Creating visualizations
4. `methodology/DATA_LEAKAGE_FIXES.md` - Validation best practices
5. `architecture/MULTI_HORIZON_IMPLEMENTATION.md` - System architecture

---

**Maintained By**: V4 Development Team
**Last Reorganization**: 2025-11-12 (Topic-based structure created)
