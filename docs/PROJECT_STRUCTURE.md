# Project Structure

_Simplified high-level view of the repository layout._

```
.
├── README.md
├── data/
│   ├── input/
│   │   └── [competition input data files]
│   └── submission/
│       └── submission.csv
├── docs/
│   ├── DATA_CARD.md
│   ├── IMPACT_STATEMENT.md
│   ├── PROBLEM_STATEMENT.md
│   ├── PROJECT_STRUCTURE.md
│   ├── STAKEHOLDER_ENGAGEMENT_PLAN.md
│   ├── SUBMISSION_FORMAT.md
│   └── SUBMISSION.md
├── scripts/
│   ├── all_local.py
│   └── external_reranking_api.py
└── resources/
    └── [archived legacy artifacts]
```

This layout keeps submission-facing materials easy to review while retaining earlier artifacts in `resources/` and a checked-in final payload in `data/submission/`.
