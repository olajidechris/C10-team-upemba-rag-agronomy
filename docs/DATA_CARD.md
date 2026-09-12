# Data Card

Extracted from: `resources/Challenge 2 Form Responses.md`

- **Input vs. Output:** The model's input consists of short, natural language questions from farmers, and the output is a ranked CSV list containing the QueryId and the top-5 predicted DocumentIds.
- **Classification Task:** The dataset utilizes graded relevance rather than simple binary classification. Documents are scored as 3 (Perfect), 2 (Relevant), 1 (Marginal), or 0 (Not relevant/hard negatives).
- **Labelling Strategy:** Ground truth labels are provided in the `qrels_train.csv` file. The system's ranking quality is strictly evaluated using the nDCG@5 metric to ensure the highest quality advice appears first.
- **Data Creation & Provenance:** The 695-document knowledge base was provided by the Kaggle competition hosts. The corpus originates from two sources: open-access CC-BY agricultural research (e.g., CGIAR) and synthetic public-domain documents.
- **Web Collection Ethics:** The dataset adopted ethical sourcing. Included grounded texts maintain adherence to CC-BY licensing by excluding non-commercial or no-derivatives materials and retaining attribution links to original authors.
- **Bias and Representation:** The dataset captures semantic differences between expert agronomic jargon and conversational farmer queries to reduce lexical-system bias.
- **Values and Community Alignment:** The dataset supports lightweight embedding and retrieval models for operation on standard CPU hardware, supporting agricultural resilience in under-resourced farming communities.
