# Stakeholder Engagement Plan

Extracted from: `resources/Challenge 4 form responses.md`

## Project
Optimizing RAG Document Retrieval for Agronomic Advice

## Stakeholder Group 1: African Smallholder Farmers
- **Who they are:** Rural agricultural practitioners across Sub-Saharan Africa managing small plots and relying on crop yields.
- **Why they are important:** They are the primary end-users; their colloquial queries define what the retrieval pipeline must understand.
- **What matters most:** Fast, understandable, accurate interventions for pests, diseases, and climate stress.
- **Potential benefits and risks:**
  - Benefits: Timely interventions that protect crop yields and resilience.
  - Risks: Misdiagnosis leading to incorrect treatment and crop loss.
  - Key consideration: Language gap between formal agronomy terms and local phrasing.
- **Engagement method:** Participatory Design through interactive farmer sessions to map colloquial symptom descriptions to formal factsheets.
- **Influence on design:** Improves handling of vocabulary mismatch in the hybrid BM25 + dense embedding retrieval pipeline.

## Stakeholder Group 2: Agricultural Extension Workers
- **Who they are:** Frontline advisors from government/NGO networks supporting rural farms.
- **Why they are important:** They are likely field operators and define deployment constraints.
- **What matters most:** Fast, reliable, resource-conscious tooling on low-cost CPU hardware.
- **Potential benefits and risks:**
  - Benefits: Faster access to relevant factsheets from the full document corpus.
  - Risks: Latency or retrieval of hard negatives.
- **Engagement method:** User Committees to evaluate retrieval quality and deployment feasibility on CPU-only prototypes.
- **Influence on design:** Guides ranking evaluation, edge-device performance targets, and reranker trade-offs for low-latency use.
