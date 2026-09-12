# Impact Statement

Extracted from: `resources/Challenge 3 form responses.md`

**Purpose of the project:** Our project, Optimizing RAG Document Retrieval for Agronomic Advice, aims to develop a CPU-efficient, semantic document retrieval pipeline. It addresses a critical problem for smallholder farmers and agricultural extension workers in Africa who struggle to access context-specific advice during field operations. Existing agricultural search tools rely heavily on formal technical keywords, which creates a major barrier for farmers who typically use localized, everyday natural language to describe their crop issues.

**Positive impacts:** By bridging this terminology gap, our solution democratizes access to expert agronomic knowledge. Smallholder farmers and agricultural stakeholders will be able to retrieve highly relevant factsheets covering crop diseases, pests, soil management, and climate adaptation using their everyday language. This direct access supports faster, more informed decision-making to build agricultural resilience and protect local food security.

**Potential risks and harms:** The primary ethical risk is the potential for the system to misdiagnose an issue or provide irrelevant advice, such as confusing a pest prevention guide with an active treatment guide, or mistaking one nutrient deficiency for another. Because farming decisions are high-stakes, incorrect agronomic advice could directly lead to poor farming practices and subsequent crop failure.

**Values:** Our solution is guided by Inclusivity, Accessibility, and Resource-Consciousness. It keeps rural voices included and is designed to operate affordably on low-cost CPU hardware or edge devices in rural extension offices without dependence on expensive cloud GPU infrastructure.

**Mitigation strategies:** To minimize risks, the system uses a hybrid retrieval strategy that combines sparse lexical retrieval with dense semantic vector representations to capture both exact keywords and contextual intent. The pipeline is optimized for rank-sensitive metrics (nDCG@5) so the most accurate documents are consistently ranked at the top.

**Limitations and uncertainties:** The knowledge base incorporates synthetic and LLM-rewritten text that may include simplifications. Retrieved documents are intended for education and benchmarking and should not be used for real farming decisions without expert agronomic verification. The dataset is focused on Sub-Saharan African smallholder agriculture and is not representative of all regions.
