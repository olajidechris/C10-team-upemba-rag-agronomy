### **Challenge 3 Form Responses**

&nbsp;

**Full Name** \* Christopher Olajide

**Team Name** \* Upemba

**Are you one of the team leaders?** \* Yes

**Selected Project** \* Optimizing RAG Document Retrieval for Agronomic Advice

### 

**Describe the most significant positive impact your project could have on your intended community.** \*&nbsp;

The most significant positive impact is democratizing access to expert agronomic knowledge for smallholder farmers and agricultural extension workers in Africa. They will benefit by being able to retrieve highly relevant factsheets regarding crop diseases, pests, and soil management using their localized, everyday natural language rather than strict technical keywords. This impact is vitally important because it bridges a critical terminology gap, enabling faster and more informed decision-making to protect crop yields, build agricultural resilience, and ensure local food security against environmental threats.

&nbsp;

**Identify the most significant risk associated with your project and explain how your team plans to minimise or manage it.** \*&nbsp;

The most significant risk is the potential for the AI retrieval system to provide irrelevant advice or misdiagnose an issue, such as confusing a pest prevention guide with a treatment guide or supplying factsheets for the wrong crop. This could result in incorrect agronomic interventions and subsequent crop failure. Our team plans to manage this by combining exact keyword precision with contextual intent to better capture the farmer's specific needs. Furthermore, we will heavily optimize the pipeline using rank-sensitive metrics (nDCG@5) to ensure the safest, most direct factsheets are always ranked at the top.

&nbsp;

**How has completing the Problem Statement, Data Card, and Impact Statement influenced the way your team thinks about developing responsible AI?** \*&nbsp;

Completing these frameworks has shifted our focus from purely technical model performance to prioritizing the human problem and the ethical realities of our end users. It highlighted the necessity of designing an inclusive system that accommodates colloquial language, while ensuring the underlying dataset accurately represents rural contexts and penalizes dangerous "hard negatives". Most importantly, it reinforced our commitment to resource-conscious AI, teaching us that an ethical AI system must be practically deployable on low-cost CPU edge devices if it is to genuinely benefit resource-constrained communities.

&nbsp;

&nbsp;

&nbsp;

### 

### **Upload Your Impact Statement (PDF)**

*Save the following section as a PDF and upload it to the form. It meets the 300–500 word requirement (approx. 330 words).*

**Impact Statement**

**Purpose of the project:** Our project, Optimizing RAG Document Retrieval for Agronomic Advice, aims to develop a CPU-efficient, semantic document retrieval pipeline. It addresses a critical problem for smallholder farmers and agricultural extension workers in Africa who struggle to access context-specific advice during field operations. Existing agricultural search tools rely heavily on formal technical keywords, which creates a major barrier for farmers who typically use localized, everyday natural language to describe their crop issues.

**Positive impacts:** By bridging this terminology gap, our solution democratizes access to expert agronomic knowledge. Smallholder farmers and agricultural stakeholders will be able to retrieve highly relevant factsheets covering crop diseases, pests, soil management, and climate adaptation using their everyday language. This direct access supports faster, more informed decision-making to build agricultural resilience and protect local food security.

**Potential risks and harms:** The primary ethical risk is the potential for the system to misdiagnose an issue or provide irrelevant advice, such as confusing a pest prevention guide with an active treatment guide, or mistaking one nutrient deficiency for another. Because farming decisions are high-stakes, incorrect agronomic advice could directly lead to poor farming practices and subsequent crop failure.

**Values:** Our solution is heavily guided by the core values of Inclusivity, Accessibility, and Resource-Consciousness. We champion inclusivity by ensuring rural voices are not excluded by strict lexical search systems. We prioritize resource-consciousness by deliberately ensuring the retrieval pipeline operates affordably on low-cost CPU hardware or edge devices in rural extension offices, entirely avoiding reliance on expensive cloud GPU infrastructure.

**Mitigation strategies:** To minimize risks, we are implementing a hybrid retrieval strategy that combines sparse lexical retrieval with dense semantic vector representations to capture both exact keywords and contextual intent. We will strictly optimize the system for rank-sensitive metrics (nDCG@5) to ensure the most accurate documents are consistently placed at the top of the search results.

**Limitations and uncertainties:** We acknowledge that our knowledge base incorporates synthetic and LLM-rewritten text which may contain simplifications. Therefore, the retrieved documents are intended for education and benchmarking, and must not be used for real farming decisions without expert agronomic verification. Additionally, the dataset is strictly focused on Sub-Saharan African smallholder agriculture and is not representative of other regions or farming systems.

### 