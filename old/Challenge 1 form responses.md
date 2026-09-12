### **Challenge 1 Form Responses**

&nbsp;

**Team Name** \* Upemba

**Are you one of the team leaders?** \* Yes

**Names of Active Team Members** \* Christopher Olajide (Lead), Ezekiel Olaniyi-Omolofi, Chisaneme Aloni, Elizabeth Sanyaolu, Sumayah Adegbite, Oma Anosike Ihuoma

**Selected Project** \* Optimizing RAG Document Retrieval for Agronomic Advice

&nbsp;

**Describe the problem that the selected project is trying to solve.** \*&nbsp;

Existing agricultural information retrieval systems rely heavily on traditional lexical keyword searches and sparse vector models like TF-IDF. These approaches fail to bridge the gap between formal technical agronomic terminology and the informal, localized natural language queries used by farmers in the field, leading to context blindness and irrelevant search results.

&nbsp;

**Who is affected by this problem?** \*&nbsp;

Smallholder farmers and agricultural extension workers, particularly across Sub-Saharan Africa.

&nbsp;

**Why is solving this problem important?** \*&nbsp;

Timely and reliable advice on crop health, soil management, pest control, nutrient deficiencies, and climate adaptation is required to safeguard crop yields and ensure local food security. If unresolved, irrelevant search results and misdiagnoses will continue to hamper critical field interventions and decision-making.

&nbsp;

**Which values should guide your solution?** \*&nbsp;

Inclusivity, Accessibility, Resource-Consciousness, and Agricultural Resilience.

&nbsp;

**Why did you choose these values?** \*&nbsp;

Inclusivity and accessibility democratize expert agronomic knowledge by bridging the terminology gap between complex manuals and everyday queries. Resource-consciousness ensures the retrieval pipeline can be deployed affordably on low-cost hardware or edge devices in rural extension offices without relying on expensive cloud infrastructure.

&nbsp;

**What ethical risks should your team consider?** \*&nbsp;

The primary ethical risk is the potential for the AI to misdiagnose an issue or provide irrelevant advice, such as confusing a pest prevention guide for an active treatment query. Incorrect agronomic interventions directly lead to poor farming practices and subsequent crop failure.

&nbsp;

**How will your team reduce these risks?** \*&nbsp;

We will implement a hybrid retrieval strategy combining sparse lexical retrieval with dense semantic vector representations to capture both exact keywords and high-level contextual intent. Additionally, a lightweight Cross-Encoder re-ranker optimized for the nDCG@5 metric will ensure the most direct factsheets are placed at the top of the search results.

&nbsp;

**What outcomes would define success?** \*&nbsp;

Success is defined by developing a CPU-efficient, semantic document retrieval pipeline that significantly outperforms a sparse TF-IDF baseline on the nDCG@5 metric. Practically, success means the tool can be deployed on edge devices in rural areas to directly support faster, better decision-making against environmental threats.

&nbsp;

**Submit a final values-led problem statement (100–150 words).** \*&nbsp;

### **Upload Your Problem Statement (Save as PDF)**

*Save the following section as a PDF file and upload it to the form.*

&nbsp;

### **Problem Statement**

To develop a CPU-efficient, semantic document retrieval pipeline that significantly outperforms the sparse TF-IDF baseline score on the nDCG@5 metric, returning the top five most relevant agricultural extension documents for African smallholder farmers' natural language queries. Smallholder farmers require timely advice to safeguard yields, but existing systems rely on lexical keyword matching that fails to bridge the gap between technical agronomic terminology and colloquial field queries. This context blindness leads to misdiagnoses and irrelevant results. Our solution integrates sparse lexical retrieval with dense semantic embeddings to accurately capture intent without requiring GPU compute. Guided by inclusivity and resource-conscious AI, this tool democratizes expert knowledge, ensuring deployability on low-cost hardware in rural extension offices to build agricultural resilience against climate variability, soil degradation, and pest outbreaks.