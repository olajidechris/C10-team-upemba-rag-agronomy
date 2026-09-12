### **Challenge 2 Form Responses**

&nbsp;

**Team Name:** Upemba

**Selected Project:** Optimizing RAG Document Retrieval for Agronomic Advice

&nbsp;

**Dataset Summary**&nbsp;

This dataset of agricultural extension documents and relevance metrics is designed to train, evaluate and ground a document-retrieval pipeline that takes a smallholder farmer's natural language question as input and outputs the top-5 most relevant agricultural extension documents. Our research showed the corpus originates from two sources: actual documents derived from open-access materials hosted on CGSpace (CGIAR), and synthetic documents created to emulate knowledge from official publishing organizations. The knowledge base consists of 695 documents covering topics such as crop diseases, soil, and climate. It includes a set of train queries with relevance labels and test queries for evaluation. This dataset directly supports our project by providing the necessary ground truth to build a CPU-efficient, semantic retrieval pipeline.

&nbsp;

**Ethics and Representation**&nbsp;

We checked and verified that the dataset adopted ethical sourcing by utilizing open-access materials published under CC-BY and CC0 (public domain) licenses. The original works are appropriately credited with source links, respecting the intellectual property of the originating agricultural organizations. To improve fairness and representation, the dataset intentionally targets the vocabulary mismatch between formal technical terminology and the informal, localized language used by farmers, ensuring rural voices aren't excluded by strict lexical search systems. Furthermore, the inclusion of "hard negatives" penalizes systems that confuse intent, protecting users from dangerous misdiagnoses.

&nbsp;

**Values and Community Impact**&nbsp;

The key values guiding our dataset design are Inclusivity, Accessibility, and Resource-Consciousness. Because the dataset and subsequent model are optimized for a CPU-only environment, they directly support our mission of resource-conscious AI. This ensures the solution remains affordable and capable of running on low-cost hardware in rural extension offices without relying on expensive cloud infrastructure. By accurately bridging the terminology gap, we are accountable to the community's need for reliable, timely decision-making against climate variability and pest outbreaks.

### 

###  

### **Upload Your Data Card Content (PDF)**

*Save the following section as a PDF and upload it to the form.*

### 

### **Data Card**

* **Input vs. Output:** The model's input consists of short, natural language questions from farmers, and the output is a ranked CSV list containing the QueryId and the top-5 predicted DocumentIds.  
* **Classification Task:** The dataset utilizes graded relevance rather than simple binary classification. Documents are scored as 3 (Perfect), 2 (Relevant), 1 (Marginal), or 0 (Not relevant/hard negatives).  
* **Labelling Strategy:** Ground truth labels are provided in the qrels\_train.csv file. The system's ranking quality is strictly evaluated using the nDCG@5 metric to ensure the highest quality advice appears first.  
* **Data Creation & Provenance:** The 695-document knowledge base was provided by the Kaggle competition hosts. Our research showed the corpus originates from two sources: open-access CC-BY agricultural research (e.g., CGIAR) and synthetic public-domain documents.  
* **Web Collection Ethics:** We checked and verified that the dataset adopted ethical sourcing. All included "grounded" texts maintain strict adherence to CC-BY licensing by excluding non-commercial or no-derivatives materials and retaining attribution links to the original authors.  
* **Bias and Representation:** The dataset specifically captures the semantic differences between expert agronomic jargon and conversational farmer queries. This design choice mitigates the bias of traditional lexical systems that only serve users who know exact scientific terms.  
* **Values and Community Alignment:** Aligning with our core value of Resource-Consciousness, the dataset allows for the training of lightweight embedding and retrieval models. This guarantees the resulting application can operate offline on standard CPU hardware, directly supporting agricultural resilience for under-resourced farming communities.