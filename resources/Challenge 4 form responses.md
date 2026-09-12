### **Challenge 4 Form Responses**

&nbsp;

**Full Name** \* Christopher Aziki Olajide

**Team Name** \* Upemba

**Are you one of the team leaders?** \* Yes

**Selected Project** \* Optimizing RAG Document Retrieval for Agronomic Advice

&nbsp;

**Which two stakeholder groups did your team select, and why are they important to your project?** \*&nbsp;

We selected **African Smallholder Farmers** and **Agricultural Extension Workers**. Smallholder farmers are the ultimate beneficiaries whose livelihoods depend on accurate, timely advice to protect their crops from diseases and climate variability. They value accessible, localized language over technical jargon. Agricultural extension workers are the frontline operators who will deploy this tool in the field. They are prioritized because they require a highly accurate, resource-conscious tool that can operate reliably on low-cost, CPU-only hardware in areas with limited infrastructure.

&nbsp;

**Explain how your chosen engagement methods will help your team build a more responsible and inclusive AI system.** \*&nbsp;

By using **Participatory Design** with farmers, we can directly capture the colloquial terminology and local expressions they use (e.g., "yellowing lower leaves in maize") to ensure our semantic retrieval pipeline accurately bridges the gap to formal technical manuals. This increases representation and refines our dataset mapping. Implementing **User Committees** with extension workers allows us to identify deployment risks and test the system's resilience against "hard negatives" (e.g., confusing prevention guides with treatment). This improves model outputs and ensures the system operates safely on edge devices.

&nbsp;

**How do you expect stakeholder engagement to influence the long-term success and community acceptance of your project?** \*&nbsp;

Continuous stakeholder engagement ensures the final AI solution is anchored in the realities of African agriculture rather than purely academic benchmarks. Involving extension workers in the evaluation process builds trust and accountability, as they can verify that the system correctly ranks life-saving factsheets over irrelevant ones. By centering the farmers' natural language in the design process, we foster community ownership and transparency, ensuring the tool genuinely democratizes expert agronomic knowledge rather than creating another technical barrier.&nbsp;

### **Upload Your Stakeholder Engagement Plan (Save as PDF)**

*Save the following section as a PDF file and upload it to the form.*

&nbsp;

### **Stakeholder Engagement Plan: Team Upemba**

**Project:** Optimizing RAG Document Retrieval for Agronomic Advice

#### **1\. Stakeholder Group 1: African Smallholder Farmers**

* **Who they are:** Rural agricultural practitioners across Sub-Saharan Africa who manage small plots of land and rely on crop yields for their livelihood and local food security.  
* **Why they are important:** They are the ultimate end-users whose localized, colloquial queries (e.g., describing symptoms like "yellowing lower leaves") dictate the input variables the retrieval pipeline must understand.  
* **What matters most to them:** Rapid, easily understandable, and highly accurate interventions to save failing crops from pests, diseases, or climate stress.  
* **Potential Benefits & Risks:**  
  1. *Benefits:* Timely interventions that protect crop yields and foster agricultural resilience.  
  2. *Risks:* Misdiagnosis (e.g., the AI confusing a potassium deficiency for a nitrogen deficiency) leading to incorrect treatments and subsequent crop failure.  
  3. *Considerations:* The linguistic gap between formal agricultural terminology and local dialects.  
* **Engagement Method: Participatory Design**

  1. *Why it is appropriate:* It allows us to build the system's language comprehension alongside the people who will actually speak to it.  
  2. *How it will be conducted:* We will hold interactive sessions where farmers describe specific crop diseases in their own words. These colloquial phrases will be mapped directly to the formal technical factsheets in our corpus to test and tune our semantic vector representations.  
* **Influence on Project Design:** This feedback will directly refine our dataset by exposing the hybrid retrieval pipeline (BM25 \+ dense embeddings) to a wider variety of informal query structures, improving the model's ability to handle vocabulary mismatch.

&nbsp;

#### **2\. Stakeholder Group 2: Agricultural Extension Workers**

* **Who they are:** Frontline agricultural advisors and government/NGO agents who travel to rural farms to provide expert guidance and diagnose crop issues.  
* **Why they are important:** They act as the primary operators of the RAG system in the field. Their hardware limitations dictate our technical constraints.  
* **What matters most to them:** Having a fast, reliable, and resource-conscious tool that works on standard low-cost CPU hardware without requiring cloud GPU infrastructure.  
* **Potential Benefits & Risks:**  
  1. *Benefits:* The ability to instantly retrieve the exact expert factsheet needed from a 695-document corpus while standing in a field.  
  2. *Risks:* System failure due to high latency, or the pipeline surfacing "hard negatives" (e.g., retrieving a prevention manual instead of an active treatment guide).  
* **Engagement Method: User Committees**  
  1. *Why it is appropriate:* Extension workers possess the technical baseline to evaluate system accuracy and the field experience to evaluate deployment feasibility.  
  2. *How it will be conducted:* A representative group of extension workers will be given access to the CPU-only retrieval prototype. They will test the system's speed and evaluate the top-5 retrieved documents (nDCG@5) against real-world scenarios they face.  
* **Influence on Project Design:** Feedback from the committee will directly influence our evaluation metrics and deployment decisions. If the cross-encoder re-ranker is too slow on their edge devices, we will adjust the model architecture to strictly prioritize resource-conscious, low-latency execution. Their agronomic expertise will also help us tune the margin penalties for hard negatives.