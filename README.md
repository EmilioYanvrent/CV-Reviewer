## 🧾 CV Reviewer AI — Intelligent Resume Evaluation System

**CV Reviewer AI** is an automated system designed to evaluate candidate resumes by combining **Large Language Model (LLM) reasoning** and **data-driven scoring**. It assists HR teams in screening and ranking applicants efficiently based on job-specific qualifications and skills.

### 💡 Key Features

* **Automated Resume Extraction:**
  Utilizes `pdfplumber` to extract text from multiple CVs in PDF format.

* **AI-Powered Qualification Analysis:**
  Employs **DeepSeek-R1 Distill Qwen-7B**, a transformer-based LLM, to evaluate how well each CV aligns with specific job qualifications stored in a MongoDB database.

* **Skill Matching Engine:**
  Performs keyword-based skill matching to quantify the overlap between candidate skills and job requirements.

* **Dynamic Scoring System:**
  Generates two core scores — *Skills Match Score* and *Qualifications Match Score* — then provides an overall *Suitability Assessment*.

* **Structured Candidate Report:**
  Produces a detailed report including:

  1. **Summary:** Candidate background overview
  2. **Strengths:** Highlighted capabilities
  3. **Weaknesses:** Areas for improvement
  4. **Skills:** Extracted relevant competencies

* **Database Integration:**
  Connects to MongoDB for managing job listings, participant data, and saving or updating AI analysis results automatically.

* **Batch Processing Mode:**
  Supports analyzing multiple CVs in a folder, allowing mass candidate evaluation in one run.

---

### ⚙️ Tech Stack

* **Language:** Python
* **AI Model:** DeepSeek-R1 Distill Qwen-7B
* **Libraries:** Transformers, pdfplumber, PyMongo, Torch, Tkinter
* **Database:** MongoDB
* **Environment:** Local LLM deployment with GPU offloading support

---

### 🧠 Workflow Overview

1. Select a folder containing CVs (`.pdf` files).
2. Choose a target job title from the MongoDB database.
3. For each CV:

   * Extract text content.
   * Evaluate skills and qualifications via the LLM.
   * Compute match scores and suitability.
   * Save the structured analysis to MongoDB.

---

### 📈 Outcome

The system provides a **consistent, explainable, and automated** way to evaluate resumes — reducing manual screening time while maintaining analytical depth. It bridges AI research with HR process automation through **LLM-based reasoning** and **data integration**.
