# 🧠 CV Reviewer (LLM + RAG)

AI-powered CV evaluation system that analyzes resumes using Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to provide more accurate and context-aware candidate assessments.

---

## 🚀 Overview

Manual CV screening is time-consuming, inconsistent, and often keyword-based.  
This project solves that by leveraging **LLMs + RAG pipeline** to perform **semantic evaluation** of resumes, improving the quality and efficiency of candidate screening.

---

## ✨ Features

- 📄 Resume parsing and preprocessing  
- 🧠 Semantic CV evaluation using LLM  
- 🔍 Context-aware retrieval with RAG  
- 📊 Structured output for candidate insights  
- ⚡ Faster and more consistent screening compared to manual review  

---

## 🏗️ Tech Stack

- **Language:** Python  
- **LLM Framework:** LangChain  
- **LLM Runtime:** Ollama / OpenAI API  
- **Vector Database:** FAISS  
- **Database:** PostgreSQL (optional for structured data)  
- **Libraries:** Pandas, NumPy  

---

## 🧩 System Architecture

```text
[ CV Input ]
      ↓
[ Text Extraction / Parsing ]
      ↓
[ Chunking ]
      ↓
[ Embedding Generation ]
      ↓
[ Vector Storage (FAISS) ]
      ↓
[ Retrieval (Top-K relevant chunks) ]
      ↓
[ LLM Evaluation (via LangChain) ]
      ↓
[ Structured Output / Insights ]
```

🔍 How It Works (RAG Pipeline)
Input CV – Resume is uploaded and converted into raw text
Chunking & Embedding – Text is split into chunks and transformed into embeddings
Vector Storage – Embeddings are stored in FAISS for efficient similarity search
Retrieval – Relevant CV sections are retrieved based on query
LLM Evaluation – LLM evaluates candidate using retrieved context (reduces hallucination)
Output Generation – Produces structured evaluation and insights
📊 Example Use Cases
Automated CV screening
Candidate skill evaluation
Resume-job matching (extendable)
HR decision support systems
📈 Impact
⏱️ Reduces manual screening effort
🧠 Improves semantic understanding beyond keyword matching
📊 Enables more consistent candidate evaluation
▶️ How to Run
git clone https://github.com/EmilioYanvrent/CV-Reviewer.git
cd CV-Reviewer
pip install -r requirements.txt
python app.py
📌 Future Improvements
Integration with job description matching
UI dashboard for recruiters
Advanced ranking/scoring system
Deployment as API service
👨‍💻 Author

Emilio Yanvrent
AI Engineer | Data Scientist | LLM Systems

LinkedIn: https://linkedin.com/in/emilioyanvrent
GitHub: https://github.com/EmilioYanvrent
⭐ Notes

This project demonstrates practical implementation of:

LLM-powered systems
RAG architecture
Real-world AI application in HR domain
