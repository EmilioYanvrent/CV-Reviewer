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
