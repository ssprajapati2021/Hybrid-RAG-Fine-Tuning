# Hybrid RAG + Parameter-Efficient Fine-Tuning for Customer Support

## Project Overview
This project implements a Hybrid Retrieval-Augmented Generation (Hybrid RAG) pipeline combined with Parameter-Efficient Fine-Tuning (QLoRA) for customer support intent classification and policy-grounded response generation.

## Problem Statement
General-purpose LLMs often hallucinate customer support responses because they lack access to organization-specific knowledge. This project addresses the problem by combining fine-tuned intent classification with retrieval from Standard Operating Procedures (SOPs).

## Technology Stack
- Python
- Hugging Face Transformers
- PEFT (QLoRA)
- LangChain
- Chroma DB
- Sentence-Transformers (all-MiniLM-L6-v2)
- MLflow
- Google Colab

## Architecture

```
Customer Query
        │
        ▼
 Fine-Tuned Intent Router
        │
        ▼
Intent + Category
        │
        ▼
Embedding Model
        │
        ▼
Chroma Vector Database
        │
        ▼
Relevant SOP Chunk
        │
        ▼
Qwen2.5-1.5B-Instruct
        │
        ▼
Policy-Compliant Response
```

## Repository Structure

```
notebooks/
reports/
artifacts/
models/
```

## Implementation Journey

- Notebook 1 – Data Preparation & EDA
- Notebook 2 – Embeddings & Retrieval Preparation
- Notebook 3 – ChatML Dataset Creation
- Notebook 4 – Hybrid RAG Pipeline
- Notebook 5 – Baseline vs Naive RAG Evaluation
- Notebook 6 – QLoRA Fine-Tuning
- Notebook 7 – Hybrid RAG Evaluation & Comparative Analysis

## Results

### Performance Summary

| Solution | Description |
|----------|-------------|
| Baseline | Base LLM without retrieval |
| Naive RAG | Direct retrieval using customer query |
| Hybrid RAG | Fine-Tuned Intent Router + Retrieval |

Hybrid RAG achieved the highest ROUGE and BLEU scores while significantly reducing hallucinations and improving policy-grounded responses.

## Reports

- Project Proposal
- Comparative Analysis Report

## Future Work

- Multi-domain customer support
- Hybrid Search
- Re-ranking
- Multi-agent workflows
- Enterprise API integration

## Author

**Shobhnath Prajapati**

GitHub: https://github.com/ssprajapati2021