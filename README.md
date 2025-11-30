# Kasparro Agentic Facebook Performance Analyst

A multi-agent autonomous system that diagnoses Facebook Ads performance, explains ROAS fluctuations, and generates new creative recommendations using both quantitative signals and creative messaging patterns.

This project implements an Agentic Analytics pipeline similar to human marketing analysts — combining structured reasoning, automated hypothesis generation, quantitative evaluation, and creative ideation.

---

## 🚀 Core Objectives

| Capability | Description |
|-----------|-------------|
| **Diagnose ROAS changes** | Automatically detect when and why ROAS moved over time. |
| **Identify performance drivers** | Evaluate audience fatigue, creative decline, CPM/CTR drop, geo shifts, etc. |
| **Generate creative recommendations** | Create headlines, messages, CTAs for campaigns with low CTR. |
| **Structured quant evidence** | Output validated hypotheses with numeric evidence & confidence scoring. |
| **Human-readable report** | Produce a final marketing summary for decision makers. |

---

## 🧠 Multi-Agent System Architecture

| Agent | Responsibilities |
|--------|-----------------|
| **Planner Agent** | Interprets query, breaks into subtasks, assigns downstream roles. |
| **Data Agent** | Loads CSV, aggregates metrics, detects low-CTR campaigns, builds compact summaries. |
| **Insight Agent** | Generates hypotheses explaining performance patterns. |
| **Evaluator Agent** | Tests hypotheses with numeric evidence and confidence scoring. |
| **Creative Improvement Generator** | Suggests new message directions grounded in dataset style. |

### 🔁 Data & Agent Flow Diagram

```mermaid
flowchart TD
    U[User Query] --> P[Planner Agent]
    P --> D[Data Agent]
    D --> P
    P --> I[Insight Agent]
    I --> E[Evaluator Agent]
    D --> I
    D --> E
    D --> C[Creative Generator]
    E --> OUT1[(insights.json)]
    C --> OUT2[(creatives.json)]
    E --> R[Report Builder]
    C --> R
    D --> R
    R --> OUT3[(report.md)]
