# 📊 The Source Code Maintenance Time Classifications from Code Smell

This project aims to classify the **source code maintenance time** based on **code smells** identified from Java-based open-source projects. By analyzing GitHub commit histories and static code analysis results from SonarQube, we explore how specific code smells relate to the time spent maintaining software components.

---

## 🔍 Project Overview

Understanding the time required to maintain code is crucial for software project planning. This project integrates GitHub history data and SonarQube analysis to build predictive models for estimating maintenance time based on code quality issues (e.g., code smells, complexity, duplications).

---

## ⚙️ Tech Stack

- **Programming Language:** Python 3.11+
- **Environment:** Conda
- **Source Repositories:** Java (version 11+)
- **Data Collection:** GitHub REST API
- **Static Analysis:** [SonarQube](https://www.sonarsource.com/products/sonarqube/)
- **Modeling Pipeline:** [SMOET](https://github.com/software-design-lab/SMOET)
- **Hyperparameter Optimization:** [Optuna](https://optuna.org/)

---

## 📈 Workflow

### 1. Data Collection  
- Select open-source Java repositories from GitHub.  
- Use GitHub API to collect commit histories, authorship, and file-level changes.  
- Store the data in a structured DataFrame format.

### 2. Code Smell Detection  
- Analyze source code with SonarQube to extract:
  - Code Smells
  - Complexity Metrics
  - Maintainability Index

### 3. Data Cleaning & Preprocessing  
- Remove irrelevant and duplicate features.  
- Transform and normalize selected metrics.  
- Apply clustering to group similar maintenance profiles.

### 4. Classification & Model Evaluation  
- Use SMOET for model training and evaluation.  
- Apply Optuna for hyperparameter tuning.  
- Classify code into maintenance time categories (e.g., short, medium, long).

---

## 📊 Expected Outcomes

- A trained model that predicts maintenance time based on code quality metrics.  
- Analysis of code smell patterns that correlate with high maintenance cost.  
- Insights to support technical debt management and refactoring decisions.

---

