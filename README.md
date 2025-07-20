# The Source Code Maintenance Time Classifications from Code Smell

## 📌 Project Objective

This project aims to analyze and classify the **maintenance time** of source code based on detected **code smells**. By using actively maintained **open-source Java projects** (Java 11+), it extracts commit histories from GitHub and integrates **SonarQube** static code analysis to evaluate how code complexity and quality influence maintenance duration.


---

## 🔍 Project Overview

Code Smells and Their Impact on Source Code Maintenance Time

A code smell refers to a characteristic in the source code that may indicate potential structural issues or quality concerns. Such issues can lead to software errors or make the codebase more difficult to read and maintain.

This research focuses on evaluating the quality of source code during the development process by measuring the occurrence of code smells. The collected data will be used to develop predictive models using LightGBM and Random Forest. These models aim to determine whether specific types of code smells have a measurable impact on the maintenance time required for source code.

The research is guided by two key questions:

1. Do different types of code smells correlate with the maintenance time required to fix source code?

2. Which types of code smells have the most significant impact on maintenance time?

---

## 🛠️ Tools and Technologies Used

- **Programming Language:** Python (version 3.11+)
- **Environment:** Conda Virtual Environment
- **IDE:** PyCharm (with support for Remote IDE via Coder)
- **Operating System:** Linux (Ubuntu), Mac
- **Data Collection:** GitHub REST API
- **Code Analysis:** [SonarQube](https://www.sonarsource.com/products/sonarqube/) (code smells, cyclomatic complexity, duplications, maintainability index)
- **Model Evaluation Pipeline:** [SMOET](https://github.com/pchongs/SMOET) – Software Maintenance Optimization Evaluation Tool
- **Hyperparameter Tuning:** [Optuna](https://optuna.org/)
- **Source Projects:** Java-based repositories (version 11+)

---

## 📈 Workflow

### 1 Data Collection
- Select actively maintained open-source Java projects from GitHub
- Extract commit data, pull request metadata, contributors, file diffs, and timestamps
- Structure data into clean DataFrames for downstream processing

### 2 Code Smell Detection
- Analyze source code using SonarQube to detect:
  - Code smells
  - Cyclomatic complexity
  - Duplications
  - Maintainability metrics
- Integrate SonarQube results with GitHub metadata

### 3 Data Cleaning & Preprocessing
- Filter relevant features: severity of code smells, size of changes, affected files, commit intervals
- Apply clustering before classification:
  - **K-Means Clustering**
  - **Percentile-based Binning**
  - **Hierarchical Clustering**
  - **Quartile-based Grouping**

### 4 Classification & Model Evaluation
- Train classification models to predict maintenance time groupings:
  - **Random Forest Classifier**
  - **LightGBM**
- Manage pipeline using **SMOET**
- Use **Optuna** for hyperparameter optimization

---

## 🎯 Expected Outcomes and Impact

- A machine learning model that classifies Java source code based on expected **maintenance time**
- Potential applications:
  - Assist in **development cycle planning**
  - Detect **high-maintenance components** early
  - Enhance **DevOps/QA decision-making**
  - Inform **refactoring and long-term sustainability** strategies


