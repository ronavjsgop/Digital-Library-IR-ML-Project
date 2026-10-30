# 📚 Digital Library Information Retrieval System Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Library-scikit--learn-orange.svg)
![Pandas](https://img.shields.io/badge/Library-Pandas-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

> **Project Title:** Digital Library Information Retrieval System Using Machine Learning  
> **Project Domain:** Information Retrieval (IR) and Machine Learning  
> **Programming Language:** Python  
> **Dataset:** Local illustrative digital library dataset  
> **Algorithms:** K-Nearest Neighbors, Multinomial Naive Bayes, Logistic Regression  
> **Purpose:** Academic learning and demonstration of machine learning-based book classification and information retrieval.

---

## 📑 Table of Contents

1. Project Overview
2. Problem Statement
3. Project Objectives
4. Dataset Description
5. System Methodology
6. Machine Learning Algorithms
7. Evaluation Metrics
8. Experimental Results
9. Visualizations
10. Book Search Demonstration
11. Repository Structure
12. Installation and Execution Guide
13. Technologies Used
14. Conclusion
15. References

---

## 1. Project Overview

The **Digital Library Information Retrieval System** is a Python-based project that demonstrates how machine learning and information retrieval techniques can be used to organize and search digital library resources.

The system uses book information such as titles, descriptions, and categories to classify books into relevant subject areas. It also provides a content-based search feature that retrieves books based on the similarity between a user's search query and the available book information.

Three machine learning algorithms are implemented and compared to evaluate their classification performance.

### Key Highlights

- **Book Classification:** Automatically classifies books into their respective categories.
- **Information Retrieval:** Retrieves relevant books based on user search queries.
- **TF-IDF Vectorization:** Converts textual book information into numerical feature vectors.
- **Cosine Similarity:** Measures similarity between search queries and book descriptions.
- **Comparative Machine Learning:** Evaluates three different classification algorithms.
- **Performance Visualization:** Generates comparison graphs and confusion matrices.
- **Evaluation:** Uses accuracy, precision, recall, and F1-score.

---

## 2. Problem Statement

Digital libraries contain large collections of books and learning resources. Finding relevant information manually can become difficult when the number of available resources increases.

Traditional keyword-based searching may not always retrieve the most relevant books because users may describe their information needs differently from the wording used in book titles or descriptions.

This project addresses these challenges by implementing a machine learning-based digital library retrieval system.

### Major Challenges

1. **Information Organization:** Books need to be grouped into meaningful subject categories.
2. **Text Processing:** Book titles and descriptions must be converted into numerical representations.
3. **Relevant Retrieval:** Search results should be ranked according to their relevance to the user's query.
4. **Classification Performance:** Different machine learning algorithms may produce different classification results.
5. **Evaluation:** Model performance must be measured using appropriate evaluation metrics.

---

## 3. Project Objectives

The main objectives are:

1. To develop a digital library information retrieval system using Python.
2. To organize books into different subject categories.
3. To implement text preprocessing and TF-IDF vectorization.
4. To apply machine learning algorithms for book classification.
5. To retrieve relevant books using cosine similarity.
6. To compare the performance of multiple machine learning algorithms.
7. To generate graphical representations of experimental results.

---

## 4. Dataset Description

The project uses a locally stored CSV dataset containing illustrative book records.

**Dataset location:** `data/books.csv`

### Dataset Details

| Attribute | Description |
|---|---|
| Total Records | 95 books |
| Number of Categories | 7 |
| File Format | CSV |
| Data Type | Structured book information |
| Dataset Source | Included locally with the project |
| Dataset Nature | Illustrative academic dataset |

### Book Categories

The dataset contains books belonging to the following categories:

- Biology
- Chemistry
- Computer Science
- Education
- History
- Mathematics
- Physics

The dataset is intended for demonstrating the system's classification and retrieval capabilities. It should not be interpreted as a comprehensive real-world digital library collection.

---

## 5. System Methodology

The project follows a sequence of data processing, model training, evaluation, and information retrieval operations.

### System Workflow

1. Load the book dataset from the CSV file.
2. Prepare textual information for machine learning.
3. Convert text into numerical features using TF-IDF.
4. Divide the dataset into training and testing sets.
5. Train the machine learning classification algorithms.
6. Evaluate each model using classification metrics.
7. Compare the performance of the trained models.
8. Accept a search query from the user.
9. Calculate cosine similarity between the query and book information.
10. Display the most relevant books.

### TF-IDF Vectorization

Term Frequency-Inverse Document Frequency (TF-IDF) converts textual information into numerical vectors.

It gives greater importance to terms that are useful for identifying a document while reducing the influence of terms that occur frequently across documents.

### Cosine Similarity

Cosine similarity measures the similarity between two numerical text vectors.

The system uses this technique to rank books according to their relevance to the user's search query.

---

## 6. Machine Learning Algorithms

Three classification algorithms are implemented and evaluated.

### 6.1 K-Nearest Neighbors (KNN)

KNN classifies a book based on the categories of its nearest neighboring training examples.

It uses similarity between feature vectors to identify nearby records.

### 6.2 Multinomial Naive Bayes

Multinomial Naive Bayes is a probabilistic classification algorithm commonly used for text classification.

It predicts a category based on the probability of the observed text features.

### 6.3 Logistic Regression

Logistic Regression is a supervised machine learning algorithm used for classification.

It estimates the probability that a book belongs to a particular category.

---

## 7. Evaluation Metrics

The classification algorithms are evaluated using the following metrics:

- **Accuracy:** Measures the proportion of correctly classified books.
- **Precision:** Measures how many books predicted to belong to a category actually belong to that category.
- **Recall:** Measures how many books from a category are correctly identified.
- **Weighted F1-Score:** Combines precision and recall while accounting for the number of samples in each category.

Training time and prediction time are also measured to compare computational performance.

---

## 8. Experimental Results

The dataset is divided into training and testing sets using a stratified 75:25 split.

The three algorithms are evaluated using the same test dataset.

### Model Comparison Results

| Algorithm | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score |
|---|---:|---:|---:|---:|
| K-Nearest Neighbors | 58.33% | 57.08% | 58.33% | 54.62% |
| Logistic Regression | 58.33% | 50.40% | 58.33% | 50.96% |
| Multinomial Naive Bayes | 54.17% | 42.86% | 54.17% | 46.65% |

### Experimental Observations

1. KNN and Logistic Regression achieved the highest accuracy of 58.33%.
2. KNN achieved the highest weighted F1-score of 54.62%.
3. Logistic Regression achieved the same accuracy as KNN but a lower weighted F1-score.
4. Multinomial Naive Bayes achieved an accuracy of 54.17% on the test dataset.
5. KNN performed best overall based on accuracy and weighted F1-score.

**Best-performing algorithm in this experiment:** K-Nearest Neighbors (KNN).

> Note: These results are from a small illustrative dataset of 95 book records. They demonstrate the implementation and comparison of the algorithms, not their expected performance on large real-world library collections.

---

## 9. Visualizations

The project generates graphical outputs to make model evaluation easier to understand.

### Accuracy Comparison Graph

The graph compares the accuracy achieved by the three machine learning algorithms.

![Accuracy Comparison Graph](results/accuracy_comparison.png)

### Confusion Matrices

Confusion matrices are generated to examine the classification predictions and errors for the machine learning models.

The generated confusion matrix images are stored in the `results/` directory.

---

## 10. Book Search Demonstration

The system includes a content-based book search feature using TF-IDF and cosine similarity.

It accepts a text query and retrieves the five most relevant books from the dataset.

### Example 1: Machine Learning Algorithms

**Search Query:** `machine learning algorithms`

Example retrieved books:

| Book Title | Category | Similarity |
|---|---|---:|
| Artificial Intelligence | Computer Science | 0.423 |
| Machine Learning Fundamentals | Computer Science | 0.304 |
| Online Learning | Education | 0.105 |
| Numerical Methods | Mathematics | 0.104 |
| Algorithms Explained | Computer Science | 0.100 |

### Example 2: Online Learning

**Search Query:** `online learning student education`

The system retrieves relevant educational resources, including:

- Online Learning
- Distance Education
- Educational Psychology
- Digital Education Tools
- Teaching Methods

The results are ranked according to their cosine similarity scores.

---

## 11. Repository Structure

```text
Digital_Library_IR_ML_Project/
│
├── data/
│   └── books.csv
│
├── results/
│   ├── algorithm_comparison.csv
│   ├── accuracy_comparison.png
│   └── confusion matrix images
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

### File Descriptions

| File / Folder | Purpose |
|---|---|
| `main.py` | Main Python implementation |
| `data/books.csv` | Book dataset |
| `results/` | Generated evaluation results and graphs |
| `requirements.txt` | Required Python libraries |
| `README.md` | Project documentation |
| `.gitignore` | Excludes unnecessary files from Git |

---

## 12. Installation and Execution Guide

### Prerequisites

- Python installed on the system.
- Visual Studio Code (recommended).
- Git (optional, for GitHub upload).

### Windows PowerShell Setup

**Step 1: Open the project folder**

Open the extracted project folder in Visual Studio Code.

**Step 2: Open the terminal**

Select Terminal → New Terminal.

**Step 3: Create a virtual environment**

```powershell
python -m venv venv
```

**Step 4: Activate the environment**

```powershell
.\venv\Scripts\Activate.ps1
```

**Step 5: Install dependencies**

```powershell
python -m pip install -r requirements.txt
```

**Step 6: Run the project**

```powershell
python main.py
```

The program trains the classification models, displays their evaluation results, performs book searches, and generates result files in the `results` folder.

---

## 13. Technologies Used

| Technology | Application |
|---|---|
| Python | Main programming language |
| Pandas | Dataset loading and manipulation |
| Scikit-learn | Machine learning and text vectorization |
| TF-IDF | Text feature extraction |
| Cosine Similarity | Similarity-based book retrieval |
| Matplotlib | Performance visualization |
| CSV | Dataset storage |

---

## 14. Conclusion

The Digital Library Information Retrieval System demonstrates how machine learning and information retrieval techniques can be combined to organize and search digital library resources.

The project implements three machine learning algorithms: K-Nearest Neighbors, Logistic Regression, and Multinomial Naive Bayes.

The experimental comparison shows that KNN achieved the highest weighted F1-score and tied for the highest accuracy on the included test dataset.

The TF-IDF and cosine similarity-based search feature also demonstrates how relevant books can be retrieved based on user queries.

Overall, this project provides a practical introduction to text processing, supervised machine learning, information retrieval, model evaluation, and data visualization.

---

## 15. References

1. **Scikit-learn Documentation:** https://scikit-learn.org/stable/
2. **Pandas Documentation:** https://pandas.pydata.org/docs/
3. **Matplotlib Documentation:** https://matplotlib.org/stable/
4. **TF-IDF Vectorization:** https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
5. **Cosine Similarity:** https://scikit-learn.org/stable/modules/metrics.html

---

*Thank you for reviewing this project. This repository is intended for academic learning and demonstration of Digital Library Information Retrieval using Machine Learning.*
