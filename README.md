# Digital Library Information Retrieval System Using Machine Learning

## Project overview
This is an original, beginner-friendly Information Retrieval (IR) project. Users can search a small book catalogue and receive the five most relevant books ranked using TF-IDF and cosine similarity. It also compares three machine-learning algorithms for predicting a book's subject category.

## Algorithms
1. **K-Nearest Neighbors (KNN)** — predicts a category using nearby training documents.
2. **Multinomial Naive Bayes** — probabilistic text classification baseline.
3. **Logistic Regression** — learns weights for text features to predict a category.

## IR technique
- Combines each book's title and description.
- Converts text into TF-IDF vectors.
- Converts a user query into the same vector space.
- Calculates cosine similarity between the query and each book.
- Displays up to five books ranked by similarity.

## Dataset
`data/books.csv` is included in this project. It contains 95 illustrative book-catalogue records across seven subjects: Mathematics, Computer Science, Biology, History, Education, Physics and Chemistry. The descriptions are short, curated sample text for demonstrating the workflow, not a real-world benchmark dataset. Results should therefore be described as a classroom demonstration, not evidence of production-level performance.

## Requirements
- Python 3.10 or later (Python 3.11 is fine)
- Internet is **not required** for the dataset or normal execution.
- Packages in `requirements.txt`

## Run on Windows / VS Code
1. Extract the ZIP.
2. Open the extracted `Digital_Library_IR_ML_Project` folder in VS Code.
3. Open Terminal → New Terminal.
4. Run:

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

If `python` is not recognized, try `py main.py` after installing dependencies.

## Output
After evaluation, the `results` folder will contain:
- `algorithm_comparison.csv` — metrics and ranking for the three models
- `accuracy_comparison.png` — bar chart comparing accuracy
- one confusion matrix image per algorithm

The program then asks for a search query. Example:

```text
machine learning algorithms
```

Type `quit` to stop searching.

## Metrics
- **Accuracy:** percentage of test examples classified correctly.
- **Precision:** among items predicted for a category, the proportion that are correct.
- **Recall:** among examples in a category, the proportion identified correctly.
- **Weighted F1-score:** weighted combination of precision and recall.
- **Training/prediction time:** measured during the current run and can vary by computer.

## Suggested report structure
1. Abstract
2. Introduction
3. Problem statement
4. Objectives
5. Dataset description
6. Text preprocessing and TF-IDF
7. Algorithms used
8. System methodology
9. Experimental setup
10. Results and comparison table
11. Confusion matrix discussion
12. Search demonstration
13. Limitations
14. Conclusion
15. References

## Suggested problem statement
A digital library may contain many books across different subjects, making it difficult for users to locate useful material quickly. This project implements a simple text-based retrieval system that ranks books by relevance to a user's query and compares three machine-learning algorithms for subject classification.

## Important reporting note
Run the program and use the actual metrics printed by your computer and saved in the CSV. Do not invent or assume the best algorithm in advance. Since this is a small illustrative dataset, clearly mention its size and limitations in your report.

## GitHub upload
1. Create a new repository on https://github.com/.
2. Upload `main.py`, `README.md`, `requirements.txt`, and the `data` folder.
3. Do not upload the virtual environment (`venv`).
4. Run the project once and optionally include generated results if your instructor requests them.
5. Copy the repository URL for submission.
