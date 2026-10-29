from pathlib import Path
import time
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, ConfusionMatrixDisplay
)
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "books.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

def make_text(df):
    # Combine title and description so both influence classification and search.
    return (df["title"].astype(str) + ". " + df["description"].astype(str)).tolist()

def build_pipeline(classifier):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, stop_words="english",
            ngram_range=(1, 2), max_features=12000, sublinear_tf=True
        )),
        ("classifier", classifier)
    ])

def evaluate_models(df):
    texts = make_text(df)
    labels = df["category"].astype(str).tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5, metric="cosine", algorithm="brute"),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42)
    }
    rows = []

    for name, classifier in models.items():
        model = build_pipeline(classifier)
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start

        start = time.perf_counter()
        predictions = model.predict(X_test)
        prediction_time = time.perf_counter() - start

        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average="weighted", zero_division=0
        )
        rows.append({
            "algorithm": name,
            "accuracy_percent": round(accuracy * 100, 2),
            "weighted_precision_percent": round(precision * 100, 2),
            "weighted_recall_percent": round(recall * 100, 2),
            "weighted_f1_percent": round(f1 * 100, 2),
            "training_time_seconds": round(train_time, 4),
            "prediction_time_seconds": round(prediction_time, 4)
        })

        print("\n" + "=" * 68)
        print(f"MODEL: {name}")
        print("=" * 68)
        print(f"Accuracy:  {accuracy * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall:    {recall * 100:.2f}%")
        print(f"F1-score:  {f1 * 100:.2f}%")
        print(f"Training time:   {train_time:.4f} seconds")
        print(f"Prediction time: {prediction_time:.4f} seconds")
        print(classification_report(y_test, predictions, zero_division=0))

        fig, ax = plt.subplots(figsize=(9, 7))
        ConfusionMatrixDisplay.from_predictions(
            y_test, predictions, xticks_rotation=45,
            cmap="Blues", colorbar=False, ax=ax
        )
        ax.set_title(f"Digital Library — {name}")
        fig.tight_layout()
        safe = name.lower().replace(" ", "_")
        fig.savefig(RESULTS / f"{safe}_confusion_matrix.png", dpi=160)
        plt.close(fig)

    results = pd.DataFrame(rows).sort_values(
        ["accuracy_percent", "weighted_f1_percent"], ascending=False
    ).reset_index(drop=True)
    results.insert(0, "rank_by_accuracy", range(1, len(results) + 1))
    results.to_csv(RESULTS / "algorithm_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    results.sort_values("accuracy_percent").plot(
        x="algorithm", y="accuracy_percent", kind="barh",
        legend=False, ax=ax
    )
    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("Algorithm")
    ax.set_xlim(0, 100)
    ax.set_title("Digital Library — Algorithm Accuracy Comparison")
    fig.tight_layout()
    fig.savefig(RESULTS / "accuracy_comparison.png", dpi=160)
    plt.close(fig)

    print("\nFINAL RANKING (sorted by accuracy, then weighted F1-score)")
    print(results[[
        "rank_by_accuracy", "algorithm", "accuracy_percent",
        "weighted_f1_percent", "training_time_seconds"
    ]].to_string(index=False))
    print(f"\nSaved results to: {RESULTS}")
    return results

def search_books(df):
    texts = make_text(df)
    vectorizer = TfidfVectorizer(
        lowercase=True, stop_words="english",
        ngram_range=(1, 2), max_features=12000, sublinear_tf=True
    )
    document_vectors = vectorizer.fit_transform(texts)

    print("\n" + "=" * 68)
    print("DIGITAL LIBRARY SEARCH")
    print("Example: machine learning algorithms")
    print("Type 'quit' to stop searching.")
    print("=" * 68)

    while True:
        query = input("\nEnter book search query: ").strip()
        if query.lower() in {"quit", "exit", "q"}:
            break
        if not query:
            print("Please enter a search query.")
            continue

        query_vector = vectorizer.transform([query])
        scores = cosine_similarity(query_vector, document_vectors).ravel()
        top_indices = scores.argsort()[::-1][:5]

        if scores[top_indices[0]] <= 0:
            print("No close keyword match found. Try different keywords.")
            continue

        print("\nTop matching books:")
        shown = 0
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            shown += 1
            row = df.iloc[idx]
            print(f"\n{shown}. {row['title']}")
            print(f"   Category: {row['category']} | Similarity: {scores[idx]:.3f}")
            print(f"   {row['description']}")
        if shown == 0:
            print("No matching books found.")

def main():
    print("=" * 68)
    print("DIGITAL LIBRARY INFORMATION RETRIEVAL SYSTEM")
    print("Three-Algorithm Machine Learning Comparison")
    print("=" * 68)

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE).dropna(subset=["title", "description", "category"])
    print(f"Dataset loaded: {len(df)} books")
    print(f"Categories: {', '.join(sorted(df['category'].unique()))}")
    print("\nTraining and evaluating the three algorithms...")
    evaluate_models(df)
    search_books(df)
    print("\nProgram finished. See the results folder for charts and CSV metrics.")

if __name__ == "__main__":
    main()
