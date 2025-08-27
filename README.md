# Rate My Professor Analysis – Data Science Capstone

This project explores professor rating data from Rate My Professor to uncover statistical biases, model professor ratings, and predict pepper status. It combines **data cleaning**, **exploratory analysis**, **hypothesis testing**, and **machine learning** techniques using **Python**.

---

## 1) Clone the repository

```bash
git clone https://github.com/<your-username>/rate-my-professor-analysis.git
cd rate-my-professor-analysis
```

---

## 2) Set up the Python environment

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install required libraries
pip install -r requirements.txt
```

---

## 3) Project Structure

```
rate-my-professor-analysis/
├── data/
│   └── rmp_dataset.csv
├── notebooks/
│   └── rmp_analysis.ipynb
├── output/
│   └── final_report.pdf
├── README.md
├── requirements.txt
```

---

## 4) Data Description

The dataset contains anonymized records of professor ratings including:
- Quality rating (1-5 scale)
- Difficulty rating (1-5 scale)
- Pepper status (binary)
- Department, school, and gender info

---

## 5) Preprocessing Steps

```python
# Dropping nulls, converting datatypes, and standardizing column names
df.dropna(inplace=True)
df.columns = df.columns.str.lower().str.replace(' ', '_')
```

---

## 6) Statistical Analysis

- **Mann–Whitney U test** used to test gender-based bias in quality ratings.
- **Spearman correlation** used to assess monotonic relationships between difficulty and quality.
- Key Insight: Quality and difficulty are **negatively correlated**.

---

## 7) Regression Modeling

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

- **Target Variable**: Quality rating
- **R² score**: `0.56` – moderate explanatory power from difficulty and other features.

---

## 8) Classification Model for Pepper Status

```python
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
```

- Used **oversampling (SMOTE)** to balance the pepper class.
- **AUROC**: `0.85` – high capability to distinguish pepper vs. non-pepper professors.

---

## 9) Visualizations

Created charts using `matplotlib` and `seaborn`:
- Rating distributions by gender
- Correlation heatmaps
- Feature importance plots for classification

---

## 10) PDF Report

Find the full research report in `/output/final_report.pdf`, which includes:
- Objective
- Methodology
- Key findings
- Visualizations
- Model summaries

---

## 11) Run Notebook

Open the Jupyter Notebook to view all code and outputs:

```bash
jupyter notebook notebooks/rmp_analysis.ipynb
```

---

## 💡 Future Work

- Apply natural language processing (NLP) to textual comments
- Build dashboard with Streamlit or Dash

