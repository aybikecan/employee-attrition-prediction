# Employee Attrition Prediction

This project investigates employee attrition as a binary classification problem.
It compares several machine learning models, tunes the strongest candidate,
examines feature importance, and tests whether selecting the ten most important
features improves predictive performance.

## Research Questions

The notebook evaluates two hypotheses:

- **H1:** Ensemble models will outperform baseline Logistic Regression.
- **H2:** Feature selection will improve performance compared with using all
  available features.

## Project Workflow

- Download and inspect the employee attrition dataset
- Check missing values and target-class distribution
- Remove identifier columns
- Encode categorical variables and scale features
- Train Logistic Regression, Random Forest, and Gradient Boosting models
- Compare accuracy, precision, recall, F1 score, and ROC-AUC
- Tune Gradient Boosting with grid search
- Analyse feature importance
- Retrain the selected model with the ten most important features
- Perform five-fold cross-validation
- Compare ROC curves and inspect the confusion matrix

## Dataset

The project uses the
[Employee Attrition Dataset](https://www.kaggle.com/datasets/stealthtechnologies/employee-attrition-dataset)
from Kaggle.

The dataset is downloaded at runtime and is not included in this repository.
Please review the Kaggle page for its provenance, license, limitations, and terms
of use.

## Models and Results

Recorded holdout-set results:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Tuned Gradient Boosting | 0.7608 | 0.7479 | 0.7440 | 0.7459 | 0.8524 |
| Gradient Boosting | 0.7579 | 0.7448 | 0.7408 | 0.7428 | 0.8498 |
| Logistic Regression | 0.7555 | 0.7433 | 0.7362 | 0.7397 | 0.8451 |
| Random Forest | 0.7458 | 0.7349 | 0.7217 | 0.7282 | 0.8356 |
| Tuned Gradient Boosting — selected features | 0.7438 | 0.7332 | 0.7186 | 0.7258 | 0.8312 |

The best Gradient Boosting parameters were:

```text
learning_rate: 0.1
max_depth: 3
n_estimators: 200
```

Its mean five-fold cross-validation F1 score was **0.7438**.

## Findings

- Tuned Gradient Boosting produced the strongest holdout performance, offering
  modest support for H1.
- Restricting the model to the ten most important features reduced performance,
  so H2 was not supported by this experiment.
- Important features in the recorded model included marital status, job level,
  remote work, work-life balance, promotions, and distance from home.

Feature importance reflects associations learned from this dataset; it does not
establish that these variables cause employee attrition.

## Technologies

- Python
- pandas and NumPy
- scikit-learn
- Matplotlib and seaborn
- Kaggle API
- Jupyter Notebook / Google Colab

## Running the Notebook

1. Open `Employee_Attrition_Prediction.ipynb` in Google Colab.
2. Add the following values in Colab's **Secrets** panel:
   - `KAGGLE_USERNAME`
   - `KAGGLE_KEY`
3. Enable notebook access for both secrets.
4. Run the cells in order.

The notebook retrieves Kaggle credentials securely:

```python
from google.colab import userdata

os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
```

Never commit API keys, credential files, or restricted datasets to Git.

## Repository Structure

```text
employee-attrition-prediction/
├── Employee_Attrition_Prediction.ipynb
└── README.md
```

## Limitations and Responsible Use

This project is for educational and exploratory purposes. It should not be used
as an automated employment decision system or as the sole basis for hiring,
retention, promotion, disciplinary, or termination decisions.

The data may contain historical and sampling biases. Employee behaviour is
context-dependent, predictive performance is moderate, and model outputs may
produce unequal impacts across demographic or employment groups. A real
deployment would require representative data, fairness analysis, privacy
controls, external validation, monitoring, and meaningful human oversight.

