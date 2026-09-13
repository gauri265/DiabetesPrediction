

# Diabetes Prediction Using Gaussian Naive Bayes

## Project Overview

This project uses **Gaussian Naive Bayes** to predict whether a person has diabetes based on health-related and demographic attributes.

The project focuses on binary classification:

* `0` — No Diabetes
* `1` — Diabetes

## Objectives

* Predict possible diabetes cases using health-related features.
* Apply data preprocessing techniques.
* Train a Gaussian Naive Bayes classification model.
* Evaluate the model using accuracy, precision, recall, and F1-score.
* Improve the detection of diabetic cases through probability threshold evaluation.

## Dataset

The dataset originally contained **100,000 records and 8 input features**. After removing duplicate records, **96,146 records** remained.

### Features Used

* Gender
* Age
* Hypertension
* Heart Disease
* Smoking History
* BMI
* HbA1c Level
* Blood Glucose Level

### Target Variable

* `diabetes`

## Data Preprocessing

The following preprocessing steps were performed:

1. Removed duplicate records.
2. Encoded categorical features.
3. Split the dataset into training and testing sets using an 80:20 ratio.
4. Standardized the features using feature scaling.

## Algorithm Used

### Gaussian Naive Bayes

Gaussian Naive Bayes was selected because it is:

* Simple and computationally efficient.
* Suitable for probability-based classification.
* Easy to interpret.
* Suitable for continuous numerical features that can be modeled using a Gaussian distribution.

The algorithm assumes conditional independence between features and calculates the probability of each class.

## Model Evaluation

Different probability thresholds were evaluated to improve the detection of diabetic cases. A threshold of **0.10** was selected.

### Final Results

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 90.42% |
| Precision | 47.11% |
| Recall    | 70.75% |
| F1-Score  | 56.56% |

The selected threshold improved recall, allowing the model to identify a greater proportion of diabetic cases.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook

## How to Run

1. Clone this repository:

```bash
git clone https://github.com/gauri265/DiabetesPrediction.git
```

2. Navigate to the project directory:

```bash
cd DiabetesPrediction
```

3. Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

4. Open the notebook:

```bash
jupyter notebook diabetes.ipynb
```

5. Run the notebook cells in order.

## Conclusion

Gaussian Naive Bayes provided an efficient approach for diabetes classification. Probability threshold evaluation helped improve the detection of diabetic cases, making recall an important performance measure for this project.

**Note:** This project is intended for educational purposes and should not be used as a substitute for professional medical diagnosis.
