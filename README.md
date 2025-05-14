# Machine-Learning-for-Imbalanced-Fraud-Data
Here’s a polished `README.md` you can use for your credit card fraud detection project, optimized for GitHub or a portfolio:

---

# 🧠 Credit Card Fraud Detection Using Resampling Techniques

This project demonstrates how to build a **machine learning model** that detects fraudulent credit card transactions using **imbalanced classification techniques**. The dataset is highly skewed, with fraudulent cases making up a very small portion of the total, making it ideal for experimenting with **resampling strategies** like `RandomOverSampler`, `SMOTE`, and `ClusterCentroids`.

---

## 📁 Dataset

The dataset used is the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud), containing **284,807 transactions** made by European cardholders in 2013.

* **Features**: 28 principal components (`V1`–`V28`) from PCA
* **Target**: `Class`

  * `0`: Legitimate
  * `1`: Fraudulent

---

## 🛠️ Technologies Used

* Python 3.7+
* Pandas, NumPy, Matplotlib, Seaborn
* Scikit-learn
* Imbalanced-learn (`imbalanced-learn`)

---

## 📌 Objectives

* Analyze and visualize the data distribution
* Build supervised learning models for binary classification
* Improve classification performance using resampling:

  * `RandomOverSampler`
  * `SMOTE`
  * `ClusterCentroids`
  * `SMOTEENN`
* Evaluate models using:

  * Confusion Matrix
  * Precision / Recall
  * F1 Score
  * Balanced Accuracy

---

## 📊 Project Workflow

1. **Load and preprocess** the dataset
2. **EDA**: Understand the class imbalance and feature correlations
3. **Train-test split**
4. **Apply resampling** strategies to balance the data
5. **Train logistic regression models** on each resampled dataset
6. **Evaluate** performance using classification metrics
7. **Compare** effectiveness of each technique

---

## 🧪 Model Performance Summary

| Model              | Accuracy | Precision (Fraud) | Recall (Fraud) | F1 Score |
| ------------------ | -------- | ----------------- | -------------- | -------- |
| Random OverSampler | \~64%    | 1%                | \~66%          | \~2%     |
| SMOTE              | \~65%    | 1%                | \~61%          | \~2%     |
| ClusterCentroids   | \~54%    | 1%                | \~69%          | \~1%     |
| SMOTEENN           | \~64%    | 1%                | \~72%          | \~2%     |

> ⚠️ Note: Despite resampling, detecting fraud remains a challenge due to extreme class imbalance. Precision for the minority class (fraud) is low, but recall improves significantly.

---

## ✅ Conclusion

This project illustrates the importance of handling **imbalanced datasets** properly in fraud detection tasks. Techniques like `SMOTE` and `SMOTEENN` improve the recall for fraudulent transactions but may suffer in precision. Future improvements may include:

* Using **ensemble models** (e.g., EasyEnsembleClassifier, BalancedRandomForestClassifier)
* Applying **feature selection** or **dimensionality reduction**
* Trying other classifiers (e.g., XGBoost, Random Forest, Neural Networks)

---

## 📎 References

* [Kaggle: Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
* [imbalanced-learn Documentation](https://imbalanced-learn.org/)
* [Scikit-learn Documentation](https://scikit-learn.org/)


