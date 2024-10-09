# 🪻 Iris Flower Classification

This project implements a machine learning classification model to predict the species of Iris flowers based on four features: sepal length, sepal width, petal length, and petal width. It follows an end-to-end approach of data exploration, model training, and evaluation. Notably, this project is also featured as the first example in *An Introduction to Probabilistic Machine Learning* by Kevin P. Murphy.

## 📋 Table of Contents
- [📖 Overview](#overview)
- [📊 Dataset Information](#dataset-information)
- [📉 Data Analysis and Visualization](#data-analysis-and-visualization)
- [🤖 Model Training and Evaluation](#model-training-and-evaluation)
- [📈 Visualizations](#visualizations)
- [🛠️ Libraries Used](#libraries-used)
- [📝 Conclusion](#conclusion)
- [🚀 How to Use](#how-to-use)

## 📖 Overview
The project involves classifying Iris flower species (Setosa, Versicolor, Virginica) using the famous Iris dataset. It covers data exploration, model training, evaluation, and prediction. Two machine learning models are implemented: Naive Bayes and MLP Classifier.

## 📊 Dataset Information
- **Source:** Iris Flower Dataset, a widely-used dataset for classification tasks, with 150 samples across three species.
- **Features:**
  - `Sepal Length (cm)`
  - `Sepal Width (cm)`
  - `Petal Length (cm)`
  - `Petal Width (cm)`
  - `Species`: The class label representing the species (Setosa, Versicolor, Virginica)

## 📉 Data Analysis and Visualization
Key insights from the dataset:
1. **Balanced Classes:** Each species has 50 samples.
2. **Feature Correlations:** Petal length and petal width are strongly correlated, as are sepal length and petal length.
3. **Species Separation:** Setosa can be easily separated from the other two species, while Versicolor and Virginica require more advanced modeling techniques for proper classification.

## 🤖 Model Training and Evaluation
Two classifiers were used for predicting the species:
- **Naive Bayes Classifier**
- **MLP (Multi-Layer Perceptron) Classifier**

The models were evaluated using accuracy, precision, recall, and F1-score. The MLP classifier outperformed Naive Bayes, achieving the highest accuracy.

## 📈 Visualizations
- **Pair Plot:** A scatter plot matrix displaying relationships between features for each species.
- **Confusion Matrix:** A visual breakdown of classification results for each species.

## 🛠️ Libraries Used
- **Python:** ![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
- **Pandas:** For data manipulation. ![Pandas](https://img.shields.io/badge/pandas-1.2.4-orange)
- **NumPy:** For numerical operations. ![NumPy](https://img.shields.io/badge/numpy-1.19.2-orange)
- **Matplotlib:** For visualizations. ![Matplotlib](https://img.shields.io/badge/matplotlib-3.3.4-orange)
- **Seaborn:** For statistical plotting. ![Seaborn](https://img.shields.io/badge/seaborn-0.11.1-orange)
- **Scikit-learn:** For machine learning models. ![Scikit-learn](https://img.shields.io/badge/scikit--learn-0.24.1-orange)

## 📝 Conclusion
This project demonstrated the effectiveness of machine learning algorithms for classification tasks. The MLP classifier provided the best results for predicting Iris species based on sepal and petal measurements.

## 🚀 How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/devgupta2619/IRIS_FLOWER_CLASSIFICATION.git
   ```
2. Install the required libraries:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
3. Run the Jupyter Notebook or Python script to view the analysis and predictions.

---
