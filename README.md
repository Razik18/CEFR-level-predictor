# Data Mining & Machine Learning

## Project: Detecting the Difficulty Level of French Texts

### Group: Microsoft

#### Participants:
- Abderazzak Saib
- Ajiach Nabil

#### Description:
As described above in the project's Title, the main goal of this Kaggle competition is to predict the difficulty level of a French text according to The Common European Framework of Reference for Languages that describes 6 levels of language: A1, A2, B1, B2, C1, C2.

You can find all information and rules on this Kaggle link: [Kaggle Competition](https://www.kaggle.com/competitions/detecting-french-texts-difficulty-level-2023/overview)

#### Our Youtube Video 
[Watch the video](#)

#### Dataset Description
- `training_data.csv` -> the training set (with sentence and respective difficulty)
- `unlabelled_test_data.csv` -> the test set (with just sentence)
- `rajouter le fichier generer par notre model` -> submission of our prediction with only id and difficulty

#### Columns
- `id`: Numerical identifier of the sentence.
- `sentence`: A sentence in French for which you want to predict the difficulty level.
- `difficulty`: The difficulty level of the sentence (from A1 to C2). This column would be your target variable.

We can upload them from the GitHub Data folder or you can use directly the Kaggle API and your Kaggle credentials.

#### Approach
1. **Installing the Necessary Packages**
   - For language support: `spacy`, `nltk`, `sentencepiece`,`transformers datasets pandas sickit-learn`
   - For modelling: `pandas`, `numpy`, `sklearn`, `torch` etc...

2. **Models**
   - The Models we used in our analysis:
     - Logistic regression
     - KNN
     - CamemBERT
     - Random Forest models.
     - FlauBERT

3. **Data Exploration**

Before diving into model training, we conducted a thorough exploration of the dataset. We noticed that the values were well-distributed across different difficulty levels, indicating a balanced dataset. This observation led us to conclude that extensive data cleaning or balancing was not necessary for this project, allowing us to focus more on feature engineering and model tuning.

4. **Model Training and Hyperparameter Optimization**

This section documents the process of training several machine learning models, including the hyperparameter optimization for each, and provides a comparative analysis of their performance based on several metrics.

## Logistic Regression

### Hyperparameter Tuning

**Tuned Hyperparameters:**
- `C`: Inverse of regularization strength tested over [0.01, 0.1, 1, 10, 100].
- `solver`: Optimization algorithms tested [‘liblinear’, ‘saga’].

**Optimization Method:** Grid Search with 5-fold cross-validation.

**Best Hyperparameters:**
- `C`: 10
- `solver`: liblinear

**Cross-Validation Results**
- Average Accuracy: 0.82
- Best Accuracy Score: 0.85

## k-Nearest Neighbors (kNN)

### Hyperparameter Tuning

**Tuned Hyperparameters:**
- `n_neighbors`: Number of neighbors tested over [3, 5, 7, 9].
- `weights`: Weight function used in prediction [‘uniform’, ‘distance’].

**Optimization Method:** Random Search with 5-fold cross-validation.

**Best Hyperparameters:**
- `n_neighbors`: 5
- `weights`: distance

**Cross-Validation Results**
- Average Accuracy: 0.78
- Best Accuracy Score: 0.81

## Decision Tree

### Hyperparameter Tuning

**Tuned Hyperparameters:**
- `max_depth`: The maximum depth of the tree tested over [10, 20, 30, None].
- `min_samples_split`: Minimum number of samples required to split an internal node tested over [2, 5, 10].

**Optimization Method:** Grid Search with 5-fold cross-validation.

**Best Hyperparameters:**
- `max_depth`: 20
- `min_samples_split`: 5

**Cross-Validation Results**
- Average Accuracy: 0.76
- Best Accuracy Score: 0.79

## Random Forests

### Hyperparameter Tuning

**Tuned Hyperparameters:**
- `n_estimators`: Number of trees in the forest tested over [100, 200, 300, 400].
- `max_features`: Number of features to consider when looking for the best split tested over [‘auto’, ‘sqrt’].

**Optimization Method:** Grid Search with 5-fold cross-validation.

**Best Hyperparameters:**
- `n_estimators`: 300
- `max_features`: auto

**Cross-Validation Results**
- Average Accuracy: 0.84
- Best Accuracy Score: 0.87

## FlauBERT Model

### Hyperparameter Tuning

**Tuned Hyperparameters:**
- `learning_rate`: Tested over [5e-5, 3e-5, 2e-5].
- `num_train_epochs`: Tested [3, 4, 5].

**Optimization Method:** Manual Iterative Search.

**Best Hyperparameters:**
- `learning_rate`: 3e-5
- `num_train_epochs`: 4

**Cross-Validation Results**
- Average Accuracy: 0.88
- Best Accuracy Score: 0.91

## Comparative Analysis

The performance of each model was evaluated based on precision, recall, F1-score, and accuracy. The following table summarizes the results:

| Model               | Precision | Recall | F1-score | Accuracy |
|---------------------|-----------|--------|----------|----------|
| Logistic Regression | 0.81      | 0.80   | 0.80     | 0.82     |
| kNN                 | 0.77      | 0.75   | 0.76     | 0.78     |
| Random Forests      | 0.74      | 0.73   | 0.73     | 0.76     |
| CamemBERT           | 0.83      | 0.82   | 0.82     | 0.84     |
| FlauBERT Model      | 0.87      | 0.88   | 0.87     | 0.88     |

### Best Model

Based on the above metrics, the best performing model is the **FlauBERT Model**. This determination is based on the highest overall accuracy and balanced precision and recall, which are critical factors for our specific use-case of text classification.


#### 4. Submission
- We use our best model on the cleaned sentences of file 'unlabelled_test_dat.csv', convert it to the same format as in the file 'sample_submission.csv', download in CSV format, and submit on Kaggle (it gave us a score of 0.45583).

#### 6. Model Improvement
- Many ways exist to improve the accuracy of a model. For this competition, we propose two other methods: The Principal Component Analysis (PCA), which for some unknown reason didn't improve our model. Afterward, we tried the Bert Model from HuggingFace which gave us a slight improvement. You can find them in the CODE folder 'Lausanne_PCA.ipynb', 'Lausanne_bert.ipynb'.


(https://www.kaggle.com/competitions/detecting-french-texts-difficulty-level-2023/overview)
