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
- `sample_submission.csv` -> submission of our prediction with only id and difficulty

#### Columns
- `id`: Numerical identifier of the sentence.
- `sentence`: A sentence in French for which you want to predict the difficulty level.
- `difficulty`: The difficulty level of the sentence (from A1 to C2). This column would be your target variable.

We can upload them from the GitHub Data folder or you can use directly the Kaggle API and your Kaggle credentials.

#### Approach
1. **Installing the Necessary Packages**
   - For language support: `spacy`, `nltk`, `sentencepiece`,`transformers datasets pandas sickit-learn` etc...
   - For modelling: `pandas`, `numpy`, `sklearn`, `torch` etc...

2. **Models**
   - The Models we used in our analysis:
     - Logistic regression
     - KNN
     - Decision Tree
     - Random Forest models
     - CamemBERT
     - FlauBERT

#### Data Exploration and Cleaning

Before embarking on the model training journey, we invested significant effort into thoroughly exploring the dataset. This initial step is crucial as it informs subsequent decisions about data preprocessing and model selection.

## Key Observations:

- **Well-Distributed Values**: Our analysis revealed that the dataset was well-balanced across different difficulty levels. This balance is crucial in machine learning, especially for classification tasks, as it prevents model bias towards more frequent labels.

-**Distribution of Sentence Lengths by Difficulty Level** also our analysis, illustrated in the box plot, indicates a wide range of sentence lengths across different difficulty levels. Notably, the higher difficulty levels (B2 and C2) show a greater spread in sentence length, which may suggest a more complex sentence structure or vocabulary usage.

- **Quality of Data**: We examined the dataset for common issues such as missing values and duplicates. To our advantage, the dataset maintained a high standard of quality, with minimal anomalies that could potentially skew the model's performance.

- **Consistency in Labeling**: The labels for text difficulty followed a consistent pattern, reducing the need for label encoding or correction.
  
- **Textual Data Analysis**: Given that our dataset consists of textual data, we checked for language consistency, and the presence of special characters or noise. The texts were uniformly in French, with standard language use and minimal noise, which is ideal for natural language processing tasks.

## Decision on Data Cleaning:
Based on these observations, we concluded that extensive data cleaning or balancing was not necessary for our project. This finding allowed us to channel our efforts more effectively towards feature engineering and model tuning. 

####Model Training and Hyperparameter Optimization

This section documents the process of training several machine learning models, including the hyperparameter optimization for each, and provides a comparative analysis of their performance based on several metrics.

## Logistic Regression

### Hyperparameter Tuning

**Tuned Hyperparameters:**
- `C`: Inverse of regularization strength tested over [1, 10, 100].
- `solver`: Optimization algorithms tested [‘liblinear’, ‘saga’].

**Optimization Method:** Grid Search with 5-fold cross-validation.

**Best Hyperparameters:**
- `C`: 10
- `solver`: liblinear

**Results**
- Best Accuracy Score: 0.32

## k-Nearest Neighbors (kNN)

### Hyperparameter Tuning

**Tuned Hyperparameters:**
- `n_neighbors`: Number of neighbors tested over [5, 7, 9].
- `weights`: Weight function used in prediction [‘uniform’, ‘distance’].

**Optimization Method:** Random Search with 5-fold cross-validation.

**Best Hyperparameters:**
- `n_neighbors`: 5
- `weights`: distance

**Results**
- Best Accuracy Score: 0.31

## Decision Tree

### Hyperparameter Tuning

**Tuned Hyperparameters:**
- `max_depth`: The maximum depth of the tree tested over [20, 30, None].
- `min_samples_split`: Minimum number of samples required to split an internal node tested over [2, 5, 10].

**Optimization Method:** Grid Search with 5-fold cross-validation.

**Best Hyperparameters:**
- `max_depth`: 20
- `min_samples_split`: 5

**Results**

- Best Accuracy Score: 0.39

## Random Forests

### Hyperparameter Tuning

**Tuned Hyperparameters:**
- `n_estimators`: Number of trees in the forest tested over [200, 300, 400].
- `max_features`: Number of features to consider when looking for the best split tested over [‘auto’, ‘sqrt’].

**Optimization Method:** Grid Search with 5-fold cross-validation.

**Best Hyperparameters:**
- `n_estimators`: 300
- `max_features`: auto

**Results**
- Best Accuracy Score: 0.59

## CamemBERT Model

### Hyperparameter Tuning

**Tuned Hyperparameters:**
- `learning_rate`: Tested over [5e-5, 4e-5, 3e-5].
- `num_train_epochs`: Tested [3, 4, 5].
- `batch_size`: Tested [16, 32].

**Optimization Method:** AdamW.

**Best Hyperparameters:**
- `learning_rate`: 4e-5
- `num_train_epochs`: 3
- `batch_size`: 32

**Cross-Validation Results**

- Best Accuracy Score: 0.58

## FlauBERT Model

### Hyperparameter Tuning

**Tuned Hyperparameters:**
- `learning_rate`: Tested over [5e-5, 3e-5, 2e-5].
- `num_train_epochs`: Tested [3,5].

**Optimization Method:** Manual Iterative Search.

**Best Hyperparameters:**
- `learning_rate`: 3e-5
- `num_train_epochs`: 3

**Cross-Validation Results**
- Average Accuracy: 0.63
- Best Accuracy Score: 0.62

## Comparative Analysis

The performance of each model was evaluated based on precision, recall, F1-score, and accuracy. The following table summarizes the results without doing any cleaning on the data:

| Model               | Precision | Recall | F1-score | Accuracy |
|---------------------|-----------|--------|----------|----------|
| Logistic Regression | 0.42      | 0.45   | 0.40     | 0.32     |
| kNN                 | 0.47      | 0.48   | 0.46     | 0.38     |
| Decision Tree       | 0.48      | 0.49   | 0.46     | 0.39     |
| Random Forests      | 0.54      | 0.53   | 0.53     | 0.46     |
| CamemBERT           | 0.53      | 0.56   | 0.57     | 0.59     |
| FlauBERT Model      | 0.87      | 0.88   | 0.87     | 0.62     |

![Confusion Matrix de notre modele FlauBERT](chemin/vers/l'image)


# Pour utiliser l'image, vous pouvez utiliser le nom de fichier :
# image = open('nom_de_fichier.jpg', 'rb').read()

### Best Model

Based on the above metrics, the best performing model is the **FlauBERT Model**. This determination is based on the highest overall accuracy and balanced precision and recall, which are critical factors for our specific use-case of text classification.


#### 4. Submission
- We use our best model on the file 'unlabelled_test_dat.csv', convert it to the same format as in the file 'sample_submission.csv', download in CSV format, and submit on Kaggle (it gave us a score of 0.629).

#### 6. Model Improvement
To improve our model we consider these two approaches; 

-Fine-Tune Hyperparameters:
would conduct a more comprehensive hyperparameter optimization, particularly focusing on the learning rate, batch size, and the number of training epochs.
Utilize advanced techniques like Bayesian optimization for a more efficient search through the hyperparameter space, aiming to find the optimal combination that maximizes model performance.
Expand and Refine Preprocessing:

-Implement advanced text preprocessing techniques: 
This would include lemmatization to reduce words to their base or dictionary form, and custom tokenization to better capture the nuances of the French language.
Explore data augmentation strategies like synonym replacement or back-translation (translating text to another language and back) to generate additional training data, which can be particularly helpful in improving the model's robustness and handling of diverse sentence structures.


(https://www.kaggle.com/competitions/detecting-french-texts-difficulty-level-2023/overview)
