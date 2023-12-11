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
- `sample_submission.csv` -> a sample submission file in the correct format (with just difficulty)

#### Columns
- `id`: Numerical identifier of the sentence.
- `sentence`: A sentence in French for which you want to predict the difficulty level.
- `difficulty`: The difficulty level of the sentence (from A1 to C2). This column would be your target variable.

We can upload them from the GitHub Data folder or you can use directly the Kaggle API and your Kaggle credentials.

#### Approach
1. **Installing the Necessary Packages**
   - For language support: `spacy`, `nltk`
   - For modelling: `pandas`, `numpy`, `sklearn`, etc...

2. **Models**
   - The Models we used in our analysis:
     - Logistic regression
     - KNN
     - Decision tree
     - Random Forest models.

3. **Model without Cleaning**
   - We implement the models shown in class without cleaning. We use a Tf-idf vectorizer and do hyperparameter tuning to find the best hyperparameters and check the results obtained.

4. **Model with Cleaning Data**
   - This time we clean data by:
     - Removing punctuation
     - Removing stop words
     - Tokenization
     - Lemmatization
     - For vectorization, we tried different approaches to see the best combination which led us not to remove the stop words anymore since some sentences are short.

#### Results without Data Cleaning:
| Models                    | Precision | First Header | Second Header | Third Header |
|---------------------------|-----------|--------------|---------------|--------------|
| Logistic Regression       | 0.475666  | 0.479224     | 0.474447      | 0.478125     |
| Random Forest             | 0.416684  | 0.416253     | 0.401491      | 0.413542     |
| Decision Tree Improvement | 0.314862  | 0.315516     | 0.311864      | 0.315625     |
| Decision Tree             | 0.301306  | 0.300758     | 0.298111      | 0.301042     |
| KNN Model                 | 0.419700  | 0.354327     | 0.345031      | 0.354167     |

#### Results with Data Cleaning:
| Models                | Precision | First Header | Second Header | Third Header |
|-----------------------|-----------|--------------|---------------|--------------|
| Logistic Regression   | 0.501281  | 0.504158     | 0.499639      | 0.503125     |
| Random Forest         | 0.437922  | 0.420731     | 0.400117      | 0.419792     |
| Decision Tree         | 0.323231  | 0.315516     | 0.323665      | 0.327083     |
| KNN Model             | 0.412903  | 0.418594     | 0.408406      | 0.418750     |

#### 4. Submission
- We use our best model on the cleaned sentences of file 'unlabelled_test_dat.csv', convert it to the same format as in the file 'sample_submission.csv', download in CSV format, and submit on Kaggle (it gave us a score of 0.45583).

#### 6. Model Improvement
- Many ways exist to improve the accuracy of a model. For this competition, we propose two other methods: The Principal Component Analysis (PCA), which for some unknown reason didn't improve our model. Afterward, we tried the Bert Model from HuggingFace which gave us a slight improvement. You can find them in the CODE folder 'Lausanne_PCA.ipynb', 'Lausanne_bert.ipynb'.


(https://www.kaggle.com/competitions/detecting-french-texts-difficulty-level-2023/overview)
