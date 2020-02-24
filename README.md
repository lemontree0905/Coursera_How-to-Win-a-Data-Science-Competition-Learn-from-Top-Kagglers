# Coursera_How-to-Win-a-Data-Science-Competition-Learn-from-Top-Kagglers

## Week1 
### Real World Application vs Competitions
- Understanding of business problem (what do you want to do? For what? How can it help your users?) 
- Problem formalization 
- Data collecting
- Data preprocessing
- Modelling 
- Way to evaluate model in real lift
- Way to deploy model (monitor model performance and re-train it on new data)

### Main ML algorithms
- Linear 
  - Good for **sparse high dimensional** data
  - Lasso/Ridge/Logistic Regression; SVM
- Tree-base method 
  - Good for tabular data
  - Hard to capture linear dependencies
  - Decision Tree, Random Forest, GBDT, XGBoost, LightGBM
- KNN-based methods
- Neural Networks
  - Good for **image**, **sounds**, **text**, **sequences**

### Feature Preprocessing
#### Numeric Feature
- Scaling (Tree-based models doesn't depend on scaling)
  - MinMaxScaler, StandardScaler, RobustScaler
  - Outliers(to protect Linear Model from outliers): winsorization
    - Upperbound, Lowerbound = np.percetile(x,[1,99])
    
      y = np.clip(x,Upperbound,Lowerbound)     
  - Rank (scipy.stas.rankdata): set spaces between sorted values to be equal
    - Linear Model, KNN, Neural Networks can benifit from this if we have outliers
    - Concatenate train and test data
  - Log transform & Raising to power <1 (useful for **Neural Networks**)
 
#### Categorical Feature
- Ordinal feature: values are sorted in some meaningful order

- Label encoding (useful for **tree-based** model)
  - Alphabetical: sklearn.preprocessing.LabelEncoder
  - Order of apperance: Pandas.factorize
  - Frequency encoding (useful for **linear** and **tree-based** models)
    - If there were ties, use rankdata first
    - encoding = titanic.groupty('Embarked').size()
    
      encoding = encoding/len(titanic)
      
      titanic['enc'] = titanic.Embarked.map(encoding)
- One-hot encoding (useful for **non-tree-based** model)
  - pandas.get_dummies, sklearn,preprocessing.OneHotEncoder
  - Sparse matrix
  
- Interaction of categorical features can help **Linear model** and **KNN**

#### Datetime and Coordinates
- Datetime
  - Periodicity
  - Time since row-independenct/dependent event
  - Difference between dates
- Coordinates
  - Interesting places from train/test data or additional data
  - Centers of clusters
  - Aggregated statisitcs
  
### Missing values
- np.isnan(), pd.isnull().sum()
- Fillna approaches
  - -999, mean, median
  - Reconstruct value
  
### Feature extraction from texts and images
#### Texts
- Preprocessing
  - Lowercase, Lemmatization, Stemming, Stopwords
- Bag of words 
  - very large vectors, meaning of each value in vector is known
  - TFiDF can be of use as postprocessing   
  
    sklearn.feature_extraction.text.CountVectorizer
  
    sklearn.feature_extraction.text.TfidfVectorizer
  - N-grams can help to use local context

- Embeddings 
  - relatively small vectors, values in vector can be interpreted only in some cases, words with similar meaning often have similar embeddings
  - Words: word2vec, Glove, FastText, etc
  - Sentences: Doc2Vec, etc
        
#### Image
- Features can be extracted from different layers (Descriptors)
- Careful shoosing of pretrained network can help
- Finetuning allows to refine pretrained models
- Data augmentation can improve the model

## Week2 
### Exploratory data analysis
- Buiding intution about the data
  - Getting domain knowledge
  - Checking if the data is intuitive
  - Understanding how the data  was generated
- Exploring anonymized data
  - Explore individual features
    - Guess the meaning&types of columns 
      df.dtypes, df.info(), df.describe(), x.value_counts(), x.isnull()
  - Explore feature relations
- Visualizations
  - Explore individual features
    - Histograms: plt.hist(x)
    - Plot (index versus value): plt.plot(x,'.'), plt.scatter(range(len(x)), x, c=y)
  - Explore feature relations
    - Pairs: plt.scatter(x1,x2), pd.scatter_matrix(df); df.corr(), plt.matshow()
    - Groups: df.mean().sort_values().plot(style='.')
- Clean features up
  - Duplicated and constant features
    
    trainset.nunique(axix=1) == 1
    
    df.T.drop_duplicates()
    
    for f in categorical_feats:
    
    traintest[f] = traintest[f].factorize()
    
    traintest.T.drop_duplicates()
  - Check if dataset is shuffled
  
### Validation
- Validation strategies
  - Holdout: ngroups = 1 
    - sklearn.model_selection.ShuffleSplit
  - K-fold: ngroups = k 
    - sklearn.model_selection.Kfold
  - Leave-one-out: ngroups = len(train)
    - may be useful if we have too little data
    - As a general rule, most authors, and empirical evidence, suggest that 5- or 10- fold cross validation should be preferred to LOO 
    - sklearn.model_selection.LeaveOneOut

- Stratification
  - preserve the same target distribution over different folds
  - useful for **small** datasets/**unbalanced** datasets/Multiclass classification

- Note

When you found the right hyper-parameters and want to get test predictions don't forget to retrain your model using all training data!!!

- Data splitting strategies
   - Random, rowwise ; Timewise (eg. Moving window);  By id (recommedation for new users)
   - Logic of feature generation depends on the data splitting strategy
   - Set up your validation to mimic the train/test split of the competition
   
- Causes of different scores and optimal parameters & What to do
  - Too little data & Too diverse and inconsistent data & different distributions in train and test
  - Average scores from different KFold splits & Tune model on one split, evaluate score on the other
  
## Week3
### Regression metrics
- MSE, RMSE, R-squared
  - MSE: Mean Square Error
  - RMSE: Root mean square error (a bit different from MSE in the case graiend based methods)
  - R-squared
- MAE: Mean Absolute Error (**Robust to outliers**)
  - Not that sensitive to outliers as MSE
  - Widely used in **finance**
  - Median of target values is optimal for MAE
- (R)MSPE, MAPE 
  - MSPE: Mean Square Percentage Error (best constant: weighted target mean)
  - MAPE: Mean Absolute Percentage Error (best constant: weighted target median)
- (R)MSLE
  - RMSLE: Root Mean Square Logarithmic Error 
  - RMSE(log(y+1),log(y_true+1))
  - Best constant: 
  
### Classification
- Accuracy, Logloss, AUC
- Cohen's (Quadratic weighted) Kappa









