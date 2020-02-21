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
- Lowercase, Lemmatization, Stemming, Stopwords
- Bag of words 
  - very large vectors, meaning of each value in vector is known
  - sklearn.feature_extraction.text.CountVectorizer
  
    sklearn.feature_extraction.text.TfidfVectorizer
  - N-grams

- Embeddings 
  - relatively small vectors, values in vector can be interpreted only in some cases, words with similar meaning often have similar embeddings
  - Words: word2vec, Glove, FastText, etc
  - Sentences: Doc2Vec, etc
        
#### Image
