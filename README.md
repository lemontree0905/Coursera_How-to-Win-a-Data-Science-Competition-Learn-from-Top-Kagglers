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
  - RMSE(log(y_targ+1),log(y_pred+1))
  - Frequently better than MAPE, less biased towards small targets yet works with relatvie errors

### Regression metrics optimization


### Classification metrics
- Notation
  - Soft labes are clssifier's scores
  - Hard labels: arg max f(x), [f(x)>b] with b the threshold
- Accuracy, Logloss, AUC
  - Accuracy (Best constant: the most frequent class)
  - Logarithmic loss (Best constant: set alpha_i to frequency of i-th class)  
    - Binary:
    <p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=LogLoss&space;=&space;\frac{1}{N}\sum&space;y_i\text{log}(\hat{y}_i)&plus;(1-y_i)\text{log}(1-\hat{y}_i)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?LogLoss&space;=&space;\frac{1}{N}\sum&space;y_i\text{log}(\hat{y}_i)&plus;(1-y_i)\text{log}(1-\hat{y}_i)" title="LogLoss = \frac{1}{N}\sum y_i\text{log}(\hat{y}_i)+(1-y_i)\text{log}(1-\hat{y}_i)" /></a></p>
    - Multiclass:
     <p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=LogLoss&space;=&space;\frac{1}{N}\sum_i^N\sum_i^L&space;y_{il}\text{log}(\hat{y}_{il})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?LogLoss&space;=&space;\frac{1}{N}\sum_i^N\sum_i^L&space;y_{il}\text{log}(\hat{y}_{il})" title="LogLoss = \frac{1}{N}\sum_i^N\sum_i^L y_{il}\text{log}(\hat{y}_{il})" /></a></p>
     
  - Area Under Curve (AUC ROC)
    - Only for binary tasks
    - Depends only on ordering of the predictions, not on absolute values
     
- Cohen's (Quadratic weighted) Kappa
  - Cohen's Kappa = 1 - (1-accuracy)/(1-p_e)
  -  p_e = 1/N^2 sum_k(n_k1 * n_k2), what accuracy would be on average, if we randomly permute our predictions
  - 

### Classification metrics optimization
- Logloss
  - Tree-based
    XGBoost,LightGBM
    
    ~~RandomForestClassifier~~
  - Linear models (Regression,SGDRegressor)

  - How to calibrate
    - Platt scaling (just fit Logistic Regression to your prediction)
    - Isotonic regression (Just fit Isotonic Regression to your prediction)
    - Stacking (Just fit XGBoost or neural net to your predictions)
    
- Accuracy
  - Proxy loss: Logistic loss, Hinge loss
  
- AUC 
 - Tree-based
    XGBoost,LightGBM
    
    ~~RandomForestClassifier~~
 - Not working: Linear models (Regression,SGDRegressor)
 
- Quadratic weighted Kappa
  - Optimize MSE and find the right thresholds
  - Custom smooth loss for GBDT or neural nets
  par
  
### Advanced Feature Engineering I
#### Mean encoding (likelihood encoding, target encoding)
- Ways to calculate 

  Goods - number of ones in at group, Bads -  number of zeros

  - Likelihood = Goods/(Goods+Bads) = mean(target)
  - Weight of Evidenc = ln(Goods/Bads) * 100
  - Count = Goods = sum(target)
  - Diff =  Goods - Bads
 
#### Regularization

### Advanced Feature Engineering II



## Week4
### Hyperparameter Optimization

- Hyperparameter optimization software
  - Hyperopt
  - Scikit-optimize
  - Spearmint
  - GPyOpt
  - RoBO
  - SMAC3
  
- Notation
  - *para*: the larger, the powerful the model (decrease if overfit)
  - para: the larger, the heavier the constraint (increase if overfit)
  
- Tree-based models: GBDT(XGBoost,LightGBM,CatBoost), RandomForest/ExtraTrees, Others(RGF)
  
  |XGBoost| LightGBM
  | --- | --- |
  | *max_depth* | *max_depth/num_leaves* | 
  | *subsample* |	*bagging_fraction*
  | *colsample_bytree*, *colsample_bylevel* | *feature_fraction*
  | **min_child_weight**, lambda, alpha | **min_data_in_leaf**, lambda_l1, lambda_l2
  | *eta* | *learning rate*
  | *num_round* | *num_iterations*

  - Note
    - If you increase *max_depth* and can not get the model to overfit, it can be a sign that there are a lot of important    interactions to extract.
    - Freeze eta, find the number of rounds when the model begin to overfit. Multiply the number of steps by a factor of alpha and divide eta by the factor of alpha
    
  | RandomForest/ExtraTrees |
  | --- |
  | ***N_estimators*** (the higher the better)|
  | *max_depth* (7 at first) |
  | *max_feature* |
  | min_samples_leaf |
  | ***criterion*** (GINI is often better) |
  
- Neural nets
  - what framework to use: Keras, Lasagne, TensorFlow, PyTorch, MxNet
  | Neural nets |
  | --- |
  | *Number of neurons per layer* |
  | *Number of layers* |
  | Optimizers : SGD + momentum; *Adam/Adadelta/Adagrad/*... |
  | *Batch size* |
  | ***Learning rate*** |
  | ***Regularization***: L2/L1 for weights, Dropout/Dropconnect, Static dropconnect
  - Note
    - If you increase *Batch size* by a factor of alpha, you can also increase ***Learning rate*** by the same factor
    - Whenever you see a network overfitting, try first a dropout layer
    
- Linear models
  - Scikit-learn
    - SVC/SVR (libLinear, libSVM)
    - LogisticRegression/LinearRegression + regularizers
    - SGDClassifier/SGDRegressor
  - Regularization parameter (C, alpha, lambda, ...)
    - start with very value and increase it
    - SVC starts to work slower as C increases

### Tips and tricks
- Practical guide
  - Data loading
    - Do basic preprocessing and convert csv/txt files into hdf5/npy
    - Downcast the data to 32-bits
  - Performance evaluation
    - Extensive validation is not always needed
    - Start with fasted models - LightGBM, find important features
