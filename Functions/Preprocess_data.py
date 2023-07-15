def fit_processor(train_df, target_variable):
"""
    Runs data_prep function to create X_train,X_test,y_train,y_test
    Applies Simple Imputer to Categorical Features
    Applies One Hot Encoding to Categorical Features
    Applies Quantile Scaling to Numeric Features
    Returns and writes pickle file of the complete preprocessor

    Parameters
    ----------
    X_train : numpy
        dataset in NumPy format
    Will create lists of numeric and categorical 
        variables using X_train df

    Returns
    -------
    preprocessor : sklearn.Preprocessor
        sklearn preprocessor fit on the training set
    """
    #Calling data prep function to create X_train,X_test,y_train,y_test
    X_train, X_test, y_train, y_test = data_prep(train_df,target_variable)

    features = X_train.columns.to_list()
    df_numeric_features = X_train.select_dtypes(include='number')
    df_categorical_features = X_train.select_dtypes(include='object')

    numeric_features = df_numeric_features.columns.to_list()
    categorical_features = df_categorical_features.columns.to_list()
    
    pipe_num = Pipeline([
        ('impute', SimpleImputer(strategy='mean')),
        ('scaler',  MinMaxScaler())
    ])
    pipe_cat = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]) 
    preprocessor = ColumnTransformer([
        ('num', pipe_num, numeric_features),
        ('cat', pipe_cat, categorical_features)
    ])

    norm_X_train = preprocessor.fit(X_train)

    return preprocessor,X_train, X_test, y_train, y_test

###USE FOLLOWING COMMANDS AFTER CALLING THIS FUNCTION
#processor,X_train, X_test, y_train, y_test = fit_processor(train_df,target_variable)
#norm_X_train = processor.transform(X_train)
#norm_X_test = processor.transform(X_test)