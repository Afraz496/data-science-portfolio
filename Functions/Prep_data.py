#Function to prepare X_train, y_train, X_test, y_test
def data_prep(df,target_variable):
    features = df.columns.to_list()
    features.remove(target_variable)
    X = df[features]
    y = df[[target_variable]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
    return X_train, X_test, y_train, y_test