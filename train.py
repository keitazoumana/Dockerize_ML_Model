import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import sklearn

print(sklearn.__version__)
print(joblib.__version__)
print(pd.__version__)

if __name__ == "__main__":

    # 1. Load the cleaned data
    DATA_PATH = "./data/"
    FILE_NAME = "cleaned_data.csv"

    spam_data = pd.read_csv(DATA_PATH+FILE_NAME)


    X = spam_data['text']
    y = spam_data['target']
    ran_state = 2023
    t_size = 0.2 


    cv = CountVectorizer()
    X = cv.fit_transform(X.values.astype('U'))

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=t_size, 
                                                        random_state=ran_state, 
                                                        stratify = y) # To make sure "y" target is evenly distributed
                                                                    # Across training and testing data

    svm_model = LinearSVC()
    svm_model.fit(X_train, y_train)
    y_svm_model_predict = svm_model.predict(X_test)


    MODEL_FILE_NAME = "svm_best_model.joblib"
    VECTOR_FILE_NAME = "data_vectorizer.joblib"
    MODEL_FOLDER = "./models/"
    VECTOR_FOLDER = "./vectors/"

    # Serialize the model and the vectorizer
    joblib.dump(svm_model, MODEL_FOLDER+MODEL_FILE_NAME)
    joblib.dump(cv, VECTOR_FOLDER+VECTOR_FILE_NAME)