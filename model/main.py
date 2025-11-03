import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# use pickle to 
import pickle

# cleaning the data
def get_claen_data():
    data = pd.read_csv("../data/data.csv")
    # drop unnamed and id columns
    data = data.drop(["Unnamed: 32", 'id'], axis=1)

    #Diagnosis (M = malignant, B = benign)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    return data

# use the LogicRegression to create the model
def create_model(data):
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    #  use standard scaler to make the data to be on the same scale
    scaler = StandardScaler()
    # scale the data
    X = scaler.fit_transform(X)

    # split the data into train and test datasets (80%:20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # test the model
    y_pred = model.predict(X_test)
    print("Accuracy of our model: ", accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))

    return model, scaler   


# main model
def main():
    # clean the data
    data = get_claen_data()

    # create the model
    model, scaler = create_model(data)

    # Export the model in the binary file in our model folder to import in our application
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f )


if __name__== "__main__":
    main()