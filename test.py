import sys
print("pythonPath: ", sys.executable)
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
import autotabular as at

df = pd.read_csv("dataset/bank-additional-full.csv", sep=";")

trans_cols = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week",
              "poutcome", "y"]

for col in trans_cols:
    lbe = LabelEncoder()
    df[col] = lbe.fit_transform(df[col])

train_data, test_data, train_label, test_label = train_test_split(df.drop("y", axis=1), df["y"], test_size=0.3, random_state=1024)

trainset = pd.concat([train_data, train_label], axis=1)
testset = pd.concat([test_data, test_label], axis=1)

at.auto_train(train_set=trainset, label_name="y", test_set=testset)



