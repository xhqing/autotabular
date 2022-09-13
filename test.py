
"""
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import autotabular as at
from sklearn.model_selection import train_test_split

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
"""

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
import autokeras as ak

print(tf.__version__)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Initialize the image classifier.
clf = ak.ImageClassifier(overwrite=True, max_trials=1)  # Try only 1 model.(Increase accordingly)
# Feed the image classifier with training data.
clf.fit(x_train, y_train, epochs=1)  # Change no of epochs to improve the model
# Export as a Keras Model.
model = clf.export_model()

print(type(model))  # <class 'tensorflow.python.keras.engine.training.Model'>

try:
    model.save("model_autokeras", save_format="tf")
except Exception:
    model.save("model_autokeras.h5")

loaded_model = load_model("model_autokeras", custom_objects=ak.CUSTOM_OBJECTS)

predicted_y = loaded_model.predict(tf.expand_dims(x_test, -1))
print(predicted_y)



