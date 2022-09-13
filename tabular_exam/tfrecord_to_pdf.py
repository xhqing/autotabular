import tensorflow as tf
from tqdm import  tqdm
import numpy as np
from autogbm.auto_ingestion.dataset import AutoDLDataset
from tensorflow.python.keras.backend import set_session

config = tf.ConfigProto()
config.log_device_placement = False
sess = tf.Session(config=config)
set_session(sess)

def to_numpy(metadata, dataset):
    
    X = []
    Y = []

    dataset = dataset.batch(1)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    sample_count = metadata.sample_count
    for i in tqdm(range(sample_count), total=sample_count):
        try:
            example, labels = sess.run(next_element)
            X.extend(example)
            Y.extend(labels)
        except tf.errors.OutOfRangeError:
            break

    X, y = np.asarray(X), np.asarray(Y)

    return X[:, 0, 0, :, 0], y

if __name__ == "__main__":
    D_train = AutoDLDataset("/app/codalab/autodl_challenge_2019_tabular_dataset/Madeline/madeline.data/train")
    x_train, y_train = to_numpy(D_train.get_metadata().metadata_, D_train.get_dataset())
    print(x_train.shape, y_train.shape)

    D_test = AutoDLDataset("/app/codalab/autodl_challenge_2019_tabular_dataset/Madeline/madeline.data/test")
    x_test, y_test = to_numpy(D_test.get_metadata().metadata_, D_test.get_dataset())
    print(x_test.shape, y_test.shape)
