from Pipeline import Pipeline
from ensembles.RandomForest import RandomForest
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from decision_trees.DecisionTree import DecisionTree
import numpy as np


if __name__ == '__main__':
    pipeline = Pipeline()
    pipeline.read('codon_usage.csv')
    pipeline.preprocess()

    class_encoder = LabelEncoder()
    y = class_encoder.fit_transform(pipeline.df['Kingdom'])
    assert len(np.unique(y)) == len(pipeline.df['Kingdom'].value_counts())

    X = pipeline.df.iloc[:, 1:].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    rf = RandomForest(n_estimators=5,
                      criterion='gini',
                      max_depth=5,
                      min_samples_split=2,
                      min_samples_leaf=1,
                      n_jobs=-1)
    rf.fit(X_train, y_train)
    print(rf.score(X_test, y_test))
    # dt = DecisionTree(criterion='gini',
    #                   max_depth=4,
    #                   min_samples_split=2,
    #                   min_samples_leaf=1
    #                   )
    # dt.fit(X_train, y_train)
    # print(dt.score(X_test, y_test))
