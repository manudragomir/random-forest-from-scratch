from pipeline.Pipeline import Pipeline
from ensembles.RandomForest import RandomForest


if __name__ == '__main__':
    pipeline = Pipeline()
    pipeline.read('codon_usage.csv')
    pipeline.preprocess()
    X_train, X_test, y_train, y_test = pipeline.prepare_training()

    rf = RandomForest(n_estimators=10,
                      criterion='gini',
                      max_depth=15,
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
