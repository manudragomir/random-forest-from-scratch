import logging
import pickle
from datetime import datetime

from ensembles.RandomForest import RandomForest
from decision_trees.DecisionTree import DecisionTree
from pipeline.Pipeline import Pipeline
from utils.metrics import evaluate

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ConsoleUI:
    def __init__(self):
        self.__rf = None
        self.__dt = None
        self.__MENU = "Insert command:\n" \
                      "0 - Exit\n" \
                      "1 - Train Random Forest\n" \
                      "2 - Train Decision Tree\n"

    def __train_random_forest_ui(self):
        input_file = input('Input file=')

        print("Before training, choose hyperparams. ")
        n_estimators = int(input('n_estimators='))
        max_depth = int(input('max_depth='))

        X_train, X_test, y_train, y_test = Pipeline().run_all_preliminary_steps(input_file)
        rf = RandomForest(n_estimators=n_estimators, max_depth=max_depth)

        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)

        model_name = 'rf_' + datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
        logger.info(f'Performance model={model_name} is acc={rf.score(X_test, y_test)} and '
                    f'time={rf.training_time}')

        filename = model_name + '_performances.txt'
        with open(filename, 'w') as writer:
            writer.write(str(rf.analysis(y_true=y_test, y_pred=y_pred)))

        print(f'Performances were saved for current model={model_name} at file={filename}')
        self.__rf = rf

    def __train_decision_tree_ui(self):
        input_file = input('Input file=')

        print("Before training, choose hyperparams. ")
        max_depth = int(input('max_depth='))
        max_thresholds = int(input('(default=10) max_thresholds='))

        X_train, X_test, y_train, y_test = Pipeline().run_all_preliminary_steps(input_file)
        dt = DecisionTree(max_depth=max_depth, max_thresholds=max_thresholds)
        dt.fit(X_train, y_train)

        y_pred = dt.predict(X_test)

        model_name = 'dt_' + datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
        logger.info(f'Performance model={model_name} is acc={dt.score(X_test, y_test)} and '
                    f'time={dt.training_time}')

        filename = model_name + '_performances.txt'
        with open(filename, 'w') as writer:
            writer.write(str(dt.analysis(y_true=y_test, y_pred=y_pred)))

        print(f'Performances were saved for current model={model_name} at file={filename}')
        self.__dt = dt

    def run_app(self):
        commands = {
            1: self.__train_random_forest_ui,
            2: self.__train_decision_tree_ui,
        }

        while True:
            command = int(input(self.__MENU))
            if command == 0:
                print("BYE!")
                return
            commands[command]()


if __name__ == '__main__':
    console = ConsoleUI()
    console.run_app()
