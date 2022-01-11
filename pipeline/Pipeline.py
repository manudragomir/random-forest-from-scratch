import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Pipeline:
    def run_all_preliminary_steps(self, filename):
        self.read(filename)
        self.preprocess()
        return self.prepare_training()

    def read(self, filename):
        self.df = pd.read_csv(filename)
        self.classes = self.df['Kingdom'].unique()
        logger.info(self.df.head())
        logger.info(len(self.classes))
        logger.info(self.classes)
        logger.info(self.df['Kingdom'].value_counts())

    def preprocess(self, n=None):
        self.df = self.df[(self.df['Kingdom'] != 'arc') & (self.df['Kingdom'] != 'plm') & (self.df['Kingdom'] != 'phg')]
        self.df = self.df.replace({'pri': 'mam'})
        self.df = self.df.replace({'rod': 'mam'})
        self.df = self.df.drop(['DNAtype', 'SpeciesID', 'Ncodons', 'SpeciesName'], axis=1)
        self.classes = self.df['Kingdom'].unique()

        if n is not None:
            self.df = self.df[:n]

        logger.info(self.df.head())
        logger.info(self.df['Kingdom'].value_counts())
        logger.info(self.classes)

    def prepare_training(self):
        class_encoder = LabelEncoder()
        y = class_encoder.fit_transform(self.df['Kingdom'])
        assert len(np.unique(y)) == len(self.df['Kingdom'].value_counts())
        X = self.df.iloc[:, 1:].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        return X_train, X_test, y_train, y_test
