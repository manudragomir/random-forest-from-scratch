import logging

import pandas as pd

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Pipeline:
    def __init__(self):
        pass

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
