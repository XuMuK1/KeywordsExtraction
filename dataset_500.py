import os
import re

class DataReader(object):
    """Helper class that provides dataset loading utilities in usual ML fashion."""
    
    """
        How to use:
        
        dataset = DataReader().fetch_data()
        x, y = dataset.train_data
        x[0], y[0] #first text and keywords for it
    """

    def __init__(self):
        self.data_path = "data/500N-KPCrowd-v1.1/CorpusAndCrowdsourcingAnnotations"
        self.train_dir_name = "train"
        self.test_dir_name = "test"
        self.train_data = None
        self.test_data = None
        
    def load_data(self, data_dir):
        texts = []
        answers = []
        
        working_path = os.path.join(os.path.relpath(self.data_path), data_dir)
        directory = os.fsencode(working_path)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".txt") and not filename.endswith("-justTitle.txt"): 
                with open(os.path.join(working_path, filename), 'r', encoding="utf8") as file:
                    texts.append(file.read())
                    
                filename = re.sub('\.txt$', '', filename)
                filename += ".key"
                
                with open(os.path.join(working_path, filename), 'r', encoding="utf8") as file:
                    answers.append(file.read().split('\n'))

        return texts, answers

    def fetch_data(self, logging = True):
        print("Data loading started...")
        self.train_data = self.load_data(self.train_dir_name)
        #self.test_data = self.load_data(self.test_dir_name) since there are no actually test texts, only titles
        print("data has been loaded!")
        return self
