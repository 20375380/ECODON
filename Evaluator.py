import torch

class BasicEvaluator():
    def __init__(self, config, data, model):
        self.model = model
        self.data = data

    def evaluate(self):
        self.model.eval()

