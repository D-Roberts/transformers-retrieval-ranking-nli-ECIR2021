
from mfactcheck.multi_nli.train import Trainer
class MNLIPredictor:
    def __init__(self, args):
        self.model = Trainer(model=model, args=args)

    def predict(self, payload):
        self.model.predict()