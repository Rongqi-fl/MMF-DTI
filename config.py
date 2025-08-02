

class hyperparameter():
    def __init__(self):
        self.Learning_rate = 1e-4
        self.Epoch = 200
        self.Batch_size = 16
        self.Patience = 50
        self.decay_interval = 10
        self.lr_decay = 0.5
        self.weight_decay = 1e-4
        self.loss_epsilon = 1
        self.compound_structure_dim = 78
        self.sequence_structure_dim = 54
        self.out_dim = 160
        self.compound_llm_dim = 384
        self.sequence_llm_dim = 640
        self.numlayers = 3
        self.drop = 0.1

