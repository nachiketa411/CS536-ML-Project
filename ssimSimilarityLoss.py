import nn


class SSIMSimilarityLoss(nn.Module):

    def __init__(self, learning_rate, num_labels):
        super().__init__()
        self.lr = learning_rate

        pass