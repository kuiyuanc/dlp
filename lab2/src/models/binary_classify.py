from torch.nn import Module, Sequential, Sigmoid


# TODO: extend to multiclass
class BinaryClassify2d(Module):
    def __init__(self, codec):
        super(BinaryClassify2d, self).__init__()

        self.nn = Sequential(codec, Sigmoid())
        self.name = codec.name if hasattr(codec, "name") else codec.__class__.__name__

    def forward(self, x):
        return self.nn(x)

class BinaryClassify2dOld(Module):
    def __init__(self, codec):
        super(BinaryClassify2dOld, self).__init__()

        self.codec = codec
        self.sigmoid = Sigmoid()
        self.name = codec.name if hasattr(codec, "name") else codec.__class__.__name__

    def forward(self, x):
        return self.sigmoid(self.codec(x))
