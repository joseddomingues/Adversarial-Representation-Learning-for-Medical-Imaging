import torch.nn as nn


class BreastClassifier(nn.Module):
    def __init__(self):
        super(BreastClassifier, self).__init__()

        activation = "relu"
        self.cb1 = ConvolutionBlock(3, 32, activation)
        self.cb2 = ConvolutionBlock(32, 64, activation)
        self.cb3 = ConvolutionBlock(64, 128, activation)
        self.cb4 = ConvolutionBlock(128, 256, activation)
        self.cb5 = ConvolutionBlock(256, 512, activation)
        self.cb6 = ConvolutionBlock(512, 1024, activation)

        activation2 = "relu"
        self.fcb1 = FullConvolutionBlock(1024 * 5 * 5, 512, activation2)
        self.fcb2 = FullConvolutionBlock(512, 256, activation2)
        self.fcb3 = FullConvolutionBlock(256, 128, activation2)
        self.fcb4 = FullConvolutionBlock(128, 64, activation2)
        self.fcb5 = FullConvolutionBlock(64, 32, activation2)
        self.last = nn.Linear(32, 3)

    def forward(self, x):
        x = self.cb1(x)
        x = self.cb2(x)
        x = self.cb3(x)
        x = self.cb4(x)
        x = self.cb5(x)
        x = self.cb6(x)

        x = self.fcb1(x)
        x = self.fcb2(x)
        x = self.fcb3(x)
        x = self.fcb4(x)
        x = self.fcb5(x)

        x = self.last(x)
        return x


class FullConvolutionBlock(nn.Module):
    def __init__(self, inf, outf, fnct):
        super(FullConvolutionBlock, self).__init__()
        self.dense = nn.Linear(inf, outf)
        self.dp = nn.Dropout(0.2)
        self.act_fnc = get_activation_function(fnct)

    def forward(self, x):
        x = self.dense(x)
        x = self.dp(x)
        x = self.act_fnc(x)
        return x


class ConvolutionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, fnct):
        super(ConvolutionBlock, self).__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, padding="same")
        self.act_func = get_activation_function(fnct)
        self.batch_norm = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.act_func(x)
        x = self.batch_norm(x)
        x = self.pool(x)
        return x


def get_activation_function(fnct):
    """
    Gets the activation function that matches the given word
    @param fnct: Function name
    @return: The activation function
    """
    functions_map = {"lrelu": nn.LeakyReLU(0.3, inplace=True),
                     "elu": nn.ELU(alpha=1.0, inplace=True),
                     "prelu": nn.PReLU(num_parameters=1, init=0.25),
                     "selu": nn.SELU(inplace=True),
                     "relu": nn.ReLU(inplace=True)
                     }
    return functions_map[fnct]
