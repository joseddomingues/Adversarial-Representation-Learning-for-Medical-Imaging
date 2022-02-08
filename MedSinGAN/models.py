import copy

import torch
import torch.nn as nn


def weights_init(model):
    """
    Initiates the models weight according to best practices
    @param model: Model to initiate the weights
    @return: Nothing
    """

    classname = model.__class__.__name__
    if classname.find('Conv2d') != -1:
        model.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)


def get_activation(opt):
    """
    Get the respective activation function according to the options map
    @param opt: Option map to get the respective activation function
    @return: The activation function according to the map
    """

    activations = {"lrelu": nn.LeakyReLU(opt.lrelu_alpha, inplace=True),
                   "elu": nn.ELU(alpha=1.0, inplace=True),
                   "prelu": nn.PReLU(num_parameters=1, init=0.25),
                   "selu": nn.SELU(inplace=True)
                   }
    return activations[opt.activation]


def upsample(x, size):
    """
    Upsamples the given pipeline_data_inbreast in a certain scale
    @param x: Data to upsample
    @param size: Scale to upsample in each measure
    @return: The upsampled pipeline_data_inbreast
    """

    x_up = torch.nn.functional.interpolate(x, size=size, mode='bicubic', align_corners=True)
    return x_up


class ConvBlock(nn.Sequential):
    """
    Convolution Block used in the model construction
    Combination of Sequential layers (Conv2d - BatchNormalization - Activation Function)
    """

    def __init__(self, in_channel, out_channel, ker_size, padd, opt, generator=False):
        """
        Constructor of the Conv Block
        @param in_channel: Number of in channels
        @param out_channel: Number of out channels
        @param ker_size: Kernel size
        @param padd: Padding to apply
        @param opt: Map of options
        @param generator: Boolean to add batch normalization layer
        """

        super(ConvBlock, self).__init__()
        self.add_module('conv',
                        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=ker_size, stride=1,
                                  padding=padd))
        if generator and opt.batch_norm:
            self.add_module('norm', nn.BatchNorm2d(out_channel))
        self.add_module(opt.activation, get_activation(opt))


class Discriminator(nn.Module):
    """
    Discriminator model that will evaluate the generator along the way
    """

    def __init__(self, opt):
        """
        Constructor of the Discriminator model
        @param opt: Option map
        """

        super(Discriminator, self).__init__()

        # Get the option map and the number of channels per conv layer
        self.opt = opt
        num_channels = int(opt.nfc)

        # HEAD
        self.head = ConvBlock(in_channel=opt.nc_im, out_channel=num_channels, ker_size=opt.ker_size, padd=opt.padd_size,
                              opt=opt)

        # BODY
        self.body = nn.Sequential()
        for i in range(opt.num_layer):
            block = ConvBlock(in_channel=num_channels, out_channel=num_channels, ker_size=opt.ker_size,
                              padd=opt.padd_size,
                              opt=opt)
            self.body.add_module('block%d' % i, block)

        # TAIL
        self.tail = nn.Conv2d(in_channels=num_channels, out_channels=1, kernel_size=opt.ker_size, padding=opt.padd_size)

    def forward(self, x):
        """
        Forward pass simply applies the layers by order
        @param x: Data to feed the network to classify the input
        @return: The classification made by the network
        """

        head = self.head(x)
        body = self.body(head)
        out = self.tail(body)
        return out


class GrowingGenerator(nn.Module):
    """
    Generator model for generating images from a single one
    """

    def __init__(self, opt):
        """
        Constructor of the Generator Model
        @param opt: Option map
        """

        super(GrowingGenerator, self).__init__()

        # Get the option map and the number of channels per conv layer
        self.opt = opt
        num_channels = int(opt.nfc)

        # Establish the padding layers and blocks
        self._pad = nn.ZeroPad2d(1)
        self._pad_block = nn.ZeroPad2d(opt.num_layer - 1) \
            if opt.train_mode == "generation" or opt.train_mode == "animation" \
            else nn.ZeroPad2d(opt.num_layer)

        # HEAD
        self.head = ConvBlock(in_channel=opt.nc_im, out_channel=num_channels, ker_size=opt.ker_size, padd=opt.padd_size,
                              opt=opt, generator=True)

        # BODY
        self.body = torch.nn.ModuleList([])
        _first_stage = nn.Sequential()
        for i in range(opt.num_layer):
            block = ConvBlock(in_channel=num_channels, out_channel=num_channels, ker_size=opt.ker_size,
                              padd=opt.padd_size, opt=opt, generator=True)
            _first_stage.add_module('block%d' % i, block)
        self.body.append(_first_stage)

        # TAIL
        self.tail = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=opt.nc_im, kernel_size=opt.ker_size,
                      padding=opt.padd_size),
            nn.Tanh())

    def init_next_stage(self):
        """
        Prepare the model for the next stage by adding a copy of the first stage to the end of the body
        @return: Nothing
        """

        self.body.append(copy.deepcopy(self.body[-1]))

    def forward(self, noise, real_shapes, noise_amp):
        """
        Forward pass that also applies upsampling
        @param noise: Noise that is the input to the network
        @param real_shapes: Shappes of the real images to calculate other noises
        @param noise_amp: The noise ampliation that should be done
        @return: The new generated output
        """

        # 1 - Apply a padding of 1 to some noise and then the head block
        x = self.head(self._pad(noise[0]))

        # we do some upsampling for training models for unconditional generation to increase
        # the image diversity at the edges of generated images
        # 2 - Upsample the noise, padd it with the pad block and then apply the first stage of the body
        if self.opt.train_mode == "generation" or self.opt.train_mode == "animation":
            x = upsample(x, size=[x.shape[2] + 2, x.shape[3] + 2])
        x = self._pad_block(x)
        x_prev_out = self.body[0](x)

        # 3 - For each block of the body apply upsampling and then the block
        for idx, block in enumerate(self.body[1:], 1):
            if self.opt.train_mode == "generation" or self.opt.train_mode == "animation":
                x_prev_out_1 = upsample(x_prev_out, size=[real_shapes[idx][2], real_shapes[idx][3]])
                x_prev_out_2 = upsample(x_prev_out, size=[real_shapes[idx][2] + self.opt.num_layer * 2,
                                                          real_shapes[idx][3] + self.opt.num_layer * 2])
                x_prev = block(x_prev_out_2 + noise[idx] * noise_amp[idx])
            else:
                x_prev_out_1 = upsample(x_prev_out, size=real_shapes[idx][2:])
                x_prev = block(self._pad_block(x_prev_out_1 + noise[idx] * noise_amp[idx]))
            x_prev_out = x_prev + x_prev_out_1

        # 4 - Apply the padding to the resulting block and then the tail
        out = self.tail(self._pad(x_prev_out))
        return out
