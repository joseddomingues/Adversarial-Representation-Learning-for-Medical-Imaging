import argparse


def get_arguments():
    """
    Obtains the arguments of the parser
    @return: An map of options
    """

    parser = argparse.ArgumentParser()

    # workspace:
    parser.add_argument('--not_cuda', action='store_true', help='disables cuda', default=0)
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--experiment_name', help='Experiment Name For MLFlow', type=str, default='Experiment_1')

    # stage hyper parameters:
    parser.add_argument('--nfc', type=int, help='number of filters (channels) per conv layer', default=64)
    parser.add_argument('--ker_size', type=int, help='kernel size', default=3)
    parser.add_argument('--num_layer', type=int, help='number of layers per stage', default=3)
    parser.add_argument('--padd_size', type=int, help='net pad size', default=0)

    # pyramid parameters:
    parser.add_argument('--nc_im', type=int, help='image # channels', default=3)
    parser.add_argument('--noise_amp', type=float, help='additive noise cont weight', default=0.1)
    parser.add_argument('--im_min_size', type=int, help='image minimal size at the coarser scale', default=25)
    parser.add_argument('--im_max_size', type=int, help='image max size at the coarser scale', default=250)
    parser.add_argument('--train_depth', type=int, help='how many layers are trained if growing', default=3)
    parser.add_argument('--start_scale', type=int, help='at which stage to start training', default=0)

    # optimization hyper parameters:
    parser.add_argument('--niter', type=int, help='number of epochs to train per scale', default=2000)
    parser.add_argument('--gamma', type=float, help='scheduler gamma', default=0.1)
    parser.add_argument('--lr_g', type=float, help='learning rate, default=0.0005', default=0.0005)
    parser.add_argument('--lr_d', type=float, help='learning rate, default=0.0005', default=0.0005)
    parser.add_argument('--beta1', type=float, help='beta1 for ADAM. default=0.5', default=0.5)
    parser.add_argument('--Gsteps', type=int, help='Generator inner steps', default=3)
    parser.add_argument('--Dsteps', type=int, help='Discriminator inner steps', default=3)
    parser.add_argument('--lambda_grad', type=float, help='gradient penalty weight', default=0.1)
    parser.add_argument('--alpha', type=float, help='reconstruction loss weight', default=10)
    parser.add_argument('--activation', help="activation function {lrelu, prelu, elu, selu}", default='lrelu')
    parser.add_argument('--lrelu_alpha', type=float, help='alpha for leaky relu', default=0.05)
    parser.add_argument('--batch_norm', action='store_true', help='use batch norm in generator', default=0)

    return parser
