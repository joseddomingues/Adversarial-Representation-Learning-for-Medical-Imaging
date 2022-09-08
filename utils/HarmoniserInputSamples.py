import copy
import datetime
import math
import os
import random
from math import pi

import dateutil.tz
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from albumentations import HueSaturationValue, GaussNoise, OneOf, \
    Compose
from albumentations.augmentations.transforms import ChannelShuffle, Cutout, InvertImg, ToSepia, MultiplicativeNoise, \
    ChannelDropout
from scipy.ndimage import filters, measurements, interpolation
from skimage import color
from skimage import io as img
from skimage import morphology, filters
from torch.cuda.amp import autocast


def denorm(x):
    """

    @param x:
    @return:
    """

    out = (x + 1) / 2
    return out.clamp(0, 1)


def norm(x):
    """

    @param x:
    @return:
    """

    out = (x - 0.5) * 2
    return out.clamp(-1, 1)


def convert_image_np(inp):
    """

    @param inp:
    @return:
    """

    if inp.shape[1] == 3:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1, :, :, :])
        inp = inp.numpy().transpose((1, 2, 0))
    else:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1, -1, :, :])
        inp = inp.numpy().transpose((0, 1))

    inp = np.clip(inp, 0, 1)
    return inp


def generate_noise(size, num_samp=1, device='cuda', type='gaussian', scale=1):
    """

    @param size:
    @param num_samp:
    @param device:
    @param type:
    @param scale:
    @return:
    """

    if type == 'gaussian':
        noise = torch.randn(num_samp, size[0], round(size[1] / scale), round(size[2] / scale), device=device)
        noise = upsampling(noise, size[1], size[2])

    elif type == 'gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device) + 5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise = noise1 + noise2

    elif type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)

    else:
        raise NotImplementedError
    return noise


def upsampling(im, sx, sy):
    """

    @param im:
    @param sx:
    @param sy:
    @return:
    """

    m = nn.Upsample(size=[round(sx), round(sy)], mode='bilinear', align_corners=True)
    return m(im)


def move_to_gpu(t):
    """

    @param t:
    @return:
    """

    if torch.cuda.is_available():
        t = t.to(torch.device('cuda'))
    return t


def move_to_cpu(t):
    """

    @param t:
    @return:
    """

    t = t.to(torch.device('cpu'))
    return t


def save_image(name, image):
    """

    @param name:
    @param image:
    @return:
    """
    plt.imsave(name, convert_image_np(image), vmin=0, vmax=1)


def sample_random_noise(depth, reals_shapes, opt):
    """

    @param depth:
    @param reals_shapes:
    @param opt:
    @return:
    """

    noise = []
    for d in range(depth + 1):
        if d == 0:
            noise.append(generate_noise([opt.nc_im, reals_shapes[d][2], reals_shapes[d][3]],
                                        device=opt.device).detach())
        else:
            if opt.train_mode == "generation" or opt.train_mode == "animation":
                noise.append(generate_noise([opt.nfc, reals_shapes[d][2] + opt.num_layer * 2,
                                             reals_shapes[d][3] + opt.num_layer * 2],
                                            device=opt.device).detach())
            else:
                noise.append(generate_noise([opt.nfc, reals_shapes[d][2], reals_shapes[d][3]],
                                            device=opt.device).detach())

    return noise


def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device, given_scaler):
    """

    @param netD:
    @param real_data:
    @param fake_data:
    @param LAMBDA:
    @param device:
    @param scaler: Scaler to improve gradient calculation. Gets faster performance with half precision
    @return:
    """

    MSGGan = False
    if MSGGan:
        alpha = torch.rand(1, 1)
        alpha = alpha.to(device)  # cuda() #gpu) #if use_cuda else alpha

        interpolates = [alpha * rd + ((1 - alpha) * fd) for rd, fd in zip(real_data, fake_data)]
        interpolates = [i.to(device) for i in interpolates]
        interpolates = [torch.autograd.Variable(i, requires_grad=True) for i in interpolates]

        disc_interpolates = netD(interpolates)

    else:
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(device)  # cuda() #gpu) #if use_cuda else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.to(device)  # .cuda()
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        with autocast():
            disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=given_scaler.scale(disc_interpolates), inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                    # .cuda(), #if use_cuda else torch.ones(
                                    # disc_interpolates.size()),
                                    create_graph=True, retain_graph=True, only_inputs=True)  # [0]

    inv_scale = 1. / given_scaler.get_scale()
    gradients = [p * inv_scale for p in gradients]
    gradients = gradients[0]

    # LAMBDA = 1
    with autocast():
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

    del interpolates
    del gradients

    return gradient_penalty


def read_image(image_name, nc_im, not_cuda):
    """

    @param opt:
    @return:
    """
    x = img.imread(image_name)
    x = np2torch(x, nc_im, not_cuda)
    x = x[:, 0:3, :, :]
    return x


def read_image_dir(dir, nc_im, not_cuda):
    """

    @param dir:
    @param opt:
    @return:
    """

    x = img.imread(dir)
    x = np2torch(x, nc_im, not_cuda)
    x = x[:, 0:3, :, :]
    return x


def np2torch(x, nc_im, not_cuda):
    """

    @param x:
    @param opt:
    @return:
    """

    if nc_im == 3:
        x = x[:, :, :, None]
        x = x.transpose((3, 2, 0, 1)) / 255
    else:
        x = color.rgb2gray(x)
        x = x[:, :, None, None]
        x = x.transpose(3, 2, 0, 1)
    x = torch.from_numpy(x)
    if not not_cuda:
        x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor) if not not_cuda else x.type(torch.FloatTensor)
    x = norm(x)
    return x


def torch2uint8(x):
    """

    @param x:
    @return:
    """

    x = x[0, :, :, :]
    x = x.permute((1, 2, 0))
    x = 255 * denorm(x)
    x = x.cpu().numpy()
    x = x.astype(np.uint8)
    return x


def read_image2np(opt):
    """

    @param opt:
    @return:
    """

    x = img.imread('%s' % opt.input_name)
    x = x[:, :, 0:3]
    return x


def save_networks(netG, netDs, z, opt, scaler):
    """

    @param netG:
    @param netDs:
    @param z:
    @param opt:
    @param scaler: scaler to save that improves performance
    @return:
    """

    torch.save(netG.state_dict(), '%s/netG.pth' % opt.outf)
    if isinstance(netDs, list):
        for i, netD in enumerate(netDs):
            torch.save(netD.state_dict(), '%s/netD_%s.pth' % (opt.outf, str(i)))
    else:
        torch.save(netDs.state_dict(), '%s/netD.pth' % opt.outf)
    torch.save(z, '%s/z_opt.pth' % opt.outf)

    if scaler:
        torch.save(scaler, '%s/scaler.pth' % opt.outf)


def adjust_scales2image(real_, nc_im, not_cuda, im_max_size, train_stages, im_min_size):
    """

    @param real_:
    @param opt:
    @return:
    """

    scale1 = min(im_max_size / max([real_.shape[2], real_.shape[3]]), 1)
    real = imresize(real_, scale1, nc_im, not_cuda)

    stop_scale = train_stages - 1
    scale_factor = math.pow(im_min_size / (min(real.shape[2], real.shape[3])), 1 / stop_scale)

    return real, scale1, stop_scale, scale_factor


def create_reals_pyramid(real, train_mode, stop_scale, scale_factor, nc_im, not_cuda):
    """

    @param real:
    @param opt:
    @return:
    """

    reals = []
    # use old rescaling method for harmonization
    if train_mode == "harmonization":
        for i in range(stop_scale):
            scale = math.pow(scale_factor, stop_scale - i)
            curr_real = imresize(real, scale, nc_im=nc_im, not_cuda=not_cuda)
            reals.append(curr_real)
    # use new rescaling method for all other tasks
    else:
        for i in range(stop_scale):
            scale = math.pow(scale_factor,
                             ((stop_scale - 1) / math.log(stop_scale)) * math.log(stop_scale - i) + 1)
            curr_real = imresize(real, scale, nc_im=nc_im, not_cuda=not_cuda)
            reals.append(curr_real)
    reals.append(real)
    return reals


def load_trained_model(opt):
    """

    @param opt:
    @return:
    """

    dir = generate_dir2save(opt)

    if os.path.exists(dir):
        Gs = torch.load('%s/Gs.pth' % dir, map_location="cuda:{}".format(torch.cuda.current_device()))
        Zs = torch.load('%s/Zs.pth' % dir, map_location="cuda:{}".format(torch.cuda.current_device()))
        reals = torch.load('%s/reals.pth' % dir, map_location="cuda:{}".format(torch.cuda.current_device()))
        NoiseAmp = torch.load('%s/NoiseAmp.pth' % dir, map_location="cuda:{}".format(torch.cuda.current_device()))
        scaler = torch.load('%s/scaler.pth' % dir, map_location="cuda:{}".format(torch.cuda.current_device()))
    else:
        print('no trained model exists: {}'.format(dir))

    return Gs, Zs, reals, NoiseAmp, scaler


def generate_dir2save(opt):
    """

    @param opt:
    @return:
    """

    training_image_name = opt.input_name[:-4].split("/")[-1]
    dir2save = 'TrainedModels/{}/'.format(training_image_name)
    dir2save += opt.timestamp
    dir2save += "_{}".format(opt.train_mode)
    if opt.train_mode == "harmonization" or opt.train_mode == "editing":
        if opt.fine_tune:
            dir2save += "_{}".format("fine-tune")
    dir2save += "_niter_{}_lr_scale_{}_nstages_{}".format(opt.niter, opt.lr_scale, opt.train_stages)
    if opt.batch_norm:
        dir2save += "_BN"
    dir2save += "_act_" + opt.activation
    if opt.activation == "lrelu":
        dir2save += "_" + str(opt.lrelu_alpha)

    return dir2save


def post_config(opt):
    """

    @param opt:
    @return:
    """

    # init fixed parameters
    opt.device = torch.device("cpu" if opt.not_cuda else "cuda:{}".format(opt.gpu))
    opt.noise_amp_init = opt.noise_amp
    opt.timestamp = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if torch.cuda.is_available() and opt.not_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return opt


def load_config(opt):
    """

    @param opt:
    @return:
    """

    if not os.path.exists(opt.model_dir):
        print("Model not found: {}".format(opt.model_dir))
        exit()

    with open(os.path.join(opt.model_dir, 'parameters.txt'), 'r') as f:
        params = f.readlines()
        for param in params:
            param = param.split("-")
            param = [p.strip() for p in param]
            param_name = param[0]
            param_value = param[1]
            try:
                param_value = int(param_value)
            except ValueError:
                try:
                    param_value = float(param_value)
                except ValueError:
                    pass
            setattr(opt, param_name, param_value)
    return opt


def dilate_mask(mask, opt, nc_im, not_cuda):
    """

    @param mask:
    @param opt:
    @return:
    """

    if opt.train_mode == "harmonization":
        element = morphology.disk(radius=7)
    if opt.train_mode == "editing":
        element = morphology.disk(radius=20)
    mask = torch2uint8(mask)
    # For each channel
    new_mask = []
    mask1 = mask[:, :, 0]
    mask1 = morphology.binary_dilation(mask1, footprint=element)
    mask1 = filters.gaussian(mask1, sigma=5)
    new_mask.append(mask1)
    mask2 = mask[:, :, 1]
    mask2 = morphology.binary_dilation(mask2, footprint=element)
    mask2 = filters.gaussian(mask2, sigma=5)
    new_mask.append(mask2)
    mask3 = mask[:, :, 2]
    mask3 = morphology.binary_dilation(mask3, footprint=element)
    mask3 = filters.gaussian(mask3, sigma=5)
    new_mask.append(mask3)
    mask = np.array(new_mask)
    mask = mask.transpose(1, 2, 0)
    # End
    nc_im = opt.nc_im
    opt.nc_im = 1
    mask = np2torch(mask, nc_im, not_cuda)
    opt.nc_im = nc_im
    mask = mask.expand(1, 3, mask.shape[2], mask.shape[3])
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    return mask


def shuffle_grid(image, max_tiles=5):
    """

    @param image:
    @param max_tiles:
    @return:
    """
    tiles = []
    img_w, img_h = image.shape[0], image.shape[1]
    _max_tiles = random.randint(1, max_tiles)
    # _max_tiles = random.randint(3,3)
    if _max_tiles == 1:
        w_min, h_min = int(img_w * 0.2), int(img_h * 0.2)
        w_max, h_max = int(img_w * 0.5), int(img_h * 0.5)
        x_translation_min, y_translation_min = int(img_w * 0.05), int(img_h * 0.05)
        x_translation_max, y_translation_max = int(img_w * 0.15), int(img_h * 0.15)
    elif _max_tiles == 2:
        w_min, h_min = int(img_w * 0.15), int(img_h * 0.15)
        w_max, h_max = int(img_w * 0.3), int(img_h * 0.3)
        x_translation_min, y_translation_min = int(img_w * 0.05), int(img_h * 0.05)
        x_translation_max, y_translation_max = int(img_w * 0.1), int(img_h * 0.1)
    elif _max_tiles == 3:
        w_min, h_min = int(img_w * 0.1), int(img_h * 0.1)
        w_max, h_max = int(img_w * 0.2), int(img_h * 0.2)
        x_translation_min, y_translation_min = int(img_w * 0.05), int(img_h * 0.05)
        x_translation_max, y_translation_max = int(img_w * 0.1), int(img_h * 0.1)
    else:
        w_min, h_min = int(img_w * 0.1), int(img_h * 0.1)
        w_max, h_max = int(img_w * 0.15), int(img_h * 0.15)
        x_translation_min, y_translation_min = int(img_w * 0.05), int(img_h * 0.05)
        x_translation_max, y_translation_max = int(img_w * 0.1), int(img_h * 0.1)

    for _ in range(_max_tiles):
        x, y = random.randint(0, img_w), random.randint(0, img_h)
        w, h = random.randint(w_min, w_max), random.randint(h_min, h_max)
        if x + w >= img_w:
            w = img_w - x
        if y + h >= img_h:
            h = img_h - y
        x_t, y_t = random.randint(x_translation_min, x_translation_max), random.randint(y_translation_min,
                                                                                        y_translation_max)
        if random.random() < 0.5:
            x_t, y_t = -x_t, -y_t
            if x + x_t < 0:
                x_t = -x
            if y + y_t < 0:
                y_t = -y
        else:
            if x + x_t + w >= img_w:
                x_t = img_w - w - x
            if y + y_t + h >= img_h:
                y_t = img_h - h - y
        tiles.append([x, y, w, h, x + x_t, y + y_t])

    new_image = copy.deepcopy(image)
    for tile in tiles:
        x, y, w, h, x_new, y_new = tile
        new_image[x_new:x_new + w, y_new:y_new + h, :] = image[x:x + w, y:y + h, :]

    return new_image


class Augment:
    """

    """

    def __init__(self):
        super().__init__()
        self._transform = self.strong_aug()

    def strong_aug(self):
        """

        @return:
        """

        color_r = random.randint(0, 256)
        color_g = random.randint(0, 256)
        color_b = random.randint(0, 256)
        num_holes = random.randint(1, 2)
        if num_holes == 2:
            max_h_size = random.randint(15, 30)
            max_w_size = random.randint(15, 30)
        else:
            max_h_size = random.randint(30, 60)
            max_w_size = random.randint(30, 60)
        return Compose([
            OneOf([
                OneOf([
                    MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, per_channel=True, p=0.2),
                    GaussNoise()]),
                OneOf([
                    InvertImg(),
                    ToSepia()]),
                OneOf([
                    ChannelDropout(channel_drop_range=(1, 1), fill_value=0),
                    ChannelShuffle()]),
                HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.1)],
                p=0.25),
            Cutout(num_holes=num_holes, max_h_size=max_h_size, max_w_size=max_w_size,
                   fill_value=[color_r, color_g, color_b], p=0.9),
        ])

    def transform(self, **x):
        """

        @param x:
        @return:
        """

        _transform = self.strong_aug()
        return _transform(**x)


def generate_gif(dir2save, netG, fixed_noise, reals, noise_amp, opt, alpha=0.1, beta=0.9, start_scale=1,
                 num_images=100, fps=10):
    """

    @param dir2save:
    @param netG:
    @param fixed_noise:
    @param reals:
    @param noise_amp:
    @param opt:
    @param alpha:
    @param beta:
    @param start_scale:
    @param num_images:
    @param fps:
    @return:
    """

    def denorm_for_gif(img):
        """

        @param img:
        @return:
        """
        img = denorm(img).detach()
        img = img[0, :, :, :].cpu().numpy()
        img = img.transpose(1, 2, 0) * 255
        img = img.astype(np.uint8)
        return img

    reals_shapes = [r.shape for r in reals]
    all_images = []

    with torch.no_grad():
        noise_random = sample_random_noise(len(fixed_noise) - 1, reals_shapes, opt)
        z_prev1 = [0.99 * fixed_noise[i] + 0.01 * noise_random[i] for i in range(len(fixed_noise))]
        z_prev2 = fixed_noise
        for _ in range(num_images):
            noise_random = sample_random_noise(len(fixed_noise) - 1, reals_shapes, opt)
            diff_curr = [beta * (z_prev1[i] - z_prev2[i]) + (1 - beta) * noise_random[i] for i in
                         range(len(fixed_noise))]
            z_curr = [alpha * fixed_noise[i] + (1 - alpha) * (z_prev1[i] + diff_curr[i]) for i in
                      range(len(fixed_noise))]

            if start_scale > 0:
                z_curr = [fixed_noise[i] for i in range(start_scale)] + [z_curr[i] for i in
                                                                         range(start_scale, len(fixed_noise))]

            z_prev2 = z_prev1
            z_prev1 = z_curr

            sample = netG(z_curr, reals_shapes, noise_amp)
            sample = denorm_for_gif(sample)
            all_images.append(sample)
    imageio.mimsave('{}/start_scale={}_alpha={}_beta={}.gif'.format(dir2save, start_scale, alpha, beta), all_images,
                    fps=fps)


# [Batch Size, Channels (Depth), Height (Rows), Width (Columns)]
def np2torch(x, opt):
    """

    @param x:
    @param opt:
    @return:
    """

    if opt.nc_im == 3:
        x = x[:, :, :, None]
        x = x.transpose((3, 2, 0, 1)) / 255
    else:
        x = color.rgb2gray(x)
        x = x[:, :, None, None]
        x = x.transpose(3, 2, 0, 1)
    x = torch.from_numpy(x)
    if not opt.not_cuda:
        x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor) if not opt.not_cuda else x.type(torch.FloatTensor)
    # x = x.type(torch.cuda.FloatTensor)
    x = norm(x)
    return x


def torch2uint8(x):
    """

    @param x:
    @return:
    """

    x = x[0, :, :, :]
    x = x.permute((1, 2, 0))
    x = 255 * denorm(x)
    try:
        x = x.cpu().numpy()
    except:
        x = x.detach().cpu().numpy()
    x = x.astype(np.uint8)
    return x


def imresize(im, scale, nc_im, not_cuda):
    """

    @param im:
    @param scale:
    @param opt:
    @return:
    """

    im = torch2uint8(im)
    im = imresize_in(im, scale_factor=scale)
    im = np2torch(im, nc_im, not_cuda)
    return im


def imresize_to_shape(im, output_shape, nc_im, not_cuda):
    """

    @param im:
    @param output_shape:
    @param opt:
    @return:
    """

    im = torch2uint8(im)
    im = imresize_in(im, output_shape=output_shape)
    im = np2torch(im, nc_im, not_cuda)
    return im


def imresize_in(im, scale_factor=None, output_shape=None, kernel=None, antialiasing=True, kernel_shift_flag=False):
    """

    @param im:
    @param scale_factor:
    @param output_shape:
    @param kernel:
    @param antialiasing:
    @param kernel_shift_flag:
    @return:
    """

    # First standardize values and fill missing arguments (if needed) by deriving scale from output shape or vice versa
    scale_factor, output_shape = fix_scale_and_size(im.shape, output_shape, scale_factor)

    # For a given numeric kernel case, just do convolution and sub-sampling (downscaling only)
    if type(kernel) == np.ndarray and scale_factor[0] <= 1:
        return numeric_kernel(im, kernel, scale_factor, output_shape, kernel_shift_flag)

    # Choose interpolation method, each method has the matching kernel size
    method, kernel_width = {
        "cubic": (cubic, 4.0),
        "lanczos2": (lanczos2, 4.0),
        "lanczos3": (lanczos3, 6.0),
        "box": (box, 1.0),
        "linear": (linear, 2.0),
        None: (cubic, 4.0)  # set default interpolation method as cubic
    }.get(kernel)

    # Antialiasing is only used when downscaling
    antialiasing *= (scale_factor[0] < 1)

    # Sort indices of dimensions according to scale of each dimension. since we are going dim by dim this is efficient
    sorted_dims = np.argsort(np.array(scale_factor)).tolist()

    # Iterate over dimensions to calculate local weights for resizing and resize each time in one direction
    out_im = np.copy(im)
    for dim in sorted_dims:
        # No point doing calculations for scale-factor 1. nothing will happen anyway
        if scale_factor[dim] == 1.0:
            continue

        # for each coordinate (along 1 dim), calculate which coordinates in the input image affect its result and the
        # weights that multiply the values there to get its result.
        weights, field_of_view = contributions(im.shape[dim], output_shape[dim], scale_factor[dim],
                                               method, kernel_width, antialiasing)

        # Use the affecting position values and the set of weights to calculate the result of resizing along this 1 dim
        out_im = resize_along_dim(out_im, dim, weights, field_of_view)

    return out_im


def fix_scale_and_size(input_shape, output_shape, scale_factor):
    """

    @param input_shape:
    @param output_shape:
    @param scale_factor:
    @return:
    """

    # First fixing the scale-factor (if given) to be standardized the function expects (a list of scale factors in the
    # same size as the number of input dimensions)
    if scale_factor is not None:
        # By default, if scale-factor is a scalar we assume 2d resizing and duplicate it.
        if np.isscalar(scale_factor):
            scale_factor = [scale_factor, scale_factor]

        # We extend the size of scale-factor list to the size of the input by assigning 1 to all the unspecified scales
        scale_factor = list(scale_factor)
        scale_factor.extend([1] * (len(input_shape) - len(scale_factor)))

    # Fixing output-shape (if given): extending it to the size of the input-shape, by assigning the original input-size
    # to all the unspecified dimensions
    if output_shape is not None:
        output_shape = list(np.uint(np.array(output_shape))) + list(input_shape[len(output_shape):])

    # Dealing with the case of non-give scale-factor, calculating according to output-shape. note that this is
    # sub-optimal, because there can be different scales to the same output-shape.
    if scale_factor is None:
        scale_factor = 1.0 * np.array(output_shape) / np.array(input_shape)

    # Dealing with missing output-shape. calculating according to scale-factor
    if output_shape is None:
        output_shape = np.uint(np.ceil(np.array(input_shape) * np.array(scale_factor)))

    return scale_factor, output_shape


def contributions(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    """
    This function calculates a set of 'filters' and a set of field_of_view that will later on be applied
    such that each position from the field_of_view will be multiplied with a matching filter from the
    'weights' based on the interpolation method and the distance of the sub-pixel location from the pixel centers
    around it. This is only done for one dimension of the image.

    @param in_length:
    @param out_length:
    @param scale:
    @param kernel:
    @param kernel_width:
    @param antialiasing:
    @return:
    """

    # When anti-aliasing is activated (default and only for downscaling) the receptive field is stretched to size of
    # 1/sf. this means filtering is more 'low-pass filter'.
    fixed_kernel = (lambda arg: scale * kernel(scale * arg)) if antialiasing else kernel
    kernel_width *= 1.0 / scale if antialiasing else 1.0

    # These are the coordinates of the output image
    out_coordinates = np.arange(1, out_length + 1)

    # These are the matching positions of the output-coordinates on the input image coordinates.
    # Best explained by example: say we have 4 horizontal pixels for HR and we downscale by SF=2 and get 2 pixels:
    # [1,2,3,4] -> [1,2]. Remember each pixel number is the middle of the pixel.
    # The scaling is done between the distances and not pixel numbers (the right boundary of pixel 4 is transformed to
    # the right boundary of pixel 2. pixel 1 in the small image matches the boundary between pixels 1 and 2 in the big
    # one and not to pixel 2. This means the position is not just multiplication of the old pos by scale-factor).
    # So if we measure distance from the left border, middle of pixel 1 is at distance d=0.5, border between 1 and 2 is
    # at d=1, and so on (d = p - 0.5).  we calculate (d_new = d_old / sf) which means:
    # (p_new-0.5 = (p_old-0.5) / sf)     ->          p_new = p_old/sf + 0.5 * (1-1/sf)
    match_coordinates = 1.0 * out_coordinates / scale + 0.5 * (1 - 1.0 / scale)

    # This is the left boundary to start multiplying the filter from, it depends on the size of the filter
    left_boundary = np.floor(match_coordinates - kernel_width / 2)

    # Kernel width needs to be enlarged because when covering has sub-pixel borders, it must 'see' the pixel centers
    # of the pixels it only covered a part from. So we add one pixel at each side to consider (weights can zeroize them)
    expanded_kernel_width = np.ceil(kernel_width) + 2

    # Determine a set of field_of_view for each each output position, these are the pixels in the input image
    # that the pixel in the output image 'sees'. We get a matrix whos horizontal dim is the output pixels (big) and the
    # vertical dim is the pixels it 'sees' (kernel_size + 2)
    field_of_view = np.squeeze(np.uint(np.expand_dims(left_boundary, axis=1) + np.arange(expanded_kernel_width) - 1))

    # Assign weight to each pixel in the field of view. A matrix whos horizontal dim is the output pixels and the
    # vertical dim is a list of weights matching to the pixel in the field of view (that are specified in
    # 'field_of_view')
    weights = fixed_kernel(1.0 * np.expand_dims(match_coordinates, axis=1) - field_of_view - 1)

    # Normalize weights to sum up to 1. be careful from dividing by 0
    sum_weights = np.sum(weights, axis=1)
    sum_weights[sum_weights == 0] = 1.0
    weights = 1.0 * weights / np.expand_dims(sum_weights, axis=1)

    # We use this mirror structure as a trick for reflection padding at the boundaries
    mirror = np.uint(np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))))
    field_of_view = mirror[np.mod(field_of_view, mirror.shape[0])]

    # Get rid of  weights and pixel positions that are of zero weight
    non_zero_out_pixels = np.nonzero(np.any(weights, axis=0))
    weights = np.squeeze(weights[:, non_zero_out_pixels])
    field_of_view = np.squeeze(field_of_view[:, non_zero_out_pixels])

    # Final products are the relative positions and the matching weights, both are output_size X fixed_kernel_size
    return weights, field_of_view


def resize_along_dim(im, dim, weights, field_of_view):
    """

    @param im:
    @param dim:
    @param weights:
    @param field_of_view:
    @return:
    """

    # To be able to act on each dim, we swap so that dim 0 is the wanted dim to resize
    tmp_im = np.swapaxes(im, dim, 0)

    # We add singleton dimensions to the weight matrix so we can multiply it with the big tensor we get for
    # tmp_im[field_of_view.T], (bsxfun style)
    weights = np.reshape(weights.T, list(weights.T.shape) + (np.ndim(im) - 1) * [1])

    # This is a bit of a complicated multiplication: tmp_im[field_of_view.T] is a tensor of order image_dims+1.
    # for each pixel in the output-image it matches the positions the influence it from the input image (along 1 dim
    # only, this is why it only adds 1 dim to the shape). We then multiply, for each pixel, its set of positions with
    # the matching set of weights. we do this by this big tensor element-wise multiplication (MATLAB bsxfun style:
    # matching dims are multiplied element-wise while singletons mean that the matching dim is all multiplied by the
    # same number
    tmp_out_im = np.sum(tmp_im[field_of_view.T] * weights, axis=0)

    # Finally we swap back the axes to the original order
    return np.swapaxes(tmp_out_im, dim, 0)


def numeric_kernel(im, kernel, scale_factor, output_shape, kernel_shift_flag):
    """

    @param im:
    @param kernel:
    @param scale_factor:
    @param output_shape:
    @param kernel_shift_flag:
    @return:
    """

    # See kernel_shift function to understand what this is
    if kernel_shift_flag:
        kernel = kernel_shift(kernel, scale_factor)

    # First run a correlation (convolution with flipped kernel)
    out_im = np.zeros_like(im)
    for channel in range(np.ndim(im)):
        out_im[:, :, channel] = filters.correlate(im[:, :, channel], kernel)

    # Then subsample and return
    return out_im[np.round(np.linspace(0, im.shape[0] - 1 / scale_factor[0], output_shape[0])).astype(int)[:, None],
           np.round(np.linspace(0, im.shape[1] - 1 / scale_factor[1], output_shape[1])).astype(int), :]


def kernel_shift(kernel, sf):
    """

    @param kernel:
    @param sf:
    @return:
    """

    # There are two reasons for shifting the kernel:
    # 1. Center of mass is not in the center of the kernel which creates ambiguity. There is no possible way to know
    #    the degradation process included shifting so we always assume center of mass is center of the kernel.
    # 2. We further shift kernel center so that top left result pixel corresponds to the middle of the sfXsf first
    #    pixels. Default is for odd size to be in the middle of the first pixel and for even sized kernel to be at the
    #    top left corner of the first pixel. that is why different shift size needed between od and even size.
    # Given that these two conditions are fulfilled, we are happy and aligned, the way to test it is as follows:
    # The input image, when interpolated (regular bicubic) is exactly aligned with ground truth.

    # First calculate the current center of mass for the kernel
    current_center_of_mass = measurements.center_of_mass(kernel)

    # The second ("+ 0.5 * ....") is for applying condition 2 from the comments above
    wanted_center_of_mass = np.array(kernel.shape) / 2 + 0.5 * (sf - (kernel.shape[0] % 2))

    # Define the shift vector for the kernel shifting (x,y)
    shift_vec = wanted_center_of_mass - current_center_of_mass

    # Before applying the shift, we first pad the kernel so that nothing is lost due to the shift
    # (biggest shift among dims + 1 for safety)
    kernel = np.pad(kernel, np.int(np.ceil(np.max(shift_vec))) + 1, 'constant')

    # Finally shift the kernel and return
    return interpolation.shift(kernel, shift_vec)


# These next functions are all interpolation methods. x is the distance from the left pixel center


def cubic(x):
    """

    @param x:
    @return:
    """

    absx = np.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return ((1.5 * absx3 - 2.5 * absx2 + 1) * (absx <= 1) +
            (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((1 < absx) & (absx <= 2)))


def lanczos2(x):
    """

    @param x:
    @return:
    """

    return (((np.sin(pi * x) * np.sin(pi * x / 2) + np.finfo(np.float32).eps) /
             ((pi ** 2 * x ** 2 / 2) + np.finfo(np.float32).eps))
            * (abs(x) < 2))


def box(x):
    """

    @param x:
    @return:
    """

    return ((-0.5 <= x) & (x < 0.5)) * 1.0


def lanczos3(x):
    """

    @param x:
    @return:
    """

    return (((np.sin(pi * x) * np.sin(pi * x / 3) + np.finfo(np.float32).eps) /
             ((pi ** 2 * x ** 2 / 3) + np.finfo(np.float32).eps))
            * (abs(x) < 3))


def linear(x):
    """

    @param x:
    @return:
    """

    return (x + 1) * ((-1 <= x) & (x < 0)) + (1 - x) * ((0 <= x) & (x <= 1))


if __name__ == "__main__":
    opt = {}

    # Reads the image
    # Needs: input_name, nc_im, not_cuda
    real = read_image("", nc_im=3, not_cuda=False)

    # Adjusts the scales of the image
    # Needs: im_max_size, scale1, stop_scale, train_stages, scale_factor, im_min_size,
    real, scale1, stop_scale, scale_factor = adjust_scales2image(real, nc_im=3, not_cuda=False, im_max_size=614,
                                                                 im_min_size=120, train_stages=16)

    # Create the scales reals pyramids
    # Needs: train_mode, stop_scale, scale_factor,
    reals = create_reals_pyramid(real, train_mode="harmonisation", stop_scale=stop_scale, scale_factor=scale_factor,
                                 nc_im=3, not_cuda=False)

    img_to_augment = convert_image_np(reals[-1]) * 255.0

    data = {"image": img_to_augment}
    aug = Augment()
    augmented = aug.transform(**data)
    image = augmented["image"]
    save_image(f"sample_image.png", image)
