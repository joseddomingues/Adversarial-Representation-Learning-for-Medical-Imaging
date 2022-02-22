import os

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.cuda.amp import GradScaler, autocast
from mlflow import log_param, log_metric, start_run
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

import functions as functions
import models as models
from imresize import imresize_to_shape


def train(opt):
    """
    Trains the network globally
    @param opt: configuration map defined in config.py
    @return: Nothing
    """
    with start_run(nested=True, run_name=opt.experiment_name):

        # Log parameters to mlflow
        log_param("N Iterations", opt.niter)
        log_param("Learning Scale Rate", opt.lr_scale)
        log_param("N Stages", opt.train_stages)
        log_param("N Concurrent Stages", opt.train_depth)
        log_param("Activation Function", opt.activation)

        print("Training model with the following parameters:")
        print("\t number of stages: {}".format(opt.train_stages))
        print("\t number of concurrently trained stages: {}".format(opt.train_depth))
        print("\t learning rate scaling: {}".format(opt.lr_scale))
        print("\t non-linearity: {}".format(opt.activation))

        # Reads the image
        real = functions.read_image(opt)

        # Adjusts the scales of the image
        real = functions.adjust_scales2image(real, opt)

        # Create the scales reals pyramids
        reals = functions.create_reals_pyramid(real, opt)
        print("Training on image pyramid: {}".format([r.shape for r in reals]))
        print("")

        # Loads the naive image into memory
        if opt.naive_img != "":
            naive_img = functions.read_image_dir(opt.naive_img, opt)
            naive_img_large = imresize_to_shape(naive_img, reals[-1].shape[2:], opt)
            naive_img = imresize_to_shape(naive_img, reals[0].shape[2:], opt)
            naive_img = functions.convert_image_np(naive_img) * 255.0
        else:
            naive_img = None
            naive_img_large = None

        # If fine tune assigns image to augment to naive, otherwise use base real
        if opt.fine_tune:
            img_to_augment = naive_img
        else:
            img_to_augment = functions.convert_image_np(reals[0]) * 255.0

        if opt.train_mode == "editing":
            opt.noise_scaling = 0.1

        # If fine tune add stages-1 stage blocks to generator
        generator = init_G(opt)
        if opt.fine_tune:
            for _ in range(opt.train_stages - 1):
                generator.init_next_stage()
            generator.load_state_dict(torch.load('{}/{}/netG.pth'.format(opt.model_dir, opt.train_stages - 1),
                                                 map_location="cuda:{}".format(torch.cuda.current_device())))

        # Fixed noise and noise ampliation to use
        fixed_noise = []
        noise_amp = []

        # Metrics step
        metrics_step = 0

        for scale_num in range(opt.start_scale, opt.train_stages):
            opt.out_ = functions.generate_dir2save(opt)
            opt.outf = '%s/%d' % (opt.out_, scale_num)
            try:
                os.makedirs(opt.outf)
            except OSError:
                print(OSError)
                pass
            functions.save_image('{}/real_scale.jpg'.format(opt.outf), reals[scale_num])

            # If fine tune, load Discriminator, since Generator has stages-1 more blocks
            d_curr = init_D(opt)
            if opt.fine_tune:
                d_curr.load_state_dict(torch.load('{}/{}/netD.pth'.format(opt.model_dir, opt.train_stages - 1),
                                                  map_location="cuda:{}".format(torch.cuda.current_device())))
            # Otherwise, load Discriminator and increase Generator
            elif scale_num > 0:
                d_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_, scale_num - 1)))
                generator.init_next_stage()

            # Create the Writer to output stats
            writer = SummaryWriter(log_dir=opt.outf)

            # Trains the Network on a specific scale
            fixed_noise, noise_amp, generator, d_curr = train_single_scale(d_curr, generator, reals, img_to_augment,
                                                                           naive_img, naive_img_large, fixed_noise,
                                                                           noise_amp, opt, scale_num, writer,
                                                                           metrics_step)

            # Save stats, delete current discriminator and repeat loop
            torch.save(fixed_noise, '%s/fixed_noise.pth' % opt.out_)
            torch.save(generator, '%s/G.pth' % opt.out_)
            torch.save(reals, '%s/reals.pth' % opt.out_)
            torch.save(noise_amp, '%s/noise_amp.pth' % opt.out_)
            del d_curr

        # Close writer and return
        writer.close()
    return


def train_single_scale(netD, netG, reals, img_to_augment, naive_img, naive_img_large,
                       fixed_noise, noise_amp, opt, depth, writer, metrics_step):
    """
    Trains the network on a specific scale
    @param netD: Discriminator
    @param netG: Generator
    @param reals: Array of the real images with different scales
    @param img_to_augment:
    @param naive_img:
    @param naive_img_large:
    @param fixed_noise: Noise to add in the iteration
    @param noise_amp: Noise to add
    @param opt: configuration map defined in config.py
    @param depth: Current depth (scale)
    @param writer: Writer
    @param metrics_step: Metrics step to log in mlflow
    @return:
    """

    # Creates scaler for half precision
    scaler = GradScaler()

    # Get the shapes of the different scales and then the current real image (According to current scale)
    reals_shapes = [real.shape for real in reals]
    real = reals[depth]
    aug = functions.Augment()

    alpha = opt.alpha

    ############################
    # define z_opt for training on reconstruction
    ###########################
    if opt.fine_tune:
        fixed_noise = torch.load('%s/fixed_noise.pth' % opt.model_dir,
                                 map_location="cuda:{}".format(torch.cuda.current_device()))
        z_opt = fixed_noise[depth]
    else:
        if depth == 0:
            if opt.train_mode == "harmonization":
                z_opt = reals[0]
            elif opt.train_mode == "editing":
                z_opt = reals[0] + opt.noise_scaling * functions.generate_noise([opt.nc_im,
                                                                                 reals_shapes[depth][2],
                                                                                 reals_shapes[depth][3]],
                                                                                device=opt.device).detach()
        else:
            z_opt = functions.generate_noise([opt.nfc, reals_shapes[depth][2], reals_shapes[depth][3]],
                                             device=opt.device)

        # Append the noise to the fixed initial noise
        fixed_noise.append(z_opt.detach())

    ############################
    # define optimizers, learning rate schedulers, and learning rates for lower stages
    ###########################

    # setup optimizers for D
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))

    # setup optimizers for G
    # remove gradients from stages that are not trained
    # only trains opt.train_depth stages each time, each with a different learning rate
    for block in netG.body[:-opt.train_depth]:
        for param in block.parameters():
            param.requires_grad = False

    # set different learning rate for lower stages
    parameter_list = [
        {"params": block.parameters(), "lr": opt.lr_g * (opt.lr_scale ** (len(netG.body[-opt.train_depth:]) - 1 - idx))}
        for idx, block in enumerate(netG.body[-opt.train_depth:])]

    # add parameters of head and tail to training
    if depth - opt.train_depth < 0:
        parameter_list += [{"params": netG.head.parameters(), "lr": opt.lr_g * (opt.lr_scale ** depth)}]
    parameter_list += [{"params": netG.tail.parameters(), "lr": opt.lr_g}]
    optimizerG = optim.Adam(parameter_list, lr=opt.lr_g, betas=(opt.beta1, 0.999))

    # define learning rate schedules
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD, milestones=[0.8 * opt.niter],
                                                      gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[0.8 * opt.niter],
                                                      gamma=opt.gamma)

    ############################
    # calculate noise_amp
    ###########################
    if opt.fine_tune:
        noise_amp = torch.load('%s/noise_amp.pth' % opt.model_dir,
                               map_location="cuda:{}".format(torch.cuda.current_device()))
    else:
        if depth == 0:
            noise_amp.append(1)
        else:
            noise_amp.append(0)

            with autocast():
                z_reconstruction = netG(fixed_noise, reals_shapes, noise_amp)

                # define criterion and calculate the loss
                criterion = nn.MSELoss()
                rec_loss = criterion(z_reconstruction, real)

                # calculate RMSE, multiply byt the initial amp and change the last one to it
                RMSE = torch.sqrt(rec_loss).detach()
                _noise_amp = opt.noise_amp_init * RMSE

            noise_amp[-1] = _noise_amp
            del z_reconstruction

    # start training
    _iter = tqdm(range(opt.niter))
    for iter in _iter:
        _iter.set_description('stage [{}/{}]:'.format(depth, opt.stop_scale))

        ############################
        # (0) sample augmented training image
        ###########################
        noise = []
        for d in range(depth + 1):
            if d == 0:
                if opt.fine_tune:
                    if opt.train_mode == "harmonization":
                        noise.append(functions.np2torch(naive_img, opt))
                    elif opt.train_mode == "editing":
                        noise.append(functions.np2torch(naive_img, opt) + opt.noise_scaling * functions.generate_noise(
                            [opt.nc_im, reals_shapes[d][2], reals_shapes[d][3]], device=opt.device).detach())
                else:
                    if opt.train_mode == "harmonization":
                        data = {"image": img_to_augment}
                        augmented = aug.transform(**data)
                        image = augmented["image"]
                        noise.append(functions.np2torch(image, opt))
                    elif opt.train_mode == "editing":
                        image = functions.shuffle_grid(img_to_augment)
                        image = functions.np2torch(image, opt) + \
                                opt.noise_scaling * functions.generate_noise([3, reals_shapes[d][2],
                                                                              reals_shapes[d][3]],
                                                                             device=opt.device).detach()
                        noise.append(image)
            else:
                noise.append(functions.generate_noise([opt.nfc, reals_shapes[d][2], reals_shapes[d][3]],
                                                      device=opt.device).detach())

        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        for j in range(opt.Dsteps):

            # train with real
            netD.zero_grad()

            with autocast():
                output = netD(real)
                errD_real = -output.mean()

                # train with fake
                # generator only trains in the last iteration of Dsteps
                if j == opt.Dsteps - 1:
                    fake = netG(noise, reals_shapes, noise_amp)
                else:
                    with torch.no_grad():
                        fake = netG(noise, reals_shapes, noise_amp)

                # classify the result from generator
                output = netD(fake.detach().clone())
                errD_fake = output.mean()

                # calculate penalty, do backward pass and step
                gradient_penalty = functions.calc_gradient_penalty(netD, real, fake, opt.lambda_grad, opt.device)
                errD_total = errD_real + errD_fake + gradient_penalty

            scaler.scale(errD_total).backward()

        scaler.step(optimizerD)
        schedulerD.step()
        del noise

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################

        with autocast():
            # Once again classify the fake after update
            output = netD(fake)
            errG = -output.mean()

            # having alpha != 0 then generate new output from noise and calculate MSE
            if alpha != 0:
                loss = nn.MSELoss()
                rec = netG(fixed_noise, reals_shapes, noise_amp)
                rec_loss = alpha * loss(rec, real)
            else:
                rec_loss = 0

        # zero grads and apply backward pass
        netG.zero_grad()

        with autocast():
            errG_total = errG + rec_loss

        scaler.scale(errG_total).backward()

        # for _ in range(opt.Gsteps):
        scaler.step(optimizerG)
        schedulerG.step()

        ############################
        # (3) Log Metrics
        ############################
        log_metric('Discriminator Train Loss Real', -errD_real.item(), step=metrics_step)
        log_metric('Discriminator Train Loss Fake', errD_fake.item(), step=metrics_step)
        log_metric('Discriminator Train Loss Gradient Penalty', gradient_penalty.item(), step=metrics_step)
        log_metric('Discriminator Loss', errD_total.item(), step=metrics_step)
        log_metric('Generator Train Loss', errG.item(), step=metrics_step)
        log_metric('Generator Train Loss Reconstruction', rec_loss.item(), step=metrics_step)
        log_metric('Generator Loss', errG_total.item(), step=metrics_step)
        metrics_step += 1

        ############################
        # (4) Log Results
        ############################
        if iter % 250 == 0 or iter + 1 == opt.niter:
            writer.add_scalar('Loss/train/D/real/{}'.format(j), -errD_real.item(), iter + 1)
            writer.add_scalar('Loss/train/D/fake/{}'.format(j), errD_fake.item(), iter + 1)
            writer.add_scalar('Loss/train/D/gradient_penalty/{}'.format(j), gradient_penalty.item(), iter + 1)
            writer.add_scalar('Loss/train/G/gen', errG.item(), iter + 1)
            writer.add_scalar('Loss/train/G/reconstruction', rec_loss.item(), iter + 1)

            functions.save_image('{}/fake_sample_{}.jpg'.format(opt.outf, iter + 1), fake.detach())
            functions.save_image('{}/reconstruction_{}.jpg'.format(opt.outf, iter + 1), rec.detach())

            # generate_samples(netG, img_to_augment, naive_img, naive_img_large, aug, opt, depth,
            #                  noise_amp, writer, reals, iter + 1)
        # elif opt.fine_tune and iter % 100 == 0:
        #     generate_samples(netG, img_to_augment, naive_img, naive_img_large, aug, opt, depth,
        #                      noise_amp, writer, reals, iter + 1)

        scaler.update()

    # saves the networks
    functions.save_networks(netG, netD, z_opt, opt, scaler)
    return fixed_noise, noise_amp, netG, netD


def generate_samples(netG, img_to_augment, naive_img, naive_img_large, aug, opt, depth,
                     noise_amp, writer, reals, iter, n=16):
    """
    Generate samples to log results
    @param netG: Generator
    @param img_to_augment:
    @param naive_img:
    @param naive_img_large:
    @param aug:
    @param opt: configuration map defined in config.py
    @param depth: Current depth (scale)
    @param noise_amp: Noise to ampl
    @param writer: Writer to log results
    @param reals: List of reals images with different scales
    @param iter: Current iteration
    @param n: Number of samples to generate
    @return:
    """

    opt.out_ = functions.generate_dir2save(opt)
    dir2save = '{}/harmonized_samples_stage_{}'.format(opt.out_, depth)
    reals_shapes = [r.shape for r in reals]
    _name = "harmonized" if opt.train_mode == "harmonization" else "edited"
    images = []
    try:
        os.makedirs(dir2save)
    except OSError:
        pass

    if naive_img is not None:
        n = n - 1
    if opt.fine_tune:
        n = 1
    with torch.no_grad():
        for idx in range(n):
            noise = []
            for d in range(depth + 1):
                if d == 0:
                    if opt.fine_tune:
                        if opt.train_mode == "harmonization":
                            augmented_image = functions.np2torch(naive_img, opt)
                            noise.append(augmented_image)
                        elif opt.train_mode == "editing":
                            augmented_image = functions.np2torch(naive_img, opt)
                            noise.append(augmented_image + opt.noise_scaling *
                                         functions.generate_noise([opt.nc_im, reals_shapes[d][2],
                                                                   reals_shapes[d][3]], device=opt.device).detach())
                    else:
                        if opt.train_mode == "harmonization":
                            data = {"image": img_to_augment}
                            augmented = aug.transform(**data)
                            augmented_image = functions.np2torch(augmented["image"], opt)
                            noise.append(augmented_image)
                        elif opt.train_mode == "editing":
                            image = functions.shuffle_grid(img_to_augment)
                            augmented_image = functions.np2torch(image, opt)
                            noise.append(augmented_image + opt.noise_scaling *
                                         functions.generate_noise([opt.nc_im, reals_shapes[d][2],
                                                                   reals_shapes[d][3]], device=opt.device).detach())
                else:
                    noise.append(functions.generate_noise([opt.nfc, reals_shapes[d][2], reals_shapes[d][3]],
                                                          device=opt.device).detach())

            with autocast():
                sample = netG(noise, reals_shapes, noise_amp)
            functions.save_image('{}/{}_naive_sample.jpg'.format(dir2save, idx), augmented_image)
            functions.save_image('{}/{}_{}_sample.jpg'.format(dir2save, idx, _name), sample.detach())
            augmented_image = imresize_to_shape(augmented_image, sample.shape[2:], opt)
            images.append(augmented_image)
            images.append(sample.detach())

        if opt.fine_tune:
            mask_file_name = '{}_mask{}'.format(opt.naive_img[:-4], opt.naive_img[-4:])
            augmented_image = imresize_to_shape(naive_img_large, sample.shape[2:], opt)
            if os.path.exists(mask_file_name):
                mask = get_mask(mask_file_name, augmented_image, opt)
                sample_w_mask = (1 - mask) * augmented_image + mask * sample.detach()
                functions.save_image('{}/{}_sample_w_mask_{}.jpg'.format(dir2save, _name, iter), sample_w_mask.detach())
                images = torch.cat([augmented_image, sample.detach(), sample_w_mask], 0)
                grid = make_grid(images, nrow=3, normalize=True)
                writer.add_image('{}_images_{}'.format(_name, depth), grid, iter)
            else:
                print("Warning: no mask with name {} exists for image {}".format(mask_file_name, opt.input_name))
                print("Only showing results without mask.")
                images = torch.cat([augmented_image, sample.detach()], 0)
                grid = make_grid(images, nrow=2, normalize=True)
                writer.add_image('{}_images_{}'.format(_name, depth), grid, iter)
            functions.save_image('{}/{}_sample_{}.jpg'.format(dir2save, _name, iter), sample.detach())
        else:
            if naive_img is not None:
                noise = []
                for d in range(depth + 1):
                    if d == 0:
                        if opt.train_mode == "harmonization":
                            noise.append(functions.np2torch(naive_img, opt))
                        elif opt.train_mode == "editing":
                            noise.append(functions.np2torch(naive_img, opt) + opt.noise_scaling * \
                                         functions.generate_noise([opt.nc_im, reals_shapes[d][2],
                                                                   reals_shapes[d][3]],
                                                                  device=opt.device).detach())
                    else:
                        noise.append(functions.generate_noise([opt.nfc, reals_shapes[d][2], reals_shapes[d][3]],
                                                              device=opt.device).detach())

                with autocast():
                    sample = netG(noise, reals_shapes, noise_amp)
                _naive_img = imresize_to_shape(naive_img_large, sample.shape[2:], opt)
                images.insert(0, sample.detach())
                images.insert(0, _naive_img)
                functions.save_image('{}/{}_sample_{}.jpg'.format(dir2save, _name, iter), sample.detach())

                mask_file_name = '{}_mask{}'.format(opt.naive_img[:-4], opt.naive_img[-4:])
                if os.path.exists(mask_file_name):
                    mask = get_mask(mask_file_name, _naive_img, opt)
                    sample_w_mask = (1 - mask) * _naive_img + mask * sample.detach()
                    functions.save_image('{}/{}_sample_w_mask_{}.jpg'.format(dir2save, _name, iter), sample_w_mask)

            images = torch.cat(images, 0)
            grid = make_grid(images, nrow=4, normalize=True)
            writer.add_image('{}_images_{}'.format(_name, depth), grid, iter)


def get_mask(mask_file_name, real_img, opt):
    mask = functions.read_image_dir(mask_file_name, opt)
    if mask.shape[3] != real_img.shape[3]:
        mask = imresize_to_shape(mask, [real_img.shape[2], real_img.shape[3]], opt)
    mask = functions.dilate_mask(mask, opt)
    return mask


def init_G(opt):
    """
    Creates the generator, sends it to gpu and apply weights
    @param opt: Configuration map defined in config.py
    @return: The generator network
    """

    netG = models.GrowingGenerator(opt).to(opt.device)
    netG.apply(models.weights_init)
    return netG


def init_D(opt):
    """
    Creates the discriminator, sends it to gpu and apply weights
    @param opt: Configuration map defined in config.py
    @return: The discriminator network
    """

    netD = models.Discriminator(opt).to(opt.device)
    netD.apply(models.weights_init)
    return netD
