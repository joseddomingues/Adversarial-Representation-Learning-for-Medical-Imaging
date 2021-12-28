import os
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from mlflow import log_param, log_metric, start_run
from pytorch_lightning.lite import LightningLite
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

import functions as functions
import models as models
from evaluate_generation import GenerationEvaluator


class Lite(LightningLite):

    def run(self, opt):
        """
        Trains the network globally
        @param opt: configuration map defined in config.py
        @return: Nothing
        """

        with start_run(nested=True, run_name=opt.experiment_name):

            # Log parameters to mlflow
            log_param("N Iterations", opt.niter)
            log_param("Learning Scale Rate", opt.lr_scale)
            log_param("N Training Stages", opt.train_stages)
            log_param("Train Depth", opt.train_depth)
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
            print("Training on image pyramid: {}\n".format([r.shape for r in reals]))

            # Initiate the generator model and add it to cuda
            generator = self.init_G(opt)

            # Fixed noise and noise ampliation to use
            fixed_noise = []
            noise_amp = []

            # For each scale of the number os scales will be used
            # stop_scale - Defined according to adjusting image scales
            for scale_num in range(opt.stop_scale + 1):

                # Generates the directory to save the outputs and file. Also saves the real image for that scale
                opt.out_ = functions.generate_dir2save(opt)
                opt.outf = '%s/%d' % (opt.out_, scale_num)
                try:
                    os.makedirs(opt.outf)
                except OSError:
                    print(OSError)
                    pass

                functions.save_image('{}/real_scale.jpg'.format(opt.outf), reals[scale_num])

                # Initiates the discriminator.
                # If the scale is bigger than 0 => Load the previous discriminator and init next stage
                d_curr = self.init_D(opt)
                if scale_num > 0:
                    d_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_, scale_num - 1)))
                    generator.init_next_stage()

                # Create the Writer to output stats
                writer = SummaryWriter(log_dir=opt.outf)

                # Trains the Network on a specific scale
                fixed_noise, noise_amp, generator, d_curr = self.train_single_scale(d_curr, generator, reals,
                                                                                    fixed_noise,
                                                                                    noise_amp,
                                                                                    opt, scale_num, writer)

                # Save stats, delete current discriminator and repeat loop
                torch.save(fixed_noise, '%s/fixed_noise.pth' % opt.out_)
                torch.save(generator, '%s/G.pth' % opt.out_)
                torch.save(reals, '%s/reals.pth' % opt.out_)
                torch.save(noise_amp, '%s/noise_amp.pth' % opt.out_)
                del d_curr

            # Close writer and return
            writer.close()
        return

    # Train the network on a specific scale
    # netD - configuration map defined in config.py
    # netG - configuration map defined in config.py
    # reals - configuration map defined in config.py
    # fixed_noise - configuration map defined in config.py
    # noise_amp - configuration map defined in config.py
    # opt - configuration map defined in config.py
    # depth - configuration map defined in config.py
    # writer - configuration map defined in config.py

    def train_single_scale(self, netD, netG, reals, fixed_noise, noise_amp, opt, depth, writer):
        """
        Trains the network on a specific scale
        @param netD: Discriminator
        @param netG: Generator
        @param reals: Array of the real images with different scales
        @param fixed_noise: Noise to add in the iteration
        @param noise_amp: Noise to add
        @param opt: configuration map defined in config.py
        @param depth: Current depth (scale)
        @param writer: Writer
        @return: fixed_noise, noise_amp, netG, netD
        """

        # Get the shapes of the different scales and then the current real image (According to current scale)
        reals_shapes = [real.shape for real in reals]
        real = reals[depth]

        # Get alpha
        alpha = opt.alpha

        ############################
        # define z_opt for training on reconstruction
        ###########################

        # If on the beginning then use the first real image scale unless is animation
        if depth == 0:
            if opt.train_mode == "generation" or opt.train_mode == "retarget":
                z_opt = reals[0]
            elif opt.train_mode == "animation":
                z_opt = functions.generate_noise([opt.nc_im, reals_shapes[depth][2], reals_shapes[depth][3]],
                                                 device=opt.device).detach()

        # Else then generate noise depending on what is required
        else:
            if opt.train_mode == "generation" or opt.train_mode == "animation":
                z_opt = functions.generate_noise([opt.nfc,
                                                  reals_shapes[depth][2] + opt.num_layer * 2,
                                                  reals_shapes[depth][3] + opt.num_layer * 2],
                                                 device=opt.device)
            else:
                z_opt = functions.generate_noise([opt.nfc, reals_shapes[depth][2], reals_shapes[depth][3]],
                                                 device=opt.device).detach()

        # Append the noise to the fixed initial noise
        fixed_noise.append(z_opt.detach())

        ############################
        # define optimizers, learning rate schedulers, and learning rates for lower stages
        ###########################

        # setup optimizers for D
        optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
        netD, optimizerD = self.setup(netD, optimizerD)

        # setup optimizers for G
        # remove gradients from stages that are not trained
        # only trains opt.train_depth stages each time, each with a different learning rate
        for block in netG.body[:-opt.train_depth]:
            for param in block.parameters():
                param.requires_grad = False

        # set different learning rate for lower stages
        parameter_list = [
            {"params": block.parameters(),
             "lr": opt.lr_g * (opt.lr_scale ** (len(netG.body[-opt.train_depth:]) - 1 - idx))}
            for idx, block in enumerate(netG.body[-opt.train_depth:])]

        # add parameters of head and tail to training
        if depth - opt.train_depth < 0:
            parameter_list += [{"params": netG.head.parameters(), "lr": opt.lr_g * (opt.lr_scale ** depth)}]
        parameter_list += [{"params": netG.tail.parameters(), "lr": opt.lr_g}]
        optimizerG = optim.Adam(parameter_list, lr=opt.lr_g, betas=(opt.beta1, 0.999))
        netG, optimizerG = self.setup(netD, optimizerG)

        # define learning rate schedules
        schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD, milestones=[0.8 * opt.niter],
                                                          gamma=opt.gamma)
        schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[0.8 * opt.niter],
                                                          gamma=opt.gamma)

        ############################
        # calculate noise_amp
        ###########################
        if depth == 0:
            # if the first stage then just append to noise amp
            noise_amp.append(1)
        else:
            # if not the first stage append 0 and then generate result using G
            noise_amp.append(0)
            z_reconstruction = netG(fixed_noise, reals_shapes, noise_amp)

            # define criterion and calculate the loss
            criterion = nn.MSELoss()
            rec_loss = criterion(z_reconstruction, real)

            # calculate RMSE, multiply byt the initial amp and change the last one to it
            RMSE = torch.sqrt(rec_loss).detach()
            _noise_amp = opt.noise_amp_init * RMSE
            noise_amp[-1] = _noise_amp

        # start training
        _iter = tqdm(range(opt.niter))
        for iter in _iter:
            _iter.set_description('stage [{}/{}]:'.format(depth, opt.stop_scale))

            ############################
            # (0) sample noise for unconditional generation
            ###########################
            noise = functions.sample_random_noise(depth, reals_shapes, opt)

            ############################
            # (1) Update D network: maximize D(x) + D(G(z))
            ###########################
            for j in range(opt.Dsteps):

                # train with real
                netD.zero_grad()
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
                output = netD(fake.detach())
                errD_fake = output.mean()

                # calculate penalty, do backward pass and step
                gradient_penalty = functions.calc_gradient_penalty(netD, real, fake, opt.lambda_grad, opt.device)
                errD_total = errD_real + errD_fake + gradient_penalty
                self.backward(errD_total, model=netD)
                optimizerD.step()

            ############################
            # (2) Update G network: maximize D(G(z))
            ###########################
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
            errG_total = errG + rec_loss
            self.backward(errG_total, model=netG)

            # optimizer applied G number of steps
            for _ in range(opt.Gsteps):
                optimizerG.step()

            ############################
            # (3) Log Results
            ###########################
            if iter % 250 == 0 or iter + 1 == opt.niter:
                writer.add_scalar('Loss/train/D/real/{}'.format(j), -errD_real.item(), iter + 1)
                writer.add_scalar('Loss/train/D/fake/{}'.format(j), errD_fake.item(), iter + 1)
                writer.add_scalar('Loss/train/D/gradient_penalty/{}'.format(j), gradient_penalty.item(), iter + 1)
                writer.add_scalar('Loss/train/G/gen', errG.item(), iter + 1)
                writer.add_scalar('Loss/train/G/reconstruction', rec_loss.item(), iter + 1)

                # Log metrics
                log_metric('Discriminator Train Loss Real', -errD_real.item(), step=iter + 1)
                log_metric('Discriminator Train Loss Fake', errD_fake.item(), step=iter + 1)
                log_metric('Discriminator Train Loss Gradient Penalty', gradient_penalty.item(), step=iter + 1)
                log_metric('Generator Train Loss', errG.item(), step=iter + 1)
                log_metric('Generator Train Loss Reconstruction', rec_loss.item(), step=iter + 1)

            if iter % 500 == 0 or iter + 1 == opt.niter:
                functions.save_image('{}/fake_sample_{}.jpg'.format(opt.outf, iter + 1), fake.detach())
                functions.save_image('{}/reconstruction_{}.jpg'.format(opt.outf, iter + 1), rec.detach())
                self.generate_samples(netG, opt, depth, noise_amp, writer, reals, iter + 1)

            schedulerD.step()
            schedulerG.step()

        if depth + 1 == len(reals):
            evaluator = GenerationEvaluator(opt.input_name, '{}/gen_samples_stage_{}'.format(opt.out_, depth))
            log_metric('FID', evaluator.run_fid(), step=iter + 1)
            log_metric('LPIPS', evaluator.run_lpips(), step=iter + 1)
            ssim, ms_ssim = evaluator.run_mssim()
            log_metric('SSIM', ssim, step=iter + 1)
            log_metric('MS-SSIM', ms_ssim, step=iter + 1)
            # break

        # saves the networks
        functions.save_networks(netG, netD, z_opt, opt)
        return fixed_noise, noise_amp, netG, netD

    def generate_samples(self, netG, opt, depth, noise_amp, writer, reals, iter, n=25):
        """
        Generate samples to log results
        @param netG: Generator
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
        dir2save = '{}/gen_samples_stage_{}'.format(opt.out_, depth)
        reals_shapes = [r.shape for r in reals]
        all_images = []
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        with torch.no_grad():
            for idx in range(n):
                noise = functions.sample_random_noise(depth, reals_shapes, opt)
                sample = netG(noise, reals_shapes, noise_amp)
                all_images.append(sample)
                functions.save_image('{}/gen_sample_{}.jpg'.format(dir2save, idx), sample.detach())

            all_images = torch.cat(all_images, 0)
            all_images[0] = reals[depth].squeeze()
            grid = make_grid(all_images, nrow=min(5, n), normalize=True)
            writer.add_image('gen_images_{}'.format(depth), grid, iter)

    def init_G(self, opt):
        """
        Creates the generator, sends it to gpu and apply weights
        @param opt: Configuration map defined in config.py
        @return: The generator network
        """

        netG = models.GrowingGenerator(opt)
        netG.apply(models.weights_init)
        return netG

    def init_D(self, opt):
        """
        Creates the discriminator, sends it to gpu and apply weights
        @param opt: Configuration map defined in config.py
        @return: The discriminator network
        """

        netD = models.Discriminator(opt)
        netD.apply(models.weights_init)
        return netD
