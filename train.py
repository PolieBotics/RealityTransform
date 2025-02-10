import time
import os
import numpy as np
import torch
import math
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call

# For Python >= 3.9, math.gcd is how to do gcd
def lcm(a, b):
    return abs(a * b) // math.gcd(a, b) if a and b else 0

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer

#######################################################
# Optional: PyTorch’s built-in AMP for mixed precision
#######################################################
if torch.cuda.is_available():
    from torch.cuda.amp import autocast, GradScaler
else:
    # If no CUDA, define dummy context managers/scaler
    from contextlib import contextmanager
    @contextmanager
    def autocast(enabled=True):
        yield
    class GradScaler:
        def scale(self, loss):
            return loss
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0

# Force gcd-lcm for print frequency
opt.print_freq = lcm(opt.print_freq, opt.batchSize)

# Debug mode
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

######################
# Create Data Loader
######################
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

#######################
# Create Model
#######################
model = create_model(opt)
model.train()  # ensure model is in training mode

# If multiple GPUs are specified, wrap the model
if len(opt.gpu_ids) > 1:
    print("Using multi-GPU DataParallel on GPUs: ", opt.gpu_ids)
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

########################################
# Access the Generator/Discriminator
########################################
# If wrapped in DataParallel, you have to do model.module
# If single GPU, model is used directly
if len(opt.gpu_ids) > 1:
    optimizer_G = model.module.optimizer_G
    optimizer_D = model.module.optimizer_D
    loss_names = model.module.loss_names
else:
    optimizer_G = model.optimizer_G
    optimizer_D = model.optimizer_D
    loss_names = model.loss_names

################################################
# (Optional) Native AMP GradScaler
################################################
scaler = GradScaler() if opt.fp16 else None

total_steps = (start_epoch - 1) * dataset_size + epoch_iter
display_delta = total_steps % opt.display_freq
print_delta   = total_steps % opt.print_freq
save_delta    = total_steps % opt.save_latest_freq

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()

    # If resuming mid-epoch
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size

    for i, data in enumerate(dataset, start=epoch_iter):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()

        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # Flag to decide if we save outputs now
        save_fake = (total_steps % opt.display_freq == display_delta)

        ###################################################
        # Forward and Loss Calculation Under AMP (Optional)
        ###################################################
        if scaler is not None:
            # Mixed Precision
            with autocast():
                losses, generated = model(
                    Variable(data['label']),
                    Variable(data['inst']),
                    Variable(data['image']),
                    Variable(data['feat']),
                    infer=save_fake
                )
        else:
            # No AMP
            losses, generated = model(
                Variable(data['label']),
                Variable(data['inst']),
                Variable(data['image']),
                Variable(data['feat']),
                infer=save_fake
            )

        # Aggregate losses across any GPUs
        losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
        loss_dict = dict(zip(loss_names, losses))

        # Typical: 
        #   D_fake, D_real, G_GAN, G_GAN_Feat (optional), G_VGG (optional), ...
        # Combine whichever losses you need
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN']
        if 'G_GAN_Feat' in loss_dict:
            loss_G += loss_dict['G_GAN_Feat']
        if 'G_VGG' in loss_dict:
            loss_G += loss_dict['G_VGG']

        #######################
        # Backprop G
        #######################
        optimizer_G.zero_grad()
        if scaler is not None:
            scaler.scale(loss_G).backward()
            scaler.step(optimizer_G)
        else:
            loss_G.backward()
            optimizer_G.step()

        #######################
        # Backprop D
        #######################
        optimizer_D.zero_grad()
        if scaler is not None:
            scaler.scale(loss_D).backward()
            scaler.step(optimizer_D)
            scaler.update()
        else:
            loss_D.backward()
            optimizer_D.step()

        # Print to console
        if total_steps % opt.print_freq == print_delta:
            errors = {
                k: (v.item() if hasattr(v, 'item') else v)
                for k, v in loss_dict.items()
            }
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer = Visualizer(opt)
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)

        # Save image samples
        if save_fake:
            visuals = OrderedDict([
                ('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                ('synthesized_image', util.tensor2im(generated.data[0])),
                ('real_image', util.tensor2im(data['image'][0]))
            ])
            visualizer.display_current_results(visuals, epoch, total_steps)

        # Save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            if len(opt.gpu_ids) > 1:
                model.module.save('latest')
            else:
                model.save('latest')
            print('Saved latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if epoch_iter >= dataset_size:
            break

    # End of epoch
    print('End of epoch %d / %d \t Time Taken: %d sec'
          % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    # Save model every opt.save_epoch_freq
    if epoch % opt.save_epoch_freq == 0:
        if len(opt.gpu_ids) > 1:
            model.module.save('latest')
            model.module.save(epoch)
        else:
            model.save('latest')
            model.save(epoch)
        print('Saved model at epoch %d, total_steps %d' % (epoch, total_steps))
        np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

    # Switch from local enhancer training to full net
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        if len(opt.gpu_ids) > 1:
            model.module.update_fixed_params()
        else:
            model.update_fixed_params()

    # Start decaying learning rate
    if epoch > opt.niter:
        if len(opt.gpu_ids) > 1:
            model.module.update_learning_rate()
        else:
            model.update_learning_rate()
