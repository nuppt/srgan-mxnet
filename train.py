import random
import time

from mxnet import image, ndarray, autograd
from mxnet.gluon import data as gdata
from mxnet.gluon.data.vision import transforms
from mxnet.base import numeric_types
from mxnet.gluon.data import DataLoader
import mxnet.ndarray as nd
import mxnet as mx
from mxboard import SummaryWriter
from mxnet.gluon.model_zoo import vision

from .model import *
from .dataset import *
from .monitor import *


def weights_init(params):
    for param_name in params:
        param = params[param_name]
        if param_name.find('conv') != -1:
            if param_name.find('weight') != -1:
                param.set_data(nd.random.normal(0.0,0.02,shape=param.data().shape))
            elif param_name.find('bias') != -1:
                param.set_data(nd.zeros(param.data().shape))
        elif param_name.find('batchnorm') != -1:
            if param_name.find('gamma') != -1:
                param.set_data(nd.random.normal(1.0, 0.02,shape=param.data().shape))
            elif param_name.find('beta') != -1:
                param.set_data(nd.zeros(param.data().shape))

def mse_loss(output, target):
    e = ((output - target) ** 2).mean(axis=0, exclude=True)
    return e

def vgg_feature(input, context):
    vgg19 = vision.vgg19(pretrained=True, ctx=context)
    features = vgg19.features[:22]
    return features(input)

class loss_dict:
    def __init__(self):
        self.losses = {}

    def __getitem__(self, item):
        return self.losses[item]

    def add(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.losses:
                self.losses[key] = [value]
            else:
                self.losses[key].append(value)

    def reset(self):
        self.losses = {}


def train(opt):
    sw = SummaryWriter(logdir='./logs', flush_secs=5)
    decay_every = int(opt.n_epoch / 2)

    if opt.experiment is None:
        opt.experiment = 'samples'
    os.system('mkdir {}'.format(opt.experiment))

    if opt.gpu_ids == '-1':
        context = [mx.cpu()]
    else:
        #context = mx.gpu(0)
        context = [mx.gpu(int(i)) for i in opt.gpu_ids.split(',') if i.strip()]


    ##### Prapare data for training or validation #####
    dataset = DataSet(opt.dataroot, RandomCrop(opt.fineSize), transforms.Resize(int(opt.fineSize / 4), interpolation=3),
                      transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                      ]))
    dataloader = DataLoader(dataset, batch_size=opt.batchSize,
                            shuffle=True, num_workers=int(opt.workers), last_batch='rollover')

    ##### Build Network #####
    netG = SRGenerator()
    netG.initialize(ctx=context)
    netD = SRDiscriminator()
    netD.initialize(ctx=context)

    # Enforce non-deferred initialization by one forward pass computation
    dummy_in = nd.random.uniform(0, 1, (1, 3, int(opt.fineSize / 4), int(opt.fineSize / 4)), ctx=context[0])
    netD(netG(dummy_in))

    # Our own re-setting on parameters
    weights_init(netG.collect_params())
    netG.collect_params().reset_ctx(context)
    weights_init(netD.collect_params())
    netD.collect_params().reset_ctx(context)

    optimizer_G = gluon.Trainer(params=netG.collect_params(),
                                optimizer='adam',
                                optimizer_params={'learning_rate': opt.lr_init, 'beta1': opt.beta1},
                                kvstore='local')
    optimizer_D = gluon.Trainer(params=netD.collect_params(),
                                optimizer='adam',
                                optimizer_params={'learning_rate': opt.lr_init, 'beta1': opt.beta1},
                                kvstore='local')


    ##### Stage 1/2 of Training Process #####
    # Pre-train Generator G to avoid undesired local optima when training SRGAN.
    param_file = os.path.join(opt.experiment, 'netG_init_epoch.param')
    if os.path.exists(param_file):
        netG.load_parameters(param_file, ctx=context)
    else:
        for epoch in range(opt.n_epoch_init):
            start = time.time()
            for hr_img_iter, lr_img_iter in dataloader:
                #hr_img = hr_img.as_in_context(context)
                #lr_img = lr_img.as_in_context(context)
                hr_img = gluon.utils.split_and_load(hr_img_iter, ctx_list=context)
                lr_img = gluon.utils.split_and_load(lr_img_iter, ctx_list=context)
                with autograd.record():
                    hr_img_pred = netG(lr_img)
                    loss = mse_loss(hr_img, hr_img_pred)
                    autograd.backward(loss)
                optimizer_G.step(opt.batchSize)
            nd.waitall()
            train_time = time.time() - start
            print("Epoch %d:  mse: %.8f  trainning time:%.1f sec" % (epoch, loss.mean().asscalar(), train_time))
            if epoch % 20 == 0:
                netG.save_parameters('{0}/netG_init_epoch_{1}.param'.format(opt.experiment, epoch))
            if epoch == opt.n_epoch_init - 1:
                netG.save_parameters('{0}/netG_init_epoch.param'.format(opt.experiment))

    ##### Stage 2/2 of Training Process #####
    # Jointly optimize G and D, namely train SRGAN.
    mean_mask = nd.zeros((opt.batchSize, 3, opt.fineSize, opt.fineSize), ctx=context)
    mean_mask[:, 0, :, :] = 0.485
    mean_mask[:, 1, :, :] = 0.456
    mean_mask[:, 2, :, :] = 0.406
    std_mask = nd.zeros((opt.batchSize, 3, opt.fineSize, opt.fineSize), ctx=context)
    std_mask[:, 0, :, :] = 0.229
    std_mask[:, 1, :, :] = 0.224
    std_mask[:, 2, :, :] = 0.225
    real_label = nd.ones((opt.batchSize,), ctx=context)
    fake_label = nd.zeros((opt.batchSize,), ctx=context)

    mean_mask = mx.gluon.utils.split_and_load(mean_mask, ctx_list=context)
    std_mask = mx.gluon.utils.split_and_load(std_mask, ctx_list=context)
    real_label = mx.gluon.utils.split_and_load(real_label, ctx_list=context)
    fake_label = mx.gluon.utils.split_and_load(fake_label, ctx_list=context)


    loss_d = gluon.loss.SigmoidBinaryCrossEntropyLoss()
    losses_log = loss_dict()

    for epoch in range(0, opt.n_epoch):
        for i, (hr_img, lr_img) in enumerate(dataloader):
            losses_log.reset()
            hr_img = hr_img.as_in_context(context)
            lr_img = lr_img.as_in_context(context)
            with autograd.record():
                output = netD(hr_img).reshape((-1, 1))
                errD_real = loss_d(output, real_label)
                hr_img_fake = netG(lr_img)
                output = netD(hr_img_fake.detach()).reshape((-1, 1))
                errD_fake = loss_d(output, fake_label)
                errD = errD_real + errD_fake
            autograd.backward(errD)
            optimizer_D.step(opt.batchSize)
            losses_log.add(errD=errD)
            losses_log.add(lr_img=lr_img, hr_img=hr_img, hr_img_fake=hr_img_fake)

            with autograd.record():
                errM = mse_loss(hr_img_fake, hr_img)
                fake_emb = vgg_feature(((hr_img_fake + 1) / 2 - mean_mask) / std_mask)
                real_emb = vgg_feature(((hr_img + 1) / 2 - mean_mask) / std_mask)
                errV = 6e-3 * mse_loss(fake_emb, real_emb)
                output = netD(hr_img_fake).reshape((-1, 1))
                errA = 1e-3 * loss_d(output, real_label)
                errG = errM + errV + errA
            autograd.backward(errG)
            optimizer_G.step(opt.batchSize)
            losses_log.add(errG=errG, errM=errM, errV=errV, errA=errA)
            plot_loss(sw, losses_log, epoch * len(dataloader) + i, epoch, i)

        if epoch != 0 and (epoch % decay_every == 0):
            optimizer_G.set_learning_rate(optimizer_G.learning_rate * opt.lr_decay)
            optimizer_D.set_learning_rate(optimizer_D.learning_rate * opt.lr_decay)
        if (epoch != 0) and (epoch % 10 == 0):
            plot_img(sw, losses_log)
            netG.save_parameters('{0}/netG_epoch_{1}.param'.format(opt.experiment, epoch))
            netD.save_parameters('{0}/netD_epoch_{1}.param'.format(opt.experiment, epoch))