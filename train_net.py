import mxnet as mx
import numpy as np
import os, time, shutil
from config import config
from mxnet import gluon, image, init, nd, autograd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from sklearn.model_selection import train_test_split
import warnings
import datetime
import gluoncv.utils as gutils
from gluoncv.utils import makedirs, LRSequential, LRScheduler
from nets.nets_zoo import get_model
import logging

# 设置随机种子
gutils.random.seed(0)


def list_images(root):
    root = os.path.expanduser(root)
    synsets = []
    items = []
    exts = ['.jpg', '.jpeg', '.png']
    for folder in sorted(os.listdir(root)):
        path = os.path.join(root, folder)
        if not os.path.isdir(path):
            warnings.warn('Ignoring %s, which is not a directory.' % path, stacklevel=3)
            continue
        label = len(synsets)
        synsets.append(folder)
        for filename in sorted(os.listdir(path)):
            filename = os.path.join(path, filename)
            ext = os.path.splitext(filename)[1]
            if ext.lower() not in exts:
                warnings.warn('Ignoring %s of type %s. Only support %s' % (
                    filename, ext, ', '.join(exts)))
                continue
            items.append((filename, label))
    return synsets, items


class ImageFolderDataset(gluon.data.dataset.Dataset):
    def __init__(self, synsets=None, list_images=None, flag=1, transform=None):
        self._flag = flag
        self.synsets = synsets
        self._transform = transform
        self.items = list_images

    def __getitem__(self, idx):
        img = image.imread(self.items[idx][0], self._flag)
        label = self.items[idx][1]
        if self._transform is not None:
            return self._transform(img, label)
        return img, label

    def __len__(self):
        return len(self.items)


def get_data_loader(logger):
    jitter_param = 0.4
    lighting_param = 0.1
    transform_train = transforms.Compose([
        transforms.Resize(256),
        #     ImageNetPolicy(),
        transforms.RandomFlipLeftRight(),
        transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                     saturation=jitter_param),
        transforms.RandomLighting(lighting_param),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        #     transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    synsets, all_items = list_images(root=config.DATA_PATH)
    train_items, test_items = train_test_split(all_items, test_size=config.TEST_SIZE,random_state=config.RANDOM_STATE)
    # 训练集和测试集
    train_dataset = ImageFolderDataset(synsets=synsets, list_images=train_items, flag=1)
    test_dataset = ImageFolderDataset(synsets=synsets, list_images=test_items, flag=1)
    logger.info('训练集的样本数是:%s,测试集的样本数是:%s' % (len(train_dataset), len(test_dataset)))
    train_dataloader = gluon.data.DataLoader(
        train_dataset.transform_first(transform_train),
        batch_size=config.BATCH_SIZE,
        last_batch='rollover',
        shuffle=True,
        num_workers=config.WORKERS)
    test_dataloader = gluon.data.DataLoader(
        test_dataset.transform_first(transform_test),
        batch_size=config.BATCH_SIZE,
        last_batch='rollover',
        shuffle=False,
        num_workers=config.WORKERS)
    return train_dataloader, test_dataloader


def label_transform(label, classes=config.NUM_CLASSES):
    ind = label.astype('int')
    res = nd.zeros((ind.shape[0], classes), ctx=label.context)
    res[nd.arange(ind.shape[0], ctx=label.context), ind] = 1
    return res


def test(net, ctx, val_data, dtype):
    Loss = gluon.loss.SoftmaxCrossEntropyLoss()
    test_loss = mx.metric.Loss()
    test_metric = mx.metric.Accuracy()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        outputs = [net(X.astype(dtype, copy=False)) for X in data]
        losses = [Loss(y_hat, y.astype(dtype, copy=False)) for y_hat, y in zip(outputs, label)]
        test_loss.update(0, losses)
        test_metric.update(label, outputs)
    _, test_loss = test_loss.get()
    _, test_acc = test_metric.get()
    return test_loss, test_acc


def train_basic(net, train_dataloader, valid_dataloader, num_epochs, batch_size, lr, wd, ctx, dtype, logger):
    opt_params = {'learning_rate': lr, 'momentum': 0.9, 'wd': wd}
    if dtype != 'float32':
        opt_params['multi_precision'] = True
    trainer = gluon.Trainer(net.collect_params(), 'sgd', opt_params)
    # 定义loss函数和准确率的计算
    Loss = gluon.loss.SoftmaxCrossEntropyLoss()
    train_loss = mx.metric.Loss()
    train_acc = mx.metric.Accuracy()
    prev_time = datetime.datetime.now()
    best_val_score = 0
    for epoch in range(1, num_epochs + 1):
        train_loss.reset()
        train_acc.reset()
        if epoch == 20 or epoch == 40:
            trainer.set_learning_rate(trainer.learning_rate * 0.1)
        for data, label in train_dataloader:
            Xs = gluon.utils.split_and_load(data, ctx)
            ys = gluon.utils.split_and_load(label, ctx)
            with autograd.record():
                y_hats = [net(X.astype(dtype, copy=False)) for X in Xs]
                losses = [Loss(y_hat, y.astype(dtype, copy=False)) for y_hat, y in zip(y_hats, ys)]
            for l in losses:
                l.backward()
            trainer.step(batch_size)
            train_loss.update(0, losses)
            train_acc.update(ys, y_hats)
        _, epoch_loss = train_loss.get()
        _, epoch_acc = train_acc.get()
        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = ", Time %02d:%02d:%02d" % (h, m, s)
        if valid_dataloader is not None:
            val_loss, val_acc = test(net, ctx, valid_dataloader, dtype)
            if val_acc > best_val_score:
                best_val_score = val_acc
                net.save_parameters(config.BASE_MODEL_PATH)
            epoch_str = ("Epoch %d. Loss: %f, Train acc %f, Test Loss: %f, Test acc %f " % (
                epoch, epoch_loss, epoch_acc, val_loss, val_acc))
        else:
            epoch_str = ("Epoch %d. Loss: %f, Train acc %f, "
                         % (epoch, epoch_loss,
                            epoch_acc))
        prev_time = cur_time
        logger.info(epoch_str + time_str + ', lr:' + str(trainer.learning_rate))
    logger.info('模型最好的准确率是:%f' % best_val_score)


def train_mixup(net, train_dataloader, valid_dataloader, num_epochs, batch_size, lr, wd, ctx, dtype, logger,
                warmup_epochs=5):
    # 学习率调整
    num_batches = len(train_dataloader)
    lr_scheduler = LRSequential([
        LRScheduler('linear', base_lr=0, target_lr=lr,
                    nepochs=warmup_epochs, iters_per_epoch=num_batches),
        LRScheduler('cosine', base_lr=lr, target_lr=0,
                    nepochs=num_epochs - warmup_epochs,
                    iters_per_epoch=num_batches)
    ])
    opt_params = {'learning_rate': lr, 'momentum': 0.9, 'wd': wd, 'lr_scheduler': lr_scheduler}
    if dtype != 'float32':
        opt_params['multi_precision'] = True
    trainer = gluon.Trainer(net.collect_params(), 'nag', opt_params)
    # 定义loss函数和准确率的计算
    Loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)
    train_loss = mx.metric.Loss()
    metric = mx.metric.RMSE()
    prev_time = datetime.datetime.now()
    best_val_score = 0
    alpha = 1
    logger.info("Start training with mixup.")
    for epoch in range(1, num_epochs + 1):
        train_loss.reset()
        metric.reset()
        for data, label in train_dataloader:
            lam = np.random.beta(alpha, alpha)
            if epoch >= (num_epochs - 20):
                lam = 1
            Xs = gluon.utils.split_and_load(data, ctx)
            ys = gluon.utils.split_and_load(label, ctx)
            trans = [lam * X + (1 - lam) * X[::-1] for X in Xs]
            labels = []
            for Y in ys:
                y1 = label_transform(Y)
                y2 = label_transform(Y[::-1])
                labels.append(lam * y1 + (1 - lam) * y2)
            with autograd.record():
                y_hats = [net(X.astype(dtype, copy=False)) for X in trans]
                losses = [Loss(y_hat, y.astype(dtype, copy=False)) for y_hat, y in zip(y_hats, labels)]
            for l in losses:
                l.backward()
            trainer.step(batch_size)
            train_loss.update(0, losses)
            metric.update(labels, y_hats)
        _, epoch_loss = train_loss.get()
        _, epoch_rmse = metric.get()
        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = ", Time %02d:%02d:%02d" % (h, m, s)
        if valid_dataloader is not None:
            val_loss, val_acc = test(net, ctx, valid_dataloader, dtype)
            if val_acc > best_val_score:
                best_val_score = val_acc
                net.save_parameters(config.MODEL_PATH)
            epoch_str = ("Epoch %d. Loss: %f, Train Rmse %f,Test Loss: %f ,Test acc %f" % (
                epoch, epoch_loss, epoch_rmse, val_loss, val_acc))
        else:
            epoch_str = ("Epoch %d. Loss: %f, Train Rmse %f, "
                         % (epoch, epoch_loss,
                            epoch_rmse))
        prev_time = cur_time
        logger.info(epoch_str + time_str + ', lr ' + str(trainer.learning_rate))
    logger.info('模型最好的准确率是:%f' % best_val_score)


def train_base_net(train_dataloader, test_dataloader, logger):
    base_net = get_model(name=config.BASE_MODEL_NAME, num_classes=config.NUM_CLASSES)
    base_net.output.initialize(init.Xavier(), ctx=config.CTX)
    # base_net.output.collect_params().setattr('lr_mult', 10)
    base_net.collect_params().reset_ctx(config.CTX)
    base_net.hybridize()
    if config.DTYPE != 'float32':
        base_net.cast('float16')
    for k, v in base_net.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
    train_basic(net=base_net,
                train_dataloader=train_dataloader,
                valid_dataloader=test_dataloader,
                num_epochs=50,
                batch_size=config.BATCH_SIZE,
                lr=config.LR,
                wd=config.WEIGHT_DECAY,
                ctx=config.CTX,
                dtype=config.DTYPE,
                logger=logger
                )


def train_enhance_net(train_dataloader, test_dataloader, logger):
    net = get_model(name=config.MODEL_NAME, num_classes=config.NUM_CLASSES)
    net.initialize(init.MSRAPrelu(), ctx=config.CTX)
    net.collect_params().reset_ctx(config.CTX)
    net.load_parameters(config.BASE_MODEL_PATH, allow_missing=True)
    net.hybridize()
    if config.DTYPE != 'float32':
        net.cast('float16')
    for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
    train_mixup(net=net,
                train_dataloader=train_dataloader,
                valid_dataloader=test_dataloader,
                num_epochs=config.NUM_EPOCHES,
                batch_size=config.BATCH_SIZE,
                lr=config.LR,
                wd=config.WEIGHT_DECAY,
                ctx=config.CTX,
                dtype=config.DTYPE,
                logger=logger
                )


if __name__ == '__main__':
    makedirs('./save_params')
    makedirs('./logs')
    for i in range(1, 10+1):
        config.BASE_MODEL_PATH = './save_params/%s_%s_time_%s.params' % (
            config.DATASET_NAME.lower(), config.BASE_MODEL_NAME, str(i))
        config.MODEL_PATH = './save_params/%s_%s_time_%s.params' % (
            config.DATASET_NAME.lower(), config.MODEL_NAME, str(i))
        config.LOGGING_FILE = './logs/%s_%s_time_%s.log' % (config.DATASET_NAME.lower(), config.MODEL_NAME, str(i))
        config.RANDOM_STATE = i
        ######
        logger = logging.getLogger(config.MODEL_PATH)
        filehandler = logging.FileHandler(config.LOGGING_FILE, mode='w')
        streamhandler = logging.StreamHandler()
        logger.setLevel(logging.INFO)
        logger.addHandler(filehandler)
        logger.addHandler(streamhandler)
        logger.info(config)
        ######
        logger.info('The dataset is %s' % config.DATASET_NAME.lower())
        train_dataloader, test_dataloader = get_data_loader(logger=logger)
        train_base_net(train_dataloader, test_dataloader, logger)
        logger.info('Start training: %s' % config.MODEL_NAME)
        train_enhance_net(train_dataloader, test_dataloader, logger)
