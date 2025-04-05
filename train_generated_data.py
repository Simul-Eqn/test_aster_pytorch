from __future__ import absolute_import
import sys
sys.path.append('./')

import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import os.path as osp
import numpy as np
import math
import time

import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader, SubsetRandomSampler

from lib import datasets, evaluation_metrics, models
from lib.models.model_builder import ModelBuilder
from lib.datasets.dataset import AlignCollate, labels2strs 
from lib.loss import SequenceCrossEntropyLoss
from lib.trainers import Trainer
from lib.evaluators import Evaluator
from lib.utils.logging import Logger, TFLogger
from lib.utils.serialization import load_checkpoint, save_checkpoint
from lib.utils.osutils import make_symlink_if_not_exists

from synthetic_dataset import SyntheticDataset


def get_data(voc_type, max_len:int, initial_seed, num_samples:int, height, width, batch_size, workers, is_train, keep_ratio, device):
    #print("MAX LENGHT", max_len)
    dataset = SyntheticDataset(num_samples, max_len, initial_seed, voc_type=voc_type)
    #print('total images: ', len(dataset))

    if is_train:
        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
        shuffle=True, pin_memory=True, drop_last=True,
        collate_fn=AlignCollate(imgH=height, imgW=width, keep_ratio=keep_ratio), generator=torch.Generator(device=device).manual_seed(initial_seed+15))
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True, drop_last=False,
        collate_fn=AlignCollate(imgH=height, imgW=width, keep_ratio=keep_ratio))

    return dataset, data_loader


def get_dataset(voc_type, max_len, initial_seed, num_samples):
    dataset = SyntheticDataset(num_samples, max_len, initial_seed, voc_type=voc_type)
    print('total image: ', len(dataset))
    return dataset


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    args.cuda = args.cuda and torch.cuda.is_available()
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if args.cuda: 
        torch.set_default_device('cuda')
    torch.set_default_dtype(torch.float32)
    
    
    # Redirect print to both console and log file
    if not args.evaluate:
        # make symlink
        make_symlink_if_not_exists(osp.join(args.real_logs_dir, args.logs_dir), osp.dirname(osp.normpath(args.logs_dir)))
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
        train_tfLogger = TFLogger(osp.join(args.logs_dir, 'train'))
        eval_tfLogger = TFLogger(osp.join(args.logs_dir, 'eval'))

    # Save the args to disk
    if not args.evaluate:
        cfg_save_path = osp.join(args.logs_dir, 'cfg.txt')
        cfgs = vars(args)
        with open(cfg_save_path, 'w') as f:
            for k, v in cfgs.items():
                f.write('{}: {}\n'.format(k, v))

    # Create data loaders
    if args.height is None or args.width is None:
        args.height, args.width = (32, 100)

    if not args.evaluate: 
        train_dataset, train_loader = \
        get_data(args.voc_type, int(args.max_len), args.seed+10, int(args.num_train),
                args.height, args.width, args.batch_size, args.workers, True, args.keep_ratio, device='cuda' if args.cuda else 'cpu')
    
    #print("NUM TEST", args.num_test)

    test_dataset, test_loader = \
        get_data(args.voc_type, int(args.max_len), args.seed+100, int(args.num_test),
                args.height, args.width, args.batch_size, args.workers, False, args.keep_ratio, device='cuda' if args.cuda else 'cpu')
    
    #print("MAX LEN:", args.max_len)

    if args.evaluate:
        max_len = test_dataset.max_len
    else:
        max_len = max(train_dataset.max_len, test_dataset.max_len)
        train_dataset.max_len = test_dataset.max_len = max_len
    #print("MAX LEN", train_dataset.max_len, test_dataset.max_len, max_len)
    # Create model
    model = ModelBuilder(arch=args.arch, rec_num_classes=test_dataset.rec_num_classes,
                        sDim=args.decoder_sdim, attDim=args.attDim, max_len_labels=max_len,
                        eos=test_dataset.char2id[test_dataset.EOS], STN_ON=args.STN_ON)

    # Load from checkpoint
    if args.evaluation_metric == 'accuracy':
        best_res = 0
    elif args.evaluation_metric == 'editdistance':
        best_res = math.inf
    else:
        raise ValueError("Unsupported evaluation metric:", args.evaluation_metric)
    start_epoch = 0
    start_iters = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])

        # compatibility with the epoch-wise evaluation version
        if 'epoch' in checkpoint.keys():
            start_epoch = checkpoint['epoch']
        else:
            start_iters = checkpoint['iters']
            start_epoch = int(start_iters // len(train_loader)) if not args.evaluate else 0
            best_res = checkpoint['best_res']
            print("=> Start iters {}  best res {:.1%}"
                .format(start_iters, best_res))
  
    if args.cuda:
        device = torch.device("cuda")
        model = model.to(device)
        model = nn.DataParallel(model)

    # Evaluator
    evaluator = Evaluator(model, args.evaluation_metric, args.cuda)

    if args.evaluate:
        if len(args.vis_dir) > 0:
            vis_dir = osp.join(args.logs_dir, args.vis_dir)
            if not osp.exists(vis_dir):
                os.makedirs(vis_dir)
        else:
            vis_dir = None

        start = time.time()
        evaluator.evaluate(test_loader, dataset=test_dataset, vis_dir=vis_dir)
        print('it took {0} s.'.format(time.time() - start))
        return

    # Optimizer
    param_groups = model.parameters()
    param_groups = filter(lambda p: p.requires_grad, param_groups)
    optimizer = optim.Adadelta(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4,5], gamma=0.1)

    # Trainer
    loss_weights = {}
    loss_weights['loss_rec'] = 1.
    if args.debug:
        args.print_freq = 1
    trainer = Trainer(model, args.evaluation_metric, args.logs_dir, 
                        iters=start_iters, best_res=best_res, grad_clip=args.grad_clip,
                        use_cuda=args.cuda, loss_weights=loss_weights)

    # Start training
    evaluator.evaluate(test_loader, step=0, tfLogger=eval_tfLogger, dataset=test_dataset)
    print("EPOCHS", start_epoch, args.epochs)
    for epoch in range(start_epoch, args.epochs):
        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']
        trainer.train(epoch, train_loader, optimizer, current_lr,
                    print_freq=args.print_freq,
                    train_tfLogger=train_tfLogger, 
                    is_debug=args.debug,
                    evaluator=evaluator, 
                    test_loader=test_loader, 
                    eval_tfLogger=eval_tfLogger,
                    test_dataset=test_dataset, 
                    test_freq=100)

    # Final test
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset=test_dataset)

    # Close the tensorboard logger
    train_tfLogger.close()
    eval_tfLogger.close()


if __name__ == '__main__':
  # parse the config
  from config import get_args 
  args = get_args(sys.argv[1:])
  main(args)