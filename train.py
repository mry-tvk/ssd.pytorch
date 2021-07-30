from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
from tqdm import tqdm
from layers.box_utils import jaccard, nms


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

MODEL_PATHS = {
    'body': 'ssd300_BFbootstrapBissau4p5k_prebossou_best.pth',
    'face': 'ssd300_CFbootstrap_85000.pth'
}
datasets ={
    'face':'../datasets/dataset_ft_face',
    'body': '../datasets/dataset_ft_body'
}
DEFAULT_FACE_DETECT_VISUAL_THRESHOLD = 0.37


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='COCO', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=datasets['body'], # VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default='weights/ssd300_BFbootstrapBissau4p5k_prebossou_best.pth', type=str, #
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=False, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization; make sure you run "python -m visdom.server" in terminal.')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('-t', dest='visual_threshold', default=DEFAULT_FACE_DETECT_VISUAL_THRESHOLD, type=float, help='Confidence threshold for detection')
parser.add_argument('--output_folder', default="outputs",
                help='Result root directory path')
parser.add_argument('--roi', default="body",
                help='Either face or body')
parser.add_argument('--optimizer', '--optimizer', default='SGD', type=str,
                    help='SGD or ADAM')
parser.add_argument('--num_frozen_layers', default=35,
                    help='total_layers=55 = vgg:35 -> extras:8 -> loc: 6, conf: 6')
args = parser.parse_args()

# Error: "yield from torch.randperm(n, generator=generator).tolist(). RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'"
# ref: https://discuss.pytorch.org/t/distributedsampler-runtimeerror-expected-a-cuda-device-type-for-generator-but-found-cpu/103594
# if torch.cuda.is_available():
#     if args.cuda:
#         torch.set_default_tensor_type('torch.cuda.FloatTensor')
#     if not args.cuda:
#         print("WARNING: It looks like you have a CUDA device, but aren't " +
#               "using CUDA.\nRun with --cuda for optimal training speed.")
#         torch.set_default_tensor_type('torch.FloatTensor')
# else:
#     torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

def train():
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root, image_set='%s_coco_train' % args.roi,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
        dataset_val = COCODetection(root=args.dataset_root, image_set='%s_coco_eval' % args.roi,
                                transform=BaseTransform(cfg['min_dim'], MEANS))

    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))

    if args.visdom:
        import visdom
        viz = visdom.Visdom()

    ssd_net = build_ssd('test', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net
    
    net.phase = 'train'
    ssd_net.phase = 'train'

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()
        # device = torch.device('cuda' if args.cuda else 'cpu')

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot(viz, 'Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot(viz, 'Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=False) # according to https://github.com/pytorch/pytorch/issues/57273
    
    data_loader_val = data.DataLoader(dataset_val, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=False, collate_fn=detection_collate,
                                  pin_memory=False)

    
    loss_val, loss_l_val, loss_c_val, duration_val = eval(net, data_loader_val, criterion, args)
    avg_ious, duration = test(net, 0, data_loader_val, args)

    # avg_ious = -1
    print("before fine-tuning", ' || Loss-val: %.4f ||' % (loss_val.item()), ' IOU-val: %.4f ||' % (avg_ious))
    net.phase = 'train'

    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):
        # check if new epoch
        if iteration != 0 and (iteration % epoch_size == 0):
            # epoch counter
            epoch += 1 #ref: https://github.com/amdegroot/ssd.pytorch/issues/234
            net.eval()
            
            # evaluation
            loss_val, loss_l_val, loss_c_val, duration_val = eval(net, data_loader_val, criterion, args)
            avg_ious, duration = test(net, epoch, data_loader_val, args)
            print('\nepoch ' + repr(epoch) + ' || iter ' + repr(iteration) + ' || Loss-train: %.4f ||' % ((loc_loss+conf_loss)/epoch_size))
            print('epoch ' + repr(epoch) + ' || iter ' + repr(iteration) + ' || Loss-val: %.4f ||' % (loss_val.item()), ' IOU-val: %.4f ||' % (avg_ious))
            
            if args.visdom:
                update_vis_plot(viz, epoch, loc_loss, conf_loss, epoch_plot, None,
                                'append', epoch_size)
            
            # reset epoch loss counters
            net.train()
            net.phase = 'train'

            loc_loss = 0
            conf_loss = 0

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        # images, targets = next(batch_iterator)

        try:
            images, targets, paths = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets, paths = next(batch_iterator)
                
        if args.cuda:
            images = images.cuda() # Variable(images.cuda())
            targets = [ann.cuda() for ann in targets]
        # targets = [torch.FloatTensor(ann).cuda() for ann in targets]
        # ref: https://stackoverflow.com/questions/61720460/volatile-was-removed-and-now-had-no-effect-use-with-torch-no-grad-instread
        # [Variable(ann.cuda(), volatile=True) for ann in targets]
        
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        # changed all data[0] to item() due to pytorch version:
        # ref: https://github.com/amdegroot/ssd.pytorch/issues/421
        loc_loss += loss_l.item() #.data[0]
        conf_loss += loss_c.item() #data[0]

        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')

        if args.visdom:
            update_vis_plot(viz, iteration, loss_l.item(), loss_c.item(),
                            iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 'weights/ssd300_COCO_' +
                       repr(iteration) + '.pth')
    torch.save(ssd_net.state_dict(),
               args.save_folder + '' + args.dataset + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    # ref (warning): UserWarning: nn.init.xavier_uniform is now deprecated in favor of nn.init.xavier_uniform_.init.xavier_uniform(param)
    init.xavier_uniform_(param)
    # init.xavier_uniform(param)
    # init.xavier_uniform_.init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(viz, _xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(viz, iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )

def eval(net, data_loader, criterion, args):

    # this determines what kind of output is produced, we want the same as during training
    net.phase = 'train' 
    
    # but we still don't what the model to learn
    net.eval() 
    
    with torch.no_grad():
        for (images, targets, paths) in data_loader: #tqdm(data_loader, desc="train", ncols=0, disable=True):
            if args.cuda:
                images = images.cuda()
                targets = [torch.FloatTensor(ann).cuda() for ann in targets]

            # forward
            t0 = time.time()
            out = net(images)
            
            loss_l, loss_c = criterion(out, targets) #.data) #.data)
            loss = loss_l + loss_c
            
            t1 = time.time()
            duration = t1 - t0
    return loss, loss_l, loss_c, duration

def test(net, epoch_id, data_loader, args):
    # because net is moved to torch.nn.DataParallel, it cannot capture features directly
    net.phase = 'test'
    
    net.eval()
    for images, targets, paths in tqdm(data_loader, desc="eval", ncols=0, disable=True):
        if args.cuda:
            images = images.cuda()
            targets = [torch.FloatTensor(ann).cuda() for ann in targets] 

        # forward
        t0 = time.time()
        with torch.no_grad():
            net_detections = net(images) #idx, loc, conf, prior

        ious = 0
        n_targets = 0

        # take a sample of images for visualization
        idx = np.random.choice(images.shape[0], min(images.shape[0], 10), replace=False)

        for i in idx: # iterate over images
            img = images[i].clone()
            scale = torch.Tensor([img.shape[1], img.shape[2], img.shape[1], img.shape[2]])
            score = net_detections[i, 1, :, 0]
            pt = (net_detections[i, 1, :, 1:] * scale)
            target_i = (targets[i][:,:4] * scale)

            best_boxes = nms(pt, score.clone(), thr=args.visual_threshold) # [n_priors], n_good_boxes
            iou = jaccard(target_i, pt) # [n_targets, n_priors]
            n_targets += target_i.shape[0]
            ious += iou[:,0].sum().item()
            
            filename = os.path.join(args.output_folder, 'images', str(epoch_id), os.path.basename(paths[i]))
            visualize(img, target_i, pt, score, iou, best_boxes, filename, data_loader.dataset.transform.mean)

        t1 = time.time()
        duration = t1 - t0
        
    avg_ious = (ious/n_targets) if n_targets > 0 else 0
    return avg_ious, duration

# means = data_loader.dataset.transform.mean
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize(img, target_i, pt, score, iou, best_boxes, filename, means):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    
    # print("Move data to CPU before visualization")
    img = img.cpu()
    target_i = target_i.cpu()
    pt = pt.cpu()

    out_img = img.cpu().permute(1,2,0).numpy()
    out_img = out_img[:, :, (2, 1, 0)]
    out_img += np.array(means)
    fig, ax = plt.subplots()
    ax.imshow(out_img.astype('int'))
    for b in range(best_boxes[1]):
        bb = best_boxes[0][b].item()
        coords = ( max(float(pt[bb, 0]), 0.0),
        max(float(pt[bb, 1]), 0.0),
        min(float(pt[bb, 2]), img.shape[2]),
        min(float(pt[bb, 3]), img.shape[1]))
        rect = patches.Rectangle((coords[0], coords[1]), coords[2] - coords[0], coords[3] - coords[1], lw=1, edgecolor=plt.cm.GnBu(250), facecolor='none')
        ax.add_patch(rect)
        ax.text(coords[0], coords[1] - 5, "score: %0.2f" % score[bb].item(), fontsize='xx-small', bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))
    for t in range(target_i.shape[0]):
        rect = patches.Rectangle((target_i[t, 0], target_i[t, 1]), target_i[t, 2] - target_i[t, 0], target_i[t, 3] - target_i[t, 1], ls='--', lw=1, edgecolor=plt.cm.GnBu(200), facecolor='none')
        ax.add_patch(rect)
        ax.text(target_i[t, 0], target_i[t, 3] + 10, "iou: %0.2f" % iou[t,0].item(), fontsize='xx-small', bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))
    plt.axis('off')
    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':
    train()
