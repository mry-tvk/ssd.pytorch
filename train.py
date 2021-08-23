from data import *
import copy
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

# get the Azure ML run object
from azureml.core.run import Run
run = Run.get_context()

# visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

MODEL_PATHS = {
    'body': 'ssd300_BFbootstrapBissau4p5k_prebossou_best.pth',
    'face': 'ssd300_CFbootstrap_85000.pth',
    'fine_tuned_face': 'ssd300_run24_epoch149_last_frozen33.pth' #'ssd300_CFbootstrap_85000.pth' #
}
# 'ssd300_CFbootstrap_85000.pth'
# 'ssd300_run28_epoch100_best_frozen-1.pth'
# 'ssd300_run24_epoch149_last_frozen33.pth' 
# 'ssd300_CFbootstrap_85000.pth'

datasets ={
    'face':'../datasets/dataset_ft_face',
    'body': '../datasets/dataset_ft_body'
}
DEFAULT_FACE_DETECT_VISUAL_THRESHOLD = 0.3 #0.37
# example: tape J_000080093.png

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='COCO', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=datasets['face'], # VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default='weights/'+MODEL_PATHS['face'], type=str, #
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
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
parser.add_argument('--AML', default=True, type=str2bool,
                    help='Use AML for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('-t', dest='visual_threshold', default=DEFAULT_FACE_DETECT_VISUAL_THRESHOLD, type=float, help='Confidence threshold for detection')
parser.add_argument('--output_folder', default="outputs",
                help='Result root directory path')
parser.add_argument('--roi', default="face",
                help='Either face or body')
parser.add_argument('--optimizer', '--optimizer', default='SGD', type=str,
                    help='SGD or ADAM')
parser.add_argument('--num_frozen_layers', default=-1,
                    help='total_layers=55 = vgg:35 -> extras:8 -> loc: 6, conf: 6')
parser.add_argument('--eval_interval', default=1, type=int, help='No. of epochs between two evaluation measure')
parser.add_argument('--test_interval', default=10, type=int,
                    help='No. of epochs between two test and checkpoint')
parser.add_argument('--early_stopping_patience', default=20, type=int,
                    help='No. of epochs waits before termination')
parser.add_argument('--only_validation', default=True, type=str2bool,
                    help='Stop right after initial validation')
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

def toggle_phase(net, new_phase='train'):
    if args.cuda:
        net.module.phase = new_phase
        return
    net.phase = new_phase

# -------- only validation -------- #
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def validation_with_tr_check(threshold_interval=0.1):

    # load data
    args.dataset_root = COCO_ROOT
    cfg = coco
    
    val_root = 'Users/matavako/train-hyperparameter-tune-deploy-with-pytorch/datasets/dataset_val'
    # val_root = '../datasets/dataset_val'
    dataset_validation = COCODetection(root=val_root, image_set='instances_val_%s' % args.roi,
                        transform=BaseTransform(cfg['min_dim'], MEANS))

    data_loader_validation = data.DataLoader(
                                dataset_validation,
                                args.batch_size,
                                num_workers=args.num_workers,
                                collate_fn=detection_collate,
                                pin_memory=True)
    
    models = [
        'ssd300_CFbootstrap_85000.pth',
        'ssd300_run28_epoch100_best_frozen-1.pth',
        'ssd300_run24_epoch149_last_frozen33.pth'
    ]
    colors = ['r', 'b', 'g']
    sk_fprs ={}
    sk_tprs = {}
    # model-test
    for color, model_name in zip(colors, models):
        MODEL_PATHS["fine_tuned_%s"%args.roi] = model_name
        # load model
        ssd_net = build_ssd('test', cfg['min_dim'], cfg['num_classes'])
        net = ssd_net
        net.load_weights('Users/matavako/train-hyperparameter-tune-deploy-with-pytorch/ssd.pytorch/weights/'+MODEL_PATHS["fine_tuned_%s"%args.roi])
        # net.load_weights('weights/'+MODEL_PATHS["fine_tuned_%s"%args.roi])
        
        if args.cuda:
            net = torch.nn.DataParallel(ssd_net)
            cudnn.benchmark = True
        
        val_losses = []
        ov_trs = []
        # overlap interval test
        overlap_thres = 0.4
        # while overlap_thres <= 1.0:
        ov_trs.append(overlap_thres)
        pos_overlap_thres = overlap_thres
        neg_overlap_thres = 0.5
        criterion = MultiBoxLoss(cfg['num_classes'], pos_overlap_thres, True, 0, True, 3, neg_overlap_thres,
                                False, args.cuda)

        # for conf_thresholds
        # init plot
        fig, ax = plt.subplots()

        # init vars
        loss_vals = []
        avg_ious_vals = []
        recalls = []
        precs = []
        tprs = []
        fprs = []
        
        trs = []
        tr_conf = 0.5

        avg_ious_val, recall, prec, tpr, fpr, preds, actuals = test(net, 0, data_loader_validation, args, nsamples='all', overlap_thres=overlap_thres)

        lr_auc = roc_auc_score(actuals, preds)
        sk_fpr, sk_tpr, _ = roc_curve(actuals, preds)
        sk_tprs[model_name]=sk_tpr
        sk_fprs[model_name]=sk_fpr
    
    plt.plot(sk_fprs[models[0]], sk_tprs[models[0]], linestyle='--', color= colors[0], label= models[0])
    plt.plot(sk_fprs[models[1]], sk_tprs[models[1]], linestyle='--', color= colors[1], label= models[1])
    plt.plot(sk_fprs[models[2]], sk_tprs[models[2]], linestyle='--', color= colors[2], label= models[2])
    
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    # plt.show()
    filename = "sk_roc_curves" # "tr_plot_" + MODEL_PATHS["fine_tuned_%s"%args.roi].replace('.pth','') + '_overlaptr'+str(overlap_thres) + '.png'
    figure_path = os.path.join("Users/matavako/train-hyperparameter-tune-deploy-with-pytorch/ssd.pytorch", args.output_folder, filename)

    plt.savefig(figure_path)
    plt.close()
    # tr_conf = 0.0
    # # tresholds to test
    # while tr_conf < 0.92:
    #     args.visual_threshold = tr_conf

    #     # validation
    #     with torch.no_grad():
    #         loss_val, loss_l_val, loss_c_val, duration_val = eval(net, data_loader_validation, criterion, args)
    #         avg_ious_val, recall, prec, tpr, fpr, preds, actuals = test(net, 0, data_loader_validation, args, nsamples='all', overlap_thres=overlap_thres)
    #         print("validation:", ' || Loss-val: %.4f ||' % (loss_val.item()), ' IOU-val: %.4f ||' % (avg_ious_val))
    #         print(avg_ious_val, recall, prec, tpr, fpr)
    #         trs.append(tr_conf)
    #         loss_vals.append(loss_val.item())
    #         avg_ious_vals.append(avg_ious_val)
    #         recalls.append(recall)
    #         precs.append(prec)
    #         tprs.append(tpr)
    #         fprs.append(fpr)

    #     tr_conf += threshold_interval
        
    # plot
    # plt.plot(trs, loss_vals, color='g')
    # plt.plot(trs, avg_ious_vals, color='b', label="IOUs")
    # # plt.plot(trs, recalls, color='g', label="Recall")
    # # plt.plot(trs, precs, color='r', label="Prec")
    # plt.plot(trs, tprs, color='y', label="TPR")
    # plt.plot(trs, fprs, color='c', label="FPR")
    # plt.plot(fprs, tprs, color='k', label="ROC: TPR(y)-FPR(x)")

    # # labels the x axis with Thresholds
    # plt.xlabel('Conf-thres')
    
    # # labels the y axis with IOU
    # plt.ylabel('Metrics')
    # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    # plt.tight_layout()

    # filename = "tr_plot_" + MODEL_PATHS["fine_tuned_%s"%args.roi].replace('.pth','') + '_overlaptr'+str(overlap_thres) + '.png'
    # figure_path = os.path.join("Users/matavako/train-hyperparameter-tune-deploy-with-pytorch/ssd.pytorch", args.output_folder, filename)
    # # figure_path = os.path.join(args.output_folder, filename)

    # plt.savefig(figure_path)
    # plt.close()
    # val_losses.append(np.mean(loss_vals))
    # overlap_thres += threshold_interval
    
    # plt.plot(ov_trs, val_losses, color='k', label="Val-loss")

    # # labels the x axis with Thresholds
    # plt.xlabel('overlap-thres')
    
    # # labels the y axis with IOU
    # plt.ylabel('val_loss')
    # plt.legend(loc=4)

    # filename = "overlap_tr_plot_" + MODEL_PATHS["fine_tuned_%s"%args.roi].replace('.pth','') + '.png'
    # figure_path = os.path.join("Users/matavako/train-hyperparameter-tune-deploy-with-pytorch/ssd.pytorch", args.output_folder, filename)
    # # figure_path = os.path.join(args.output_folder, filename)
    # plt.savefig(figure_path)
    # plt.close()

# if args.only_validation:
#     exit(0)



def validation():
    if args.dataset == 'COCO':
        # load data
        args.dataset_root = COCO_ROOT
        cfg = coco
        
        val_root = 'Users/matavako/train-hyperparameter-tune-deploy-with-pytorch/datasets/dataset_val'
        # val_root = '../datasets/dataset_val'
        dataset_validation = COCODetection(root=val_root, image_set='instances_val_%s' % args.roi,
                            transform=BaseTransform(cfg['min_dim'], MEANS))

        data_loader_validation = data.DataLoader(
                                    dataset_validation,
                                    args.batch_size,
                                    num_workers=args.num_workers,
                                    collate_fn=detection_collate,
                                    pin_memory=True)
        
        criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)
        
        # load model
        ssd_net = build_ssd('test', cfg['min_dim'], cfg['num_classes'])
        net = ssd_net
        net.load_weights('Users/matavako/train-hyperparameter-tune-deploy-with-pytorch/ssd.pytorch/weights/'+MODEL_PATHS["fine_tuned_%s"%args.roi])
        # net.load_weights('weights/'+MODEL_PATHS["fine_tuned_%s"%args.roi])
        
        if args.cuda:
            net = torch.nn.DataParallel(ssd_net)
            cudnn.benchmark = True

        with torch.no_grad():
            loss_val, loss_l_val, loss_c_val, duration_val = eval(net, data_loader_validation, criterion, args)
            avg_ious_val, duration_val = test(net, 0, data_loader_validation, args, nsamples='all')
            print("validation:", ' || Loss-val: %.4f ||' % (loss_val.item()), ' IOU-val: %.4f ||' % (avg_ious_val))
            
            with open(os.path.join("Users/matavako/train-hyperparameter-tune-deploy-with-pytorch/ssd.pytorch", args.output_folder, "results.txt"), 'w') as f:
            # with open(os.path.join(args.output_folder, "results.txt"), 'w') as f:
                f.write("avg_ious_val: %f\n" % avg_ious_val)
                f.write("validation:" + ' || Loss-val: %.4f ||' % (loss_val.item()) + ' IOU-val: %.4f ||' % (avg_ious_val))
                f.write("validation:" + ' || Loss-l-val: %.4f ||' % (loss_l_val.item()) + ' Loss-c-val: %.4f ||' % (loss_c_val.item()))
            if args.only_validation:
                exit(0)
    else:
        print("We have not implemented the non-COCO implementattion!")

def train():
    # --------------------------- if training ----------------------------------------------
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        
        dataset = COCODetection(root=args.dataset_root, image_set='via_ft_%s_coco_train' % args.roi,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
        dataset_val = COCODetection(root=args.dataset_root, image_set='via_ft_%s_coco_eval' % args.roi,
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

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)
    
    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    # freeze the layers
    args.num_frozen_layers = int(args.num_frozen_layers)
    if args.num_frozen_layers != -1:
        freeze_layers_up_to(net, args.num_frozen_layers)

    layer_no = 0
    for (name, module) in net.named_children():
        print(name)
        for layer in module.children():
            for param in layer.parameters():
                print('Module {}: "{}" required_grad={}!'.format(layer_no, name, param.requires_grad))
            layer_no += 1
    
    # move to cuda
    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
    #                       weight_decay=args.weight_decay)
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
                            weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
                            momentum=args.momentum, weight_decay=args.weight_decay)

    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

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

    
    with torch.no_grad():
        loss_val, loss_l_val, loss_c_val, duration_val = eval(net, data_loader_val, criterion, args)
        avg_ious_val, duration_val = test(net, 0, data_loader_val, args)

    # avg_ious = -1
    print("before fine-tuning", ' || Loss-val: %.4f ||' % (loss_val.item()), ' IOU-val: %.4f ||' % (avg_ious_val))
    toggle_phase(net, new_phase='train')

    # overall eval metrics
    best_val_iou = avg_ious_val
    best_val_loss = loss_val
    best_val_epoch = 0

    step_index = 0

    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0

    last_saved_loss = loss_val
    termination_flag = False

    # best trained network based on val_loss
    # best_model_wts = copy.deepcopy(net.state_dict())

    # create batch iterator
    net.train()
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):
        # check if new epoch
        if iteration != 0 and (iteration % epoch_size == 0):
            # epoch counter
            epoch += 1 #ref: https://github.com/amdegroot/ssd.pytorch/issues/234
            
            # evaluation
            if epoch % args.eval_interval == 0:
                net.eval()
                eval_size = epoch_size * args.eval_interval
                
                with torch.no_grad():
                    loss_val, loss_l_val, loss_c_val, duration_val = eval(net, data_loader_val, criterion, args)
                    
                    # overall eval metrics
                    if best_val_loss > loss_val:
                        best_val_loss = loss_val
                        best_val_epoch = epoch
                    elif epoch - best_val_epoch >= args.early_stopping_patience:
                        torch.save(ssd_net.state_dict(), os.path.join(args.output_folder, 'weights/ssd300_COCO_iter' +
                                '_epoch' + repr(epoch) + '_last.pth'))
                        termination_flag = True


                print('\nepoch ' + repr(epoch) + ' || iter ' + repr(iteration) + ' || Loss-train: %.4f ||' % ((loc_loss+conf_loss)/eval_size))
                print('epoch ' + repr(epoch) + ' || iter ' + repr(iteration) + ' || Loss-val: %.4f ||' % (loss_val.item()), ' IOU-val: %.4f ||' % (avg_ious_val))
                
                if args.visdom:
                    update_vis_plot(viz, epoch, loc_loss, conf_loss, epoch_plot, None,
                                    'append', eval_size)
                if args.AML:
                    # log the best val accuracy to AML run
                    run.log('loss_train', np.float((loc_loss+conf_loss)/eval_size))
                    run.log('loss_l_train', np.float(loc_loss/eval_size))
                    run.log('loss_c_train', np.float(conf_loss/eval_size))
                    # run.log('avg_iou_train', np.float(avg_ious))

                    run.log('avg_iou_val', np.float(avg_ious_val))
                    run.log('loss_val', np.float(loss_val.item()))
                    run.log('loss_l_val', np.float(loss_l_val.item()))
                    run.log('loss_c_val', np.float(loss_c_val.item()))

                    run.log('best_val_loss', np.float(best_val_loss))
                    run.log('best_val_iou', np.float(best_val_iou))
                
            # test & checkpoint
            if epoch % args.test_interval == 0:
                with torch.no_grad():
                    avg_ious_val, duration = test(net, epoch, data_loader_val, args)
                
                    # save best iou value
                    if best_val_iou < avg_ious_val:
                        best_val_iou = avg_ious_val

                    # checkpoint
                    print('Saving state, iter:', repr(iteration), ', epoch: ', repr(epoch))

                    # first time
                    if not os.path.exists(os.path.join(args.output_folder, 'weights')):
                        os.makedirs(os.path.join(args.output_folder, 'weights'))
                    
                    # best or last
                    if last_saved_loss > loss_val:
                        torch.save(ssd_net.state_dict(), os.path.join(args.output_folder, 'weights/ssd300_COCO_iter' +
                                '_epoch' + repr(epoch) + '_best.pth'))
                        last_saved_loss = loss_val

            # early termination check
            if termination_flag:
                break

            # reset epoch loss counters
            net.train()
            toggle_phase(net, new_phase='train')

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

    print('Best val loss: {:4f}'.format(best_val_loss))
    print('Best val Acc: {:4f}'.format(best_val_iou))

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
    toggle_phase(net, new_phase='train')
    
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

# def abs_distance(box_a, box_b):
#     """Compute the absolute distances of the centers of two sets of boxes. Here we operate on
#     ground truth boxes and default boxes.
#     E.g.:
#         abs(c_A - c_B)
#     Args:
#         box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
#         box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
#     Return:
#         abs distance: (tensor) Shape: [box_a.size(0), box_b.size(0)]
#     """
#     center_a = (((box_a[:, 2]-box_a[:, 0])/2.0).unsqueeze(1), (box_a[:, 3]-box_a[:, 1]/2.0).unsqueeze(1))
#     center_b = (((box_a[:, 2]-box_a[:, 0])/2.0).unsqueeze(1), (box_a[:, 3]-box_a[:, 1]/2.0).unsqueeze(1))
#     # ((box_a[:, 2]-box_a[:, 0]) *
#     #           (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
#     # area_b = ((box_b[:, 2]-box_b[:, 0]) *
#     #           (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
#     # union = area_a + area_b - inter
#     return inter / union  # [A,B]

def test(net, epoch_id, data_loader, args, nsamples=10, overlap_thres=0.5):
    '''
    nsample: number of samples to do iou evaluation
    '''
    # Recall was defined as the proportion of all positive examples ranked above a given rank, 
    # while precision is the proportion of all examples above that rank which are from the positive class. 
    
    # IOU current problems: 

    # because net is moved to torch.nn.DataParallel, it cannot capture features directly
    toggle_phase(net, new_phase='test')
    net.eval()
    # overlap_threshold = 0.45
    
    ious = 0
    n_targets = 0
    n_images = 0

    total_pos_pred = 0
    total_pos_act = 0
    total_neg_pred = 0
    total_neg_act = 0
    total_preds = 0
    total_tp = 0
    total_tn = 0

    preds = []
    actuals = []

    total_target_catched = 0

    for images, targets, paths in tqdm(data_loader, desc="eval", ncols=0, disable=True):
        device = 'cuda' if args.cuda else 'cpu'
        if args.cuda:
            images = images.cuda()
            # targets = [torch.FloatTensor(ann).cuda() for ann in targets] 
        # forward
        t0 = time.time()
        with torch.no_grad():
            net_detections = net(images) #idx, loc, conf, prior #batch_size, class, top_k, 5 (score, loc)

        n_sample_selection = images.shape[0] if nsamples=='all' else nsamples

        # take a sample of images for visualization
        idx = np.random.choice(images.shape[0], min(images.shape[0], n_sample_selection), replace=False)
        n_images += len(idx)

        # iterate over images
        for i in idx: 
            img = images[i].clone()
            scale = torch.Tensor([img.shape[1], img.shape[2], img.shape[1], img.shape[2]])
            score = net_detections[i, 1, :, 0]
            pt = (net_detections[i, 1, :, 1:] * scale)
            target_i = (targets[i][:,:4] * scale)

            # best_boxes ===> Threshold does not impact IOU calculation
            score[score < args.visual_threshold] = 0

            # iou calculation --- consider the max overlap for matching the target & preds
            iou = jaccard(target_i, pt) # [n_targets, n_priors]

            # assuming that score is sorted descending ==> it should be
            best_box_indexes = score.nonzero(as_tuple=True)[0]
            best_targets, best_targets_idx = iou.max(0)

            # nms ===> one is positive ==> two targets: most confident or most overlap?
            # each 
            # if the score is 0 => not part of the TP/FP 
            # TN => 

            # best_targets_idx = [tar_idx.item() for tar_idx, tar_overlap in zip(best_targets_idx, best_targets) if tar_overlap > overlap_threshold]
            matched = [1.0 if best_targets[pred_idx]  > overlap_thres else 0.0 for pred_idx in range(len(pt))]
            matched_idx = [best_targets_idx[pred_idx].item() for pred_idx in range(len(pt)) if best_targets[pred_idx]  > overlap_thres]
            
            total_tp += sum([1.0 if (sc > 0 and mat > 0) else 0.0 for sc, mat in zip(score, matched)])
            total_tn += sum([1.0 if (sc == 0 and mat == 0) else 0.0 for sc, mat in zip(score, matched)])

            # metrics
            total_pos_pred += len(best_box_indexes)
            total_pos_act += sum(matched)
            total_neg_act += (len(pt)-sum(matched)) 
            total_preds += len(pt)

            # pred vs act
            preds += score
            actuals += matched

            # unique correct predictions
            best_targets, best_targets_idx = iou.max(1)
            total_target_catched += len(set(matched_idx))

            # only keep the best instead of top_k 
            score = score[:len(best_box_indexes)]
            pt = pt[:len(best_box_indexes)]

            if len(pt) > 0:
                # iou calculation --- consider the max overlap for matching the target & preds
                iou = jaccard(target_i, pt) # [n_targets, n_priors]
                ious += iou[:,0].sum().item()

                # extending to target size with zero
                iou = iou[:,0] + torch.tensor([0]) * (target_i.shape[0]- iou.shape[0])
                # torch.zeros(target_i.shape[0]- iou.shape[0]) 
            else:
                # just keep it a zero vector
                iou = torch.zeros(target_i.shape[0]) #torch.tensor([0]) * (target_i.shape[0])

            n_targets += target_i.shape[0]

            # find max for each target ===> if there is zero target & we have false-positive, we are not
            # reflecting here
            # if iou.shape[1] > 0: # if there is at least one detection
            #    match_box_iou, match_idx = iou.max(1)
            #    # match_idx = match_idx.item()
            # ?
            # match_box_iou = match_box_iou + torch.tensor([0]) * (target_i.shape[0]- match_box_iou.shape[0])
            # ious += match_box_iou.sum().item()

            # visualize the target (& iou based on best-box) + nms-selected boxes (predictions)(& corresponding scores)
            filename = os.path.join("Users/matavako/train-hyperparameter-tune-deploy-with-pytorch/ssd.pytorch", args.output_folder, 'images', str(epoch_id), os.path.basename(paths[i]))
            # filename = os.path.join(args.output_folder, 'images', str(epoch_id), os.path.basename(paths[i]))
            visualize(img, target_i, pt, score, best_box_indexes=best_box_indexes, iou=iou, filename=filename, means=data_loader.dataset.transform.mean)
        t1 = time.time()
        duration = t1 - t0


    tpr = 1.0 if total_pos_pred <= 0 else (total_tp / total_pos_pred)
    fpr = 0.0 if total_neg_act <= 0 else (total_pos_pred - total_tp) / (total_pos_pred - total_tp + total_tn)
    
    recall = 1.0 if n_targets == 0 else total_target_catched / n_targets
    prec = 0.0 if total_pos_pred == 0 else total_tp / total_pos_pred

    avg_ious = (ious/n_targets) if n_targets > 0 else 0

    if args.only_validation:
        return avg_ious, recall, prec, tpr, fpr, preds, actuals

    return avg_ious, duration

def freeze_module_by_name(net, module_name):
    for (name, module) in net.named_children():
        if name == module_name:
            for param in module.parameters():
                    param.requires_grad = False
            print('Module "{}" was frozen!'.format(name))

def freeze_layers_up_to(net, to_layer=0):
    layer_counter = 0
    for (name, module) in net.named_children():
        if layer_counter >= to_layer:
            break
        elif not module.children():
            for param in module():
                param.requires_grad = False
            layer_counter += 1
        else:
            for layer in module.children():
                if layer_counter >= to_layer:
                    break
                for param in layer.parameters():
                    param.requires_grad = False
                
                print('Layer "{}" in module "{}" was frozen!'.format(layer_counter, name))
                layer_counter+=1

# means = data_loader.dataset.transform.mean

def visualize(img, target_i, pt, score, best_box_indexes, iou, filename, means):

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

    for b in range(len(best_box_indexes)): #best_boxes[1]):
        # each match_box ===>
        bb = best_box_indexes[b] #match_idx[b].item() # best_boxes[0][b].item()
        # if score[bb] < 0.37:
        #    continue
        coords = ( max(float(pt[bb, 0]), 0.0),
        max(float(pt[bb, 1]), 0.0),
        min(float(pt[bb, 2]), img.shape[2]),
        min(float(pt[bb, 3]), img.shape[1]))
        rect = patches.Rectangle((coords[0], coords[1]), coords[2] - coords[0], coords[3] - coords[1], lw=1, edgecolor=plt.cm.GnBu(250), facecolor='none')
        ax.add_patch(rect)
        ax.text(coords[0], coords[1] - 5, "score: %0.2f" % score[bb].item(), fontsize='xx-small', bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))

    # for t in range(target_i.shape[0]):
    #     # each corresponding target ===>
    #     rect = patches.Rectangle((target_i[b, 0], target_i[b, 1]), target_i[b, 2] - target_i[b, 0], target_i[b, 3] - target_i[b, 1], ls='--', lw=1, edgecolor=plt.cm.GnBu(200), facecolor='none')
    #     ax.add_patch(rect)
    #     ax.text(target_i[b, 0], target_i[b, 3] + 10, "iou: %0.2f" % iou[b].item(), fontsize='xx-small', bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))
    
    for t in range(target_i.shape[0]):
        rect = patches.Rectangle((target_i[t, 0], target_i[t, 1]), target_i[t, 2] - target_i[t, 0], target_i[t, 3] - target_i[t, 1], ls='--', lw=1, edgecolor=plt.cm.GnBu(200), facecolor='none')
        ax.add_patch(rect)
        ax.text(target_i[t, 0], target_i[t, 3] + 10, "iou: %0.2f" % iou[t].item(), fontsize='xx-small', bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))
    
    plt.axis('off')
    plt.savefig(filename)
    plt.close()

def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def net_eval_metric():
    # extract gt objects for this class
    class_recs = {}
    npos = 0

    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap

# def voc_eval(detpath,
#              annopath,
#              imagesetfile,
#              classname,
#              cachedir,
#              ovthresh=0.5,
#              use_07_metric=True):
#     """rec, prec, ap = voc_eval(detpath,
#                            annopath,
#                            imagesetfile,
#                            classname,
#                            [ovthresh],
#                            [use_07_metric])
# Top level function that does the PASCAL VOC evaluation.
# detpath: Path to detections
#    detpath.format(classname) should produce the detection results file.
# annopath: Path to annotations
#    annopath.format(imagename) should be the xml annotations file.
# imagesetfile: Text file containing the list of images, one image per line.
# classname: Category name (duh)
# cachedir: Directory for caching the annotations
# [ovthresh]: Overlap threshold (default = 0.5)
# [use_07_metric]: Whether to use VOC07's 11 point AP computation
#    (default True)
# """
# # assumes detections are in detpath.format(classname)
# # assumes annotations are in annopath.format(imagename)
# # assumes imagesetfile is a text file with each line an image name
# # cachedir caches the annotations in a pickle file
# # first load gt
#     if not os.path.isdir(cachedir):
#         os.mkdir(cachedir)
#     cachefile = os.path.join(cachedir, 'annots.pkl')
#     # read list of images
#     with open(imagesetfile, 'r') as f:
#         lines = f.readlines()
#     imagenames = [x.strip() for x in lines]
#     if not os.path.isfile(cachefile):
#         # load annots
#         recs = {}
#         for i, imagename in enumerate(imagenames):
#             recs[imagename] = parse_rec(annopath % (imagename))
#             if i % 100 == 0:
#                 print('Reading annotation for {:d}/{:d}'.format(
#                    i + 1, len(imagenames)))
#         # save
#         print('Saving cached annotations to {:s}'.format(cachefile))
#         with open(cachefile, 'wb') as f:
#             pickle.dump(recs, f)
#     else:
#         # load
#         with open(cachefile, 'rb') as f:
#             recs = pickle.load(f)


if __name__ == '__main__':
    if args.only_validation:
        validation()
        # validation_with_tr_check(threshold_interval=0.1)
    else:
        train()
