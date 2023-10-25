import os
import random
import argparse
import time
import math
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from datasets.mvtec import FSAD_Dataset_train, FSAD_Dataset_test
from utils.utils import time_file_str, time_string, convert_secs2time, AverageMeter, print_log
from models.siamese import Encoder, Predictor
from models.stn import stn_net
from losses.norm_loss import CosLoss
from utils.funcs import embedding_concat, mahalanobis_torch, rot_img, translation_img, hflip_img, rot90_img, grey_img
from utils.KCenterGreedy import KCenterGreedy
from utils.AnomalyMapGenerator import AnomalyMapGenerator
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter
from collections import OrderedDict
import warnings
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:3' if use_cuda else 'cpu')

def main():
    parser = argparse.ArgumentParser(description='Registration based Few-Shot Anomaly Detection')
    parser.add_argument('--obj', type=str, default='bottle')
    parser.add_argument('--data_type', type=str, default='mvtec')
    parser.add_argument('--data_path', type=str, default='./MVTec/')
    parser.add_argument('--epochs', type=int, default=50, help='maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate of others in SGD')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum of SGD')
    parser.add_argument('--seed', type=int, default=668, help='manual seed')
    parser.add_argument('--shot', type=int, default=2, help='shot count')
    parser.add_argument('--inferences', type=int, default=1, help='number of rounds per inference')
    parser.add_argument('--stn_mode', type=str, default='rotation_scale',
                        help='[affine, translation, rotation, scale, shear, rotation_scale, translation_scale, rotation_translation, rotation_translation_scale]')
    args = parser.parse_args()
    args.input_channel = 3

    if args.seed is None:
        args.seed = random.randint(1, 10000)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    args.prefix = time_file_str()
    args.save_dir = './logs_mpdd/'
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args.save_model_dir = './logs_mpdd/' + args.stn_mode + '/' + str(args.shot) + '/' + args.obj + '/'
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    log = open(os.path.join(args.save_dir, 'log_{}_{}.txt'.format(str(args.shot),args.obj)), 'w')
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)

    # load model and dataset
    STN = stn_net(args).to(device)
    ENC = Encoder().to(device)
    PRED = Predictor().to(device)

    # print(STN)

    STN_optimizer = optim.SGD(STN.parameters(), lr=args.lr, momentum=args.momentum)
    ENC_optimizer = optim.SGD(ENC.parameters(), lr=args.lr, momentum=args.momentum)
    PRED_optimizer = optim.SGD(PRED.parameters(), lr=args.lr, momentum=args.momentum)
    models = [STN, ENC, PRED]
    optimizers = [STN_optimizer, ENC_optimizer, PRED_optimizer]
    init_lrs = [args.lr, args.lr, args.lr]

    print('Loading Datasets')
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_dataset = FSAD_Dataset_train(args.data_path, class_name=args.obj, is_train=True, resize=args.img_size, shot=args.shot, batch=args.batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)
    test_dataset = FSAD_Dataset_test(args.data_path, class_name=args.obj, is_train=False, resize=args.img_size, shot=args.shot)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

    # start training
    save_name = os.path.join(args.save_model_dir, '{}_{}_{}_model.pt'.format(args.obj, args.shot, args.stn_mode))
    start_time = time.time()
    epoch_time = AverageMeter()
    img_roc_auc_old = 0.0
    per_pixel_rocauc_old = 0.0
    print('Loading Fixed Support Set')
    # fixed_fewshot_list = torch.load(f'./support_set/mpdd/{args.obj}/{args.shot}_{args.inferences}.pt')
    fixed_fewshot_list = torch.load(f'./mpdd_supp_set/2/b_w_2_1.pt')

    print_log((f'---------{args.stn_mode}--------'), log)

    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizers, init_lrs, epoch, args)
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        print_log(' {:3d}/{:3d} ----- [{:s}] {:s}'.format(epoch, args.epochs, time_string(), need_time), log)

        if epoch <= args.epochs:
            image_auc_list = []
            pixel_auc_list = []
            for inference_round in tqdm(range(args.inferences)):
                scores_list, test_imgs, gt_list, gt_mask_list = test(models, inference_round, fixed_fewshot_list,
                                                                     test_loader, **kwargs)
                scores = np.asarray(scores_list)
                # Normalization
                max_anomaly_score = scores.max()
                min_anomaly_score = scores.min()
                scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

                # calculate image-level ROC AUC score
                img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
                gt_list = np.asarray(gt_list)
                img_roc_auc = roc_auc_score(gt_list, img_scores)
                image_auc_list.append(img_roc_auc)

                # calculate per-pixel level ROCAUC
                gt_mask = np.asarray(gt_mask_list)
                gt_mask = (gt_mask > 0.5).astype(np.int_)
                per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
                pixel_auc_list.append(per_pixel_rocauc)

            image_auc_list = np.array(image_auc_list)
            pixel_auc_list = np.array(pixel_auc_list)
            mean_img_auc = np.mean(image_auc_list, axis = 0)
            mean_pixel_auc = np.mean(pixel_auc_list, axis = 0)

            if mean_img_auc + mean_pixel_auc > per_pixel_rocauc_old + img_roc_auc_old:
                state = {'STN': STN.state_dict(), 'ENC': ENC.state_dict(), 'PRED':PRED.state_dict()}
                torch.save(state, save_name)
                per_pixel_rocauc_old = mean_pixel_auc
                img_roc_auc_old = mean_img_auc
            print('Img-level AUC:',img_roc_auc_old)
            print('Pixel-level AUC:', per_pixel_rocauc_old)

            print_log(('Test Epoch(img, pixel): {} ({:.6f}, {:.6f}) best: ({:.3f}, {:.3f})'
            .format(epoch-1, mean_img_auc, mean_pixel_auc, img_roc_auc_old, per_pixel_rocauc_old)), log)

        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        train(models, epoch, train_loader, optimizers, log)
        train_dataset.shuffle_dataset()
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)
        
    log.close()

def train(models, epoch, train_loader, optimizers, log):
    STN = models[0]
    ENC = models[1]
    PRED = models[2]

    STN_optimizer = optimizers[0]
    ENC_optimizer = optimizers[1]
    PRED_optimizer = optimizers[2]

    STN.train()
    ENC.train()
    PRED.train()

    total_losses = AverageMeter()

    for (query_img, support_img_list, _) in tqdm(train_loader):
        STN_optimizer.zero_grad()
        ENC_optimizer.zero_grad()
        PRED_optimizer.zero_grad()

        query_img = query_img.squeeze(0).to(device)
        print("query_img", query_img.shape)
        query_feat = STN(query_img)
        """The shape of query_feat is [32, 256, 14, 14] [B,C,H,W]"""

        print("query_feat", query_feat.shape)

        support_img = support_img_list.squeeze(0).to(device)
        B,K,C,H,W = support_img.shape

        support_img = support_img.view(B * K, C, H, W)
        """The shape of support_feat is [64, 256, 14, 14] because we ARE DOING B*K images"""
        support_feat = STN(support_img)
        print("support_feat", support_feat.shape)

        """Because w ehave k-shot images"""
        support_feat = support_feat / K

        _, C, H, W = support_feat.shape
        support_feat = support_feat.view(B, K, C, H, W)
        
        support_feat = torch.sum(support_feat, dim=1)

        """The shape of z1 is [32, 256, 14, 14]"""
        z1 = ENC(query_feat)
        print("z1", z1.shape)
        """The shape of z2 is [32, 256, 14, 14]"""
        z2 = ENC(support_feat)
        print("z2", z2.shape)
        """The shape of p1 is [32, 256, 14, 14]"""
        p1 = PRED(z1)
        print("p1", p1.shape)
        """The shape of p2 is [32, 256, 14, 14]"""
        p2 = PRED(z2)
        print("p2", p2.shape)
        
        total_loss = CosLoss(p1,z2, Mean=True)/2 + CosLoss(p2,z1, Mean=True)/2
        total_losses.update(total_loss.item(), query_img.size(0))

        total_loss.backward()

        STN_optimizer.step()
        ENC_optimizer.step()
        PRED_optimizer.step()
        
    train_loss_dict[epoch] = total_losses.avg
    print_log(('Train Epoch: {} Total_Loss: {:.6f}'.format(epoch, total_losses.avg)), log)
    


def test(models, cur_epoch, fixed_fewshot_list, test_loader, **kwargs):
    STN = models[0]
    ENC = models[1]
    PRED = models[2]

    STN.eval()
    ENC.eval()
    PRED.eval()

    train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    support_imgs=[]
    count = 0
    if len(support_imgs) == 0:
        for (query_img, support_img, mask, y) in tqdm(test_loader):
            if count >= 10:  # Process only the first 10 items
                break;
            #Getting only 10 support sets otherwise we have 83 suport sets 1 set for each test image
            numpy_array = np.stack([t.numpy() for t in support_img])
            numpy_array = numpy_array.squeeze(1)
            support_imgs.append(numpy_array)
            print(count)
            count +=1
    
    new_size = [224, 224]

    #The shape support_img should be [2,3,224,224] [k, C, H, W]
    support_img = support_imgs[cur_epoch]
    # support_img = fixed_fewshot_list[cur_epoch]
    support_img = torch.from_numpy(support_img)
    # print("support_img", support_img.shape)
    
    height = support_img.shape[2]
    width = support_img.shape[3]

    #change heught and width to 224x224 if not already
    if height and width != 224:
        support_img = F.interpolate(support_img, size=new_size, mode='bilinear', align_corners=False)     #reshape image to 224x224

    
    # for img in support_img:
    #     print(img.shape)
    #     # Normalize the image tensor to [0, 1] if it isn't already (skip this step if not necessary)
    #     img = (img - img.min()) / (img.max() - img.min())

    #     # Transpose the tensor to (224, 224, 3) for visualization
    #     img = img.permute(1, 2, 0)

    #     # Convert the tensor to numpy array
    #     img_np = img.detach().numpy()

    #     # Display the image
    #     plt.imshow(img_np)
    #     plt.imsave('resized_image.png', img_np)

    augment_support_img = support_img
    # rotate img with small angle
    for angle in [-np.pi/4, -3 * np.pi/16, -np.pi/8, -np.pi/16, np.pi/16, np.pi/8, 3 * np.pi/16, np.pi/4]:
        rotate_img = rot_img(support_img, angle)
        augment_support_img = torch.cat([augment_support_img, rotate_img], dim=0)
    # translate img
    for a,b in [(0.2,0.2), (-0.2,0.2), (-0.2,-0.2), (0.2,-0.2), (0.1,0.1), (-0.1,0.1), (-0.1,-0.1), (0.1,-0.1)]:
        trans_img = translation_img(support_img, a, b)
        augment_support_img = torch.cat([augment_support_img, trans_img], dim=0)
    # hflip img
    flipped_img = hflip_img(support_img)
    augment_support_img = torch.cat([augment_support_img, flipped_img], dim=0)
    # rgb to grey img
    # greyed_img = grey_img(support_img)
    # augment_support_img = torch.cat([augment_support_img, greyed_img], dim=0)
    # rotate img in 90 degree
    for angle in [1,2,3]:
        rotate90_img = rot90_img(support_img, angle)
        augment_support_img = torch.cat([augment_support_img, rotate90_img], dim=0)
    augment_support_img = augment_support_img[torch.randperm(augment_support_img.size(0))]
    # print("augment_support_img",augment_support_img.shape)
    # torch version
    with torch.no_grad():
        support_feat = STN(augment_support_img.to(device))
    support_feat = torch.mean(support_feat, dim=0, keepdim=True)
    train_outputs['layer1'].append(STN.stn1_output)
    train_outputs['layer2'].append(STN.stn2_output)
    train_outputs['layer3'].append(STN.stn3_output)

    for k, v in train_outputs.items():
        train_outputs[k] = torch.cat(v, 0)

    # Embedding concat
    embedding_vectors = train_outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name], True)
    """The shape of embedding_vectors is [44, 448, 56, 56]""" 
    print("embedding_vectors",embedding_vectors.shape)

    # for e in embedding_vectors:
    #     print("e", e.shape)
    #Apply reshaping on supp embeddings 
    embedding_vectors = reshape_embedding(embedding_vectors)
    print("embedding_vectors reshaped",embedding_vectors.shape)

    global memory_bank

    #Applying core-set subsampling to get the embedding
    memory_bank = subsample_embedding(embedding_vectors, coreset_sampling_ratio= 0.01)
    print("memory_bank",memory_bank.shape)

    # calculate multivariate Gaussian distribution
    # B, C, H, W = embedding_vectors.size()
    # embedding_vectors = embedding_vectors.view(B, C, H * W)

    # mean = torch.mean(embedding_vectors, dim=0)
    # cov = torch.zeros(C, C, H * W).to(device)
    # I = torch.eye(C).to(device)
    # for i in range(H * W):
    #     cov[:, :, i] = torch.cov(embedding_vectors[:, :, i].T) + 0.01 * I
    # train_outputs = [mean, cov]

    # torch version
    query_imgs = []
    gt_list = []
    mask_list = []
    score_map_list = []

    for (query_img, support_img, mask, y) in tqdm(test_loader):
        #change height and width to 224x224 if not already
        height = query_img.shape[2]
        width = query_img.shape[3]
        
        if height and width != 224:
            print("changed dim")
            query_img = F.interpolate(query_img, size=new_size, mode='bilinear', align_corners=False)     #reshape image to 224x224
            mask = F.interpolate(mask, size=new_size, mode='bilinear', align_corners=False)     #reshape image to 224x224
            
        query_imgs.extend(query_img.cpu().detach().numpy())
        gt_list.extend(y.cpu().detach().numpy())
        mask_list.extend(mask.cpu().detach().numpy())
       

        
      

        # print("query_img",query_img.shape)
        
        # model prediction
        query_feat = STN(query_img.to(device))
        z1 = ENC(query_feat)
        z2 = ENC(support_feat)
        p1 = PRED(z1)
        p2 = PRED(z2)

        loss = CosLoss(p1,z2, Mean=False)/2 + CosLoss(p2,z1, Mean=False)/2
        loss_reshape = F.interpolate(loss.unsqueeze(1), size=query_img.size(2), mode='bilinear',align_corners=False).squeeze(0)
        score_map_list.append(loss_reshape.cpu().detach().numpy())

        test_outputs['layer1'].append(STN.stn1_output)
        test_outputs['layer2'].append(STN.stn2_output)
        test_outputs['layer3'].append(STN.stn3_output)

    for k, v in test_outputs.items():
        test_outputs[k] = torch.cat(v, 0)

    # Embedding concat
    embedding_vectors = test_outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name], True)
    """The shape of embedding_vectors is [83, 448, 56, 56]""" 
    
    batch_size, _, width, height = embedding_vectors.shape

    #apply reshaping on query embeddings 
    embedding_vectors = reshape_embedding(embedding_vectors)
    print(embedding_vectors.shape)

    #apply nearest neighbor search on query embeddings 
    patch_scores, locations = nearest_neighbors(embedding=embedding_vectors, n_neighbors=1)
    print("patch_scores", patch_scores.shape)
    print("locations", locations.shape)
    
    # reshape to batch dimension
    """The shape of patch_scores and locations is [83, 260288]"""
    patch_scores = patch_scores.reshape((batch_size, -1))
    locations = locations.reshape((batch_size, -1))
    print("A-patch_scores", patch_scores.shape)
    print("A-locations", locations.shape)
    
    #compute anomaly score
    anomaly_score = compute_anomaly_score(patch_scores, locations, embedding_vectors)
    print("anomaly_score", anomaly_score.shape)

    #reshape to w, h
    patch_scores = patch_scores.reshape((batch_size, 1, width, height))
    """The shape of patch_scores is [83, 1,  56, 56]"""
    
    #define the input size of the image
    input_size = [224,224]
    anomaly_map_generator = AnomalyMapGenerator(input_size=input_size)
    anomaly_map = anomaly_map_generator(patch_scores)
    """The shape of anomaly_map is [83, 1,  24, 24]"""

    # Select the first map
    # first_map = anomaly_map[0, 0, :, :]
    
    #Put it on CPU and convert to numpy
    score_map = anomaly_map.cpu().numpy()
        
    """To Generate the Heat Maps. Basically the score_map
    is the score of the patchesvwhere the anomalies are present."""   
    print(score_map.shape) 
    """The shape of score_map is (83,1, 224, 224)"""
    
    score_map = np.squeeze(score_map)
    """The shape of score_map is (83,224, 224)"""

    print(score_map.shape) 

    return score_map, query_imgs, gt_list, mask_list

    # STN = models[0]
    # ENC = models[1]
    # PRED = models[2]

    # STN.eval()
    # ENC.eval()
    # PRED.eval()

    # train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    # test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

    # support_img = fixed_fewshot_list[cur_epoch]
    # augment_support_img = support_img
    # # rotate img with small angle
    # for angle in [-np.pi / 4, -3 * np.pi / 16, -np.pi / 8, -np.pi / 16, np.pi / 16, np.pi / 8, 3 * np.pi / 16,
    #               np.pi / 4]:
    #     rotate_img = rot_img(support_img, angle)
    #     augment_support_img = torch.cat([augment_support_img, rotate_img], dim=0)
    # # translate img
    # for a, b in [(0.2, 0.2), (-0.2, 0.2), (-0.2, -0.2), (0.2, -0.2), (0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1),
    #              (0.1, -0.1)]:
    #     trans_img = translation_img(support_img, a, b)
    #     augment_support_img = torch.cat([augment_support_img, trans_img], dim=0)
    # # hflip img
    # flipped_img = hflip_img(support_img)
    # augment_support_img = torch.cat([augment_support_img, flipped_img], dim=0)
    # # rgb to grey img
    # greyed_img = grey_img(support_img)
    # augment_support_img = torch.cat([augment_support_img, greyed_img], dim=0)
    # # rotate img in 90 degree
    # for angle in [1, 2, 3]:
    #     rotate90_img = rot90_img(support_img, angle)
    #     augment_support_img = torch.cat([augment_support_img, rotate90_img], dim=0)
    # augment_support_img = augment_support_img[torch.randperm(augment_support_img.size(0))]
    # """The shape of augment_support_img which is the input tensor is [44, 3, 224, 224]""" 

    # # torch version
    # with torch.no_grad():
    #     support_feat = STN(augment_support_img.to(device))
    # """The shape of support_feat is [44, 256, 14, 14]""" 
    
    # support_feat = torch.mean(support_feat, dim=0, keepdim=True)

    # """The shape of layer1 is [44, 64, 56, 56] """
    # train_outputs['layer1'].append(STN.stn1_output)
    
    # """The shape of layer2 is [44, 128, 28, 28] """
    # train_outputs['layer2'].append(STN.stn2_output)
    
    # """The shape of layer3 is [44, 256, 14, 14] """
    # train_outputs['layer3'].append(STN.stn3_output)

    # for k, v in train_outputs.items():
    #     train_outputs[k] = torch.cat(v, 0)

    # # Embedding concat
    # embedding_vectors = train_outputs['layer1']
    # for layer_name in ['layer2', 'layer3']:
    #     embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name], True)
    # """The shape of embedding_vectors is [44, 448, 56, 56]""" 

    # # calculate multivariate Gaussian distribution
    # B, C, H, W = embedding_vectors.size()
    # embedding_vectors = embedding_vectors.view(B, C, H * W)
    # """The shape of embedding_vectors is [44, 448, 56, 56]""" 

    # mean = torch.mean(embedding_vectors, dim=0)
    # """The shape of mean is [448, 56x56 (3136)]""" 

    # cov = torch.zeros(C, C, H * W).to(device)
    # I = torch.eye(C).to(device)
    # for i in range(H * W):
    #     cov[:, :, i] = torch.cov(embedding_vectors[:, :, i].T) + 0.01 * I
    # """The shape of cov is [448, 448, 56x56 (3136)]""" 



    # train_outputs = [mean, cov]

    # # torch version
    # query_imgs = []
    # gt_list = []
    # mask_list = []
    # score_map_list = []

    # for (query_img, _, mask, y) in test_loader:
    #     query_imgs.extend(query_img.cpu().detach().numpy())
    #     gt_list.extend(y.cpu().detach().numpy())
    #     mask_list.extend(mask.cpu().detach().numpy())

    #     # model prediction
    #     query_feat = STN(query_img.to(device))
    #     print("query_feat",query_feat.shape)

    #     z1 = ENC(query_feat)
    #     z2 = ENC(support_feat)
    #     p1 = PRED(z1)
    #     p2 = PRED(z2)

    #     loss = CosLoss(p1, z2, Mean=False) / 2 + CosLoss(p2, z1, Mean=False) / 2
    #     loss_reshape = F.interpolate(loss.unsqueeze(1), size=query_img.size(2), mode='bilinear',
    #                                  align_corners=False).squeeze(0)
    #     score_map_list.append(loss_reshape.cpu().detach().numpy())

    #     test_outputs['layer1'].append(STN.stn1_output)
    #     test_outputs['layer2'].append(STN.stn2_output)
    #     test_outputs['layer3'].append(STN.stn3_output)

    # for k, v in test_outputs.items():
    #     test_outputs[k] = torch.cat(v, 0)

    # # Embedding concat
    # embedding_vectors = test_outputs['layer1']
    # for layer_name in ['layer2', 'layer3']:
    #     embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name], True)
    # """The shape of embedding_vectors is [83, 448, 56, 56]""" 

    # # calculate distance matrix
    # B, C, H, W = embedding_vectors.size()
    # embedding_vectors = embedding_vectors.view(B, C, H * W)
    # dist_list = []

    # for i in range(H * W):
    #     mean = train_outputs[0][:, i]
    #     conv_inv = torch.linalg.inv(train_outputs[1][:, :, i])
    #     dist = [mahalanobis_torch(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
    #     dist_list.append(dist)

    # dist_list = torch.tensor(dist_list).transpose(1, 0).reshape(B, H, W)

    # # upsample
    # score_map = F.interpolate(dist_list.unsqueeze(1), size=query_img.size(2), mode='bilinear',
    #                           align_corners=False).squeeze().numpy()

    # # apply gaussian smoothing on the score map
    # for i in range(score_map.shape[0]):
    #     score_map[i] = gaussian_filter(score_map[i], sigma=4)
    
    # print("score_map",score_map.shape)
    # exit()
    # return score_map, query_imgs, gt_list, mask_list

def adjust_learning_rate(optimizers, init_lrs, epoch, args):
    """Decay the learning rate based on schedule"""
    for i in range(3):
        cur_lr = init_lrs[i] * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
        for param_group in optimizers[i].param_groups:
            param_group['lr'] = cur_lr

def nearest_neighbors(embedding, n_neighbors):
    """Nearest Neighbours using brute force method and euclidean norm.

    Args:
        embedding (Tensor): Features to compare the distance with the memory bank.
        n_neighbors (int): Number of neighbors to look at

    Returns:
        Tensor: Patch scores.
        Tensor: Locations of the nearest neighbor(s).
    """
    global memory_bank
    
    # embedding_size = embedding.shape[0]
    # memory_bank_size = memory_bank.shape[0]

    # embedding_chunk_size = 100  # Adjust this value based on your GPU memory
    # memory_bank_chunk_size = 100  # Adjust this value based on your GPU memory

    # distances = []

    # for i in range(0, embedding_size, embedding_chunk_size):
    #     embedding_chunk = embedding[i:i + embedding_chunk_size, :]
    #     for j in range(0, memory_bank_size, memory_bank_chunk_size):
    #         memory_bank_chunk = memory_bank[j:j + memory_bank_chunk_size, :]
    #         distances_chunk = torch.cdist(embedding_chunk, memory_bank_chunk, p=2.0)
    #         distances.append(distances_chunk)
    #     print("round=",i)
        
    # print("Done")

    # # Concatenate all distance chunks
    # distances = torch.cat(distances, dim=0)
    # print("distances=",distances.shape)





    distances = torch.cdist(embedding, memory_bank, p=2.0)  # euclidean norm
    if n_neighbors == 1:
        # when n_neighbors is 1, speed up computation by using min instead of topk
        patch_scores, locations = distances.min(1)
    else:
        patch_scores, locations = distances.topk(k=n_neighbors, largest=False, dim=1)
    return patch_scores, locations

def reshape_embedding(embedding):
        """Reshape Embedding.

        Reshapes Embedding to the following format:
        [Batch, Embedding, Patch, Patch] to [Batch*Patch*Patch, Embedding]

        Args:
            embedding (Tensor): Embedding tensor extracted from CNN features.

        Returns:
            Tensor: Reshaped embedding tensor.
        """
        embedding_size = embedding.size(1)
        embedding = embedding.permute(0, 2, 3, 1).reshape(-1, embedding_size)
        return embedding
    
def subsample_embedding(embedding, coreset_sampling_ratio):
        """Subsample embedding based on coreset sampling and store to memory.

        Args:
            embedding (np.ndarray): Embedding tensor from the CNN
            sampling_ratio (float): Coreset sampling ratio
        """

        # Coreset Subsampling
        sampler = KCenterGreedy(embedding=embedding, sampling_ratio=coreset_sampling_ratio)
        coreset = sampler.sample_coreset()
        memory_bank = coreset
        return memory_bank
    
def compute_anomaly_score(patch_scores, locations, embedding):
        """Compute Image-Level Anomaly Score.

        Args:
            patch_scores (Tensor): Patch-level anomaly scores
            locations: Memory bank locations of the nearest neighbor for each patch location
            embedding: The feature embeddings that generated the patch scores
        Returns:
            Tensor: Image-level anomaly scores
        """
        global memory_bank

        #Set num_neighbors by your self
        num_neighbors = 9
        # Don't need to compute weights if num_neighbors is 1
        if num_neighbors == 1:
            return patch_scores.amax(1)
        batch_size, num_patches = patch_scores.shape
        # 1. Find the patch with the largest distance to it's nearest neighbor in each image
        max_patches = torch.argmax(patch_scores, dim=1)  # indices of m^test,* in the paper
        # m^test,* in the paper
        max_patches_features = embedding.reshape(batch_size, num_patches, -1)[torch.arange(batch_size), max_patches]
        # 2. Find the distance of the patch to it's nearest neighbor, and the location of the nn in the membank
        score = patch_scores[torch.arange(batch_size), max_patches]  # s^* in the paper
        nn_index = locations[torch.arange(batch_size), max_patches]  # indices of m^* in the paper
        # 3. Find the support samples of the nearest neighbor in the membank
        nn_sample = memory_bank[nn_index, :]  # m^* in the paper
        # indices of N_b(m^*) in the paper
        print("inside compute_anomaly_score")
        _, support_samples = nearest_neighbors(nn_sample, n_neighbors=num_neighbors)
        # 4. Find the distance of the patch features to each of the support samples
        distances = torch.cdist(max_patches_features.unsqueeze(1), memory_bank[support_samples], p=2.0)
        print("distances",distances.shape)

        # 5. Apply softmax to find the weights
        weights = (1 - F.softmax(distances.squeeze(1), 1))[..., 0]
        # 6. Apply the weight factor to the score
        score = weights * score  # s in the paper
        print("score",score.shape)
        return score

if __name__ == '__main__':
    main()
