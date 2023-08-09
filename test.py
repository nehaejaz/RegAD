import os
import random
import argparse
import time
import torch
import numpy as np
from torch.optim import optimizer
import torch.nn.functional as F
from tqdm import tqdm
from datasets.mvtec import FSAD_Dataset_train, FSAD_Dataset_test
from utils.utils import time_file_str, time_string, convert_secs2time, AverageMeter, print_log
from models.siamese import Encoder, Predictor
from models.stn import stn_net
from losses.norm_loss import CosLoss
from utils.funcs import embedding_concat, mahalanobis_torch, rot_img, translation_img, hflip_img, rot90_img, grey_img
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter
from collections import OrderedDict
import warnings
import csv
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

def main():
    parser = argparse.ArgumentParser(description='RegAD on MVtec')
    parser.add_argument('--obj', type=str, default='hazelnut')
    parser.add_argument('--data_type', type=str, default='mvtec')
    parser.add_argument('--data_path', type=str, default='./MPDD/')
    parser.add_argument('--epochs', type=int, default=50, help='maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate in SGD')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum of SGD')
    parser.add_argument('--seed', type=int, default=668, help='manual seed')
    parser.add_argument('--shot', type=int, default=2, help='shot count')
    parser.add_argument('--inferences', type=int, default=10, help='number of rounds per inference')
    parser.add_argument('--stn_mode', type=str, default='rotation_scale', help='[affine, translation, rotation, scale, shear, rotation_scale, translation_scale, rotation_translation, rotation_translation_scale]')
    args = parser.parse_args()

    args.input_channel = 3
    if args.seed is None:
        args.seed = random.randint(1, 10000)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)
    args.prefix = time_file_str()

    STN = stn_net(args).to(device)
    ENC = Encoder().to(device)
    PRED = Predictor().to(device)

    # load models
    CKPT_name = f'./logs_mpdd/rotation_scale/{args.shot}/{args.obj}/{args.obj}_{args.shot}_rotation_scale_model.pt'
    # CKPT_name = f'./save_checkpoints/{args.shot}/{args.obj}/{args.obj}_{args.shot}_rotation_scale_model.pt'
    model_CKPT = torch.load(CKPT_name)
    STN.load_state_dict(model_CKPT['STN'])
    ENC.load_state_dict(model_CKPT['ENC'])
    PRED.load_state_dict(model_CKPT['PRED'])
    models = [STN, ENC, PRED]

    print('Loading Datasets')
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    test_dataset = FSAD_Dataset_test(args.data_path, class_name=args.obj, is_train=False, resize=args.img_size, shot=args.shot)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

    print('Loading Fixed Support Set')
    # fixed_fewshot_list = torch.load(f'./support_set/{args.obj}/{args.shot}_{args.inferences}.pt')

    print('Start Testing:')
    start_time = time.time()

    image_auc_list = []
    pixel_auc_list = []
    for inference_round in range(args.inferences):
        print('Round {}:'.format(inference_round))
        scores_list, test_imgs, gt_list, gt_mask_list = test(args, models, inference_round, test_loader, **kwargs)
        scores = np.asarray(scores_list)

        # Normalization
        max_anomaly_score = scores.max()
        min_anomaly_score = scores.min()
        scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)
        print(scores.shape)
        # Let's display the first 5 maps
        index = 0
        for img in test_imgs:
            orig_image = img.transpose(1, 2, 0)

            # Plot the original image
            plt.imshow(orig_image)  # Assuming the original image is grayscale
            
            # Overlay the heatmap. Use the 'alpha' parameter for transparency.
            plt.imshow(scores[index], cmap='hot', alpha=0.5)  
            plt.title(f"Overlayed Anomaly Map")
            plt.axis('off')  # Hide axis
            plt.colorbar()
            plt.savefig(f"heatmaps/{str(inference_round)}/{index}.png") 
            plt.close() 
            index +=1  


        # calculate image-level ROC AUC score
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gt_list = np.asarray(gt_list)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        image_auc_list.append(img_roc_auc)
        print("img_roc_auc", img_roc_auc)
        print("image_auc_list",image_auc_list)

        # calculate per-pixel level ROCAUC
        gt_mask = np.asarray(gt_mask_list)
        gt_mask = (gt_mask > 0.5).astype(np.int_)
        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        pixel_auc_list.append(per_pixel_rocauc)
        print("per_pixel_rocauc", per_pixel_rocauc)
        print("pixel_auc_list", pixel_auc_list)
        
        """save test images for classification """
        index = 0

        for img in test_imgs:
            # Transpose from [channels, height, width] to [height, width, channels]
            img = np.transpose(img, (1, 2, 0))

            # The number you want to write
            ground_truth_lab = gt_list[index]
            score_num = img_scores[index]

            fig, ax = plt.subplots()

            # Visualize the image
            ax.imshow(img)

            # Add text at the bottom left (x=0, y=image height)
            ax.text(0, img.shape[0], str(ground_truth_lab), color='white', fontsize=16, weight='bold', verticalalignment='bottom')
            
            # Add text at the bottom right (x=image width, y=image height)
            ax.text(img.shape[1], img.shape[0], str(score_num), color='red', fontsize=16, weight='bold', verticalalignment='bottom', horizontalalignment='right')

            # Visualize the image
            plt.imshow(img)

            # Save the image
            print("results/"+str(inference_round)+'/'+str(index)+'.png')
            fig.savefig("results/"+str(inference_round)+'/'+str(index)+'.png') 
            plt.close(fig)
            index +=1  


    end_time = time.time()
    inference_time = end_time - start_time
    image_auc_list = np.array(image_auc_list)
    pixel_auc_list = np.array(pixel_auc_list)
    mean_img_auc = np.mean(image_auc_list, axis = 0)
    mean_pixel_auc = np.mean(pixel_auc_list, axis = 0)
    print('Img-level AUC:',mean_img_auc)
    print('Pixel-level AUC:', mean_pixel_auc)
    print(f"Inference time: {inference_time} seconds")

def test(args, models, cur_epoch, test_loader, **kwargs):
    STN = models[0]
    ENC = models[1]
    PRED = models[2]

    STN.eval()
    ENC.eval()
    PRED.eval()

    train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    
    support_imgs=[]
    for (query_img, support_img, mask, y) in tqdm(test_loader):
        #Getting only 10 support sets otherwise we have 83 suport sets 1 set for each test image
        for i in range(10):
            numpy_array = np.stack([t.numpy() for t in support_img])
            numpy_array = numpy_array.squeeze(1)
            support_imgs.append(numpy_array)
        break;
    
    new_size = [224, 224]

    #The shape support_img should be [2,3,224,224] [k, C, H, W]
    support_img = support_imgs[cur_epoch]
    # support_img = fixed_fewshot_list[cur_epoch]
    support_img = torch.from_numpy(support_img)
    print("support_img", support_img.shape)
    
    single_img = support_img[0]
    
    print(single_img.shape)
    # Transpose the tensor to (224, 224, 3) for visualization
    img = single_img.permute(1, 2, 0)

    # Convert the tensor to numpy array
    img_np = img.detach().numpy()

    # Display the image
    # plt.imshow(img_np)
    # plt.imsave('new.png', img_np)
    
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

    # support_img = fixed_fewshot_list[cur_epoch]
    
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
    greyed_img = grey_img(support_img)
    augment_support_img = torch.cat([augment_support_img, greyed_img], dim=0)
    # rotate img in 90 degree
    for angle in [1,2,3]:
        rotate90_img = rot90_img(support_img, angle)
        augment_support_img = torch.cat([augment_support_img, rotate90_img], dim=0)
    augment_support_img = augment_support_img[torch.randperm(augment_support_img.size(0))]
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

    # calculate multivariate Gaussian distribution
    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W)
    support_set_embeddings=embedding_vectors

    mean = torch.mean(embedding_vectors, dim=0)
    cov = torch.zeros(C, C, H * W).to(device)
    I = torch.eye(C).to(device)
    for i in range(H * W):
        cov[:, :, i] = torch.cov(embedding_vectors[:, :, i].T) + 0.01 * I
    train_outputs = [mean, cov]

    # torch version
    query_imgs = []
    gt_list = []
    mask_list = []
    score_map_list = []

    for (query_img, support_img, mask, y) in tqdm(test_loader):
        query_imgs.extend(query_img.cpu().detach().numpy())
        gt_list.extend(y.cpu().detach().numpy())
        mask_list.extend(mask.cpu().detach().numpy())
        
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

    # calculate distance matrix
    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W)
    dist_list = []
    
    print("support_set_embeddings", support_set_embeddings.shape)
    print("support_set_embeddings", embedding_vectors.shape)

    for i in range(H * W):
        mean = train_outputs[0][:, i]
        conv_inv = torch.linalg.inv(train_outputs[1][:, :, i])
        dist = [mahalanobis_torch(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
        dist_list.append(dist)
        
    # list_tensor = []
    # for dis in dist_list:
    #     # print(type(dis))
    #     for d in dis:
    #         d = d.cpu()
    #         list_tensor.append(d.numpy())

    # array_list = stacked_tensor.numpy()
    # list_tensor = [array.reshape(1, -1).tolist() for array in list_tensor]

    # # # Define the CSV file path
    # csv_file = 'distances.csv'

    # # # Write the data to the CSV file
    # with open(csv_file, 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(list_tensor)

    # print("CSV file created successfully.")

    """shape of dist_list is (83,56,56)"""
    dist_list = torch.tensor(dist_list).transpose(1, 0).reshape(B, H, W)
    print("dist_list",dist_list.shape)

    # upsample
    """shape of score_map is (83,224,224)"""
    score_map = F.interpolate(dist_list.unsqueeze(1), size=query_img.size(2), mode='bilinear',
                              align_corners=False).squeeze().numpy()
  
    print("score_map",score_map.shape)

    # apply gaussian smoothing on the score map
    for i in range(score_map.shape[0]):
        score_map[i] = gaussian_filter(score_map[i], sigma=4)
        
    """To Generate the Heat Maps. Basically the score_map
    is the score of the patchesvwhere the anomalies are present."""    
    # for i in range(score_map.shape[0]):
    #     image = score_map[i, :, :]
    #     print("image", image)
    #     # Convert tensor to PIL Image
    #     transform = transforms.ToPILImage()
    #     image = transform(image)
    #     # Convert image to grayscale
    #     image_gray = image.convert('L')
    #     # Save the image
    #     image_gray.save("rounds/"+str(cur_epoch)+"/" + str(i)+".jpg")
        
    return score_map, query_imgs, gt_list, mask_list

if __name__ == '__main__':
    main()
