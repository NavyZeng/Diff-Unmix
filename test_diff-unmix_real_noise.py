# the test code of diff-unmix

from Dim_autoencoder import LR_decompose
# from utils import *
import scipy.io as scio
import torch
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from guided_diffusion import utils
from guided_diffusion.create import create_model_and_diffusion_RS
import json
from collections import OrderedDict
from unet import UNet
import torch.nn.functional as F
import cv2
import scipy.io as sio

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0' # gpu id
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')


def LoadTest_256by256(path_test):
    scene_list = os.listdir(path_test)
    scene_list.sort()
    test_data = np.zeros((len(scene_list), 256, 256, 28))
    for i in range(len(scene_list)):
        scene_path = path_test + scene_list[i]
        img = sio.loadmat(scene_path)['img']
        test_data[i, :, :, :] = img[:256, :256, :28] 
    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
    return test_data

# @torch.no_grad() 
def diffusion_3HSI(A_y, A_x, A_c, E_y, y):
    opt = {
    'baseconfig': 'base.json',
    'gpu_ids': "0",
    'dataroot': '',
    'batch_size': 1,
    'savedir': './results',
    'eta1': 1,
    'eta2': 2,
    'seed': 0,
    'dataname': '',
    'step': 100,
    'scale': 4,
    'kernelsize': 9,
    'sig': None,
    'samplenum': 1,
    # 'diffusion': 1000,
    # 'diffusion_steps': 1000,
    'resume_state': 'I190000_E97_opt'}
    # Assuming 'base.json' contains the JSON-formatted data
    with open('./guided_diffusion/base.json', 'r') as json_file:
        json_str = json_file.read()
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)
    opt = utils.dict_to_nonedict(opt)
    # opt['diffusion']['diffusion_steps'] = opt['step']

    # Assign the value of 'step' to 'diffusion_steps'
    if opt.get('step'):
        opt['diffusion'] = {'diffusion_steps': opt['step']}

    device = torch.device("cuda")

    ## create model and diffusion process
    model, diffusion = create_model_and_diffusion_RS(opt)

    ## load model
    fix_diff = 1
    if fix_diff:
        gen_path = './guided_diffusion/I190000_E97_gen.pth'
        cks = torch.load(gen_path)
        new_cks = OrderedDict()
        for k, v in cks.items():
            newkey = k[11:] if k.startswith('denoise_fn.') else k
            new_cks[newkey] = v
        model.load_state_dict(new_cks, strict=False)
    model.to(device)
    # model.train()
    model.eval()

    ## params
    param = {'eta1': opt['eta1']}
    # print(A_y.shape)
    # exit()
    Ch, ms = A_y.shape[0], A_y.shape[-1]

    model_condition = {'A_x': A_x.to(device), 'A_c': A_c.to(device), 'A_y': A_y.to(device), 'E_y': E_y.to(device), 'y': y.to(device)}
    Rr = 3  # spectral dimensironality of subspace

    sample = diffusion.p_sample_loop(model, (1, Ch, ms, ms),
    Rr = Rr,
    clip_denoised=True,
    model_condition=model_condition,
    param=param,
    save_root=None,
    progress=True,)

    sample = (sample + 1)/2 #  must
    # sample = (sample *2 ) +0.5

    return sample


def Re_unmix(A, E, shape):
    # Use SVD to implement Hyperspectral Unmixing X = E * A
   
    b, c, h, w = shape
    R = 3
    X = E.reshape(b, c, R) @ A.reshape(b, R, h*w)
    return X.view(b, c, h, w)

def mix(A_hat, E_y, shape):
    bs, c, h, w = shape
    X_hat = torch.zeros(bs, c, h, w)
    for i in range(bs):
        A_hat_f_m = torch.reshape(A_hat[i,:,:,:], [3, 256*256])
        X_hat_m = torch.mm(torch.reshape(E_y[i,:,:], [28, 3]), A_hat_f_m)
        X_hat[i,:,:,:]   = torch.reshape(X_hat_m, [28, 256, 256])
    return X_hat


def Unmix_svd_3d(y):
    # Use SVD to implement Hyperspectral Unmixing X = E * A
    Rr = 3
    
    # Reshape input tensor to be of shape (b, c, h*w)
    b, c, h, w = y.shape
    y = y.reshape(b, c, -1)
    
    # Perform SVD
    U, S, V = torch.svd(y)
    E = U[:, :, :Rr].permute(0, 2, 1)
    A = E @ y
    # print(E.shape)
    
    # Reshape A back to original shape
    A = A.view(b, Rr, h, w)
    
    return A, E


def Unmix(y, x): 
    # Load pretrained model to implement Hyperspectral Unmixing X = E *A
    Decompose_model = LR_decompose().cuda() # spectral unmixing
    pretrained_model_path = './exp/unmixing/model_epoch_17.pth'
    checkpoint = torch.load(pretrained_model_path)
    Decompose_model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()}, strict=True)
    Decompose_model.eval()
    with torch.no_grad():
        X_y, X_x, A_y, A_x, E_y, E_x = Decompose_model(y, x)

    if E_x.shape[1] == 28:
        E_x = torch.unsqueeze(E_x, 1)
    if E_y.shape[1] == 28:
        E_y = torch.unsqueeze(E_y, 1)  

    return X_y, X_x, A_y, A_x, E_y, E_x 

def test(Condi_net):
    # test_path = "datasets/kaist_simu_data/"
    test_path = "datasets/real_data/"
    test_data = LoadTest_256by256(test_path) # load real noisy data
    test_gt = test_data.cuda().float()
    input_meas = test_gt 

    STU = 1
    if STU:
        X_y, X_x, A_y, A_x, E_y, E_x = Unmix(input_meas, test_gt) # A_x, E_x from test_gt only for visual reference or computing PSNR
    else:
        A_y, E_y = Unmix_svd_3d(input_meas)
        A_x, E_x = Unmix_svd_3d(test_gt)
    with torch.no_grad():
        A_c = Condi_net(A_y)

    # start_time = time.time()
    y = input_meas
    z = torch.empty_like(A_y)
    diffIt_A = A_y.shape[0]
    for j in range(diffIt_A):
        z[j, :, :, :] = diffusion_3HSI(A_y[j, :, :, :], A_x[j, :, :, :], A_c[j, :, :, :], E_y[j, :, :], y[j, :, :, :])
    A_hat = z #* A_init.max()

    # e_time = time.time() - start_time
    # print(f'Time {e_time}.')
    shape = test_gt.shape
    X_hat = mix(A_hat, E_y.detach(), shape)

    out_X = 1
    if out_X:
        pred = X_hat #/X_hat.max()
        truth = test_gt
    else:
        pred = A_hat
        truth = A_x

    return pred, truth, input_meas, A_y.detach()

def main():
    # model
    print('Testing model: Self-supervised HSI denoising via Diff-Unmix')
    unet = UNet(in_channels=3, out_channels=3).cuda()
    pretrained_model_path = './exp/condition_function/model/model_epoch_61.pth'

    checkpoint = torch.load(pretrained_model_path)
    unet.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()}, strict=True)
    unet.eval()
    pred, truth, input_meas, A_y = test(unet)
   
    # Show bands   
    show_image = 1
    if show_image:
        hh = 256
        ww = 256
        b1 = 26 # 26
        b2 = 27
        pred = np.transpose(pred.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
        input_meas = np.transpose(input_meas.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
        truth = np.transpose(truth.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
        
        channel_Y = pred[0, :hh, :ww, b1]
        channel_Z = input_meas[0, :hh, :ww, b1]

        channel_Y2 = pred[0, :hh, :ww, b2]
        channel_Z2 = input_meas[0, :hh, :ww, b2]
        # Create a figure with two subplots

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 5))


        # Show Y in the second subplot
        ax2.imshow(channel_Y, cmap='gray')
        ax2.set_title('Diff-Unmix')
        ax2.axis('off')

        ax1.imshow(channel_Z, cmap='gray')
        ax1.set_title('Real noisy image')
        ax1.axis('off')

        ax4.imshow(channel_Y2, cmap='gray')
        ax4.set_title('Diff-Unmix')
        ax4.axis('off')

        ax3.imshow(channel_Z2, cmap='gray')
        ax3.set_title('Real noisy image')
        ax3.axis('off')

        plt.tight_layout()
        plt.show()

    # name = opt.outf + 'Test_result_g02p015.mat'

    # save .mat
    save_mat = 0
    if save_mat:
        name = f'{opt.outf}Test_result_real_check.mat'
        print(f'Save reconstructed HSIs as {name}.')
        scio.savemat(name, {'truth': truth, 'pred': pred, 'noisy': input_meas})
    
    # save jpg
    save_img = 0
    if save_img:
        OUTPUT_folder_rgb = './exp/images/'
        save_path_our = OUTPUT_folder_rgb + 'diff-unmix_s10_g03.png'
        output = np.squeeze((pred-pred.min())/(pred.max()-pred.min()))[:, :, 14]
        cv2.imwrite(save_path_our, cv2.cvtColor(255 * output, cv2.COLOR_RGB2BGR))

        save_path_our = OUTPUT_folder_rgb + 'gt_s10.png'
        output_gt = np.squeeze((truth-truth.min())/(truth.max()-truth.min()))[:, :, 20]
        cv2.imwrite(save_path_our, cv2.cvtColor(255 * output_gt, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    main()
