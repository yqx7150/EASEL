import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import math
from NCSN_train.models.cond_refinenet_dilated_noconv import CondRefineNetDilated
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from scipy.io import loadmat,savemat
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr,compare_ssim
import glob
import h5py
import time
from skimage import img_as_float, img_as_ubyte, io
from scipy.misc import imread,imsave
from scipy.linalg import norm,orth
from scipy.stats import poisson
import dicom
from skimage.transform import radon, iradon
import odl
plt.ion()
savepath = './result/'
__all__ = ['Aapm_Runner_CTtest_10_noconv']
   
class Aapm_Runner_CTtest_10_noconv():
    def __init__(self, args, config):
        self.args = args
        self.config = config
    def write_images(self,x,image_save_path):
        x = np.array(x,dtype=np.uint8)
        cv2.imwrite(image_save_path, x)
    def test(self):
        N = 512
        ANG = 180
        VIEW = 360
        cols = rows =512
        THETA = np.linspace(0, ANG, VIEW + 1)
        THETA = THETA[:-1]
        angle_partition = odl.uniform_partition(0, 2 * np.pi, 1000)
        detector_partition = odl.uniform_partition(-360, 360, 1000)
        geometry = odl.tomo.FanBeamGeometry(angle_partition, detector_partition,
                                    src_radius=500, det_radius=500)
        reco_space = odl.uniform_discr(min_pt=[-128, -128], max_pt=[128, 128], shape=[512, 512], dtype='float32')
        ray_trafo = odl.tomo.RayTransform(reco_space, geometry)
        pseudoinverse = odl.tomo.fbp_op(ray_trafo)
        ## data load
        dataset = dicom.read_file('./L067_FD_1_1.CT.0001.0001.2015.12.22.18.09.40.840353.358074219.IMA')
        img1 = dataset.pixel_array.astype(np.float32)
        img = img1
        RescaleSlope = dataset.RescaleSlope
        RescaleIntercept = dataset.RescaleIntercept
        CT_img = img * RescaleSlope + RescaleIntercept
        # Load the score network
        states = torch.load(os.path.join(self.args.log, 'checkpoint_100000.pth'), map_location=self.config.device)
        scorenet = CondRefineNetDilated(self.config).to(self.config.device)
        scorenet = torch.nn.DataParallel(scorenet, device_ids=[0])
        scorenet.load_state_dict(states[0])
        scorenet.eval()
        ## degrade process
        pre_img = (CT_img+1000)/1000*0.02
        ATA = ray_trafo.adjoint(ray_trafo(ray_trafo.domain.one()))
        ## LOW-DOSE SINOGRAM GENERATION
        photons_per_pixel =  5e4
        mu_water = 0.02
        phantom = reco_space.element(img)
        phantom = phantom/1000.0
        proj_data = ray_trafo(phantom)
        proj_data = np.exp(-proj_data * mu_water)
        proj_data = odl.phantom.poisson_noise(proj_data * photons_per_pixel)
        proj_data = np.maximum(proj_data, 1) / photons_per_pixel
        proj_data = np.log(proj_data) * (-1 / mu_water)
        image_input = pseudoinverse(proj_data)
        image_input = image_input
        x = np.copy(image_input)
        z = np.copy(x)
        maxdegrade = np.max(image_input)
        image_gt = (CT_img-np.min(CT_img))/(np.max(CT_img)-np.min(CT_img))
        image_input = image_input.asarray()
        image_input_show = image_input.copy()
        psnr_ori = compare_psnr(255*abs(image_input/maxdegrade),255*abs(image_gt),data_range=255)
        ssim_ori = compare_ssim(abs(image_input/maxdegrade),abs(image_gt),data_range=1)
        image_input = image_input/maxdegrade
        image_gt = (CT_img-np.min(CT_img))/(np.max(CT_img)-np.min(CT_img))
        image_shape = list((1,)+(10,)+image_input.shape[0:2])
        x0 = nn.Parameter(torch.Tensor(np.zeros(image_shape)).uniform_(-1,1)).cuda()
        x01 = x0
        step_lr=0.6*0.00003
        sigmas = np.exp(np.linspace(np.log(1), np.log(0.01),12))
        n_steps_each = 150
        max_psnr = 0
        max_ssim = 0
        min_hfen = 100
        start_start = time.time()
        for idx, sigma in enumerate(sigmas):
            start_out = time.time()
            print(idx)
            lambda_recon = 1./sigma**2
            labels = torch.ones(1, device=x0.device) * idx
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            print('sigma = {}'.format(sigma))
            for step in range(n_steps_each):
                start_in = time.time()
                noise1 = torch.rand_like(x0)* np.sqrt(step_size * 2)
                grad1 = scorenet(x01, labels).detach()
                x0 = x0 + step_size * grad1
                x01 = x0 + noise1

                x0=np.array(x0.cpu().detach(),dtype = np.float32)
                x1 = np.squeeze(x0)
                x1 = np.mean(x1,axis=0)
                psnr1 = compare_psnr(255*abs(x1),255*abs(image_gt),data_range=255)
                ssim1 = compare_ssim(abs(x1),abs(image_gt),data_range=1)
                ## ********** SQS ********* ##
                hyper = 150
                sum_diff =  x - x1*maxdegrade
                norm_diff = ray_trafo.adjoint((ray_trafo(x) - proj_data))
                x_new = z - (norm_diff + 2*hyper*sum_diff)/(ATA + 2*hyper)
                z = x_new + 0.5 * (x_new - x)
                x = x_new
                x_rec = x.asarray()
                x_rec = x_rec/maxdegrade
                psnr2 = compare_psnr(255*abs(x_rec),255*abs(image_gt),data_range=255)
                ssim2 = compare_ssim(abs(x_rec),abs(image_gt),data_range=1)
                end_in = time.time()
                print("inner loop:%.2fs"%(end_in-start_in))
                psnr = compare_psnr(255*abs(x_rec/np.max(x_rec)),255*abs(image_gt),data_range=255)
                ssim = compare_ssim(abs(x_rec/np.max(x_rec)),abs(image_gt),data_range=1)
                if ssim > max_ssim:
                    max_ssim = ssim
                    max_psnr = psnr
                print("current {} step".format(step),'PSNR :', psnr,'SSIM :', ssim)
                x_mid = np.zeros([1,10,512,512],dtype=np.float32)
                x_rec = np.clip(x_rec,0,1)
                x_rec = np.expand_dims(x_rec,2)
                x_mid_1 = np.tile(x_rec,[1,1,10])
                x_mid_1 = np.transpose(x_mid_1,[2,0,1])
                x_mid[0,:,:,:] = x_mid_1
                x0 = torch.tensor(x_mid,dtype=torch.float32).cuda()
            end_out = time.time()
            print("outer iter:%.2fs"%(end_out-start_out))
        plt.ioff()
        end_end = time.time()
        print("PSNR:%.2f"%(max_psnr),"SSIM:%.2f"%(max_ssim))
        print("total time:%.2fs"%(end_end-start_start))

        
