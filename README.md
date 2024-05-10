# EASEL 
Iterative Reconstruction for Low-Dose CT using Deep Gradient Priors of Generative Model       
IEEE Transactions on Radiation and Plasma Medical Sciences, Feb. 2022.    
https://ieeexplore.ieee.org/abstract/document/9703672         
      
The Code is created based on the method described in the following paper:        
Iterative Reconstruction for Low-Dose CT using Deep Gradient Priors of Generative Model.      
Zhuonan He, Yikun Zhang, Yu Guan, Bing Guan, Shanzhou Niu, Yi Zhang, Yang Chen, Qiegen Liu.


## Abstract
Dose reduction in computed tomography (CT) is essential for decreasing radiation risk in clinical applications. Iterative reconstruction is one of the most promising ways to compensate for the increased noise due to reduction of photon flux. Rather than most existing prior-driven algorithms that benefit from manually designed prior functions or supervised learning schemes, in this work we integrate the data-consistency as a conditional term into the iterative generative model for low-dose CT. At first, a score-based generative network is used for unsupervised distribution learning and the gradient of generative density prior is learned from normal-dose images. Then, the annealing Langevin dynamics is employed to update the trained priors with conditional scheme, i.e., the distance between the reconstructed image and the manifold is minimized along with data fidelity during reconstruction. Experimental comparisons demonstrated the noise reduction and detail preservation abilities of the proposed method. 

## The flowchart of EASEL

![repeat-EASEL](https://github.com/yqx7150/EASEL/blob/master/EASEL/Figs/fig.png)
Fig. 1. The training and reconstruction paradigm of the generative model-based algorithm EASEL. It consists of two components, i.e., a denoising score matching for score estimation involving various noise magnitudes simultaneously, and an iterative cycle for reconstruction including the annealed and conditional Langevin dynamics.

## Requirements and Dependencies
    python==3.5
    Pytorch==1.4.0
    ODL==1.1.0
    astra-toolbox==1.9.9dev
    CUDA==9.0
    
## Test
    python3.5 separate_ImageNet.py --model ncsn --runner Aapm_Runner_CTtest_10_noconv --config aapm_10C.yml --doc AapmCT_10C --test --image_folder output

## Checkpoints
The pretrained checkpoints can be download pretrained models from [Baidu Drive](https://pan.baidu.com/s/1hV-_RsZi0ii7Uh_ADBEj1Q ). 
key number is "xt4l" 

## Visual Comparisons
![repeat-EASEL](https://github.com/yqx7150/EASEL/blob/master/EASEL/Figs/ret.png)
![repeat-EASEL](https://github.com/yqx7150/EASEL/blob/master/EASEL/Figs/zoom_ret.png)
Fig. 2. Reconstruction results of AAPM challenge data for different methods. From left to right: reference image, FBP, TV, K-SVD, RED-CNN, DP-ResNet,  EASEL.



### Other Related Projects

  * One Sample Diffusion Model in Projection Domain for Low-Dose CT Imaging  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/10506793)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/OSDM)

  * REDAEP: Robust and Enhanced Denoising Autoencoding Prior for Sparse-View CT Reconstruction  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/9076295)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/REDAEP)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)

  * Wavelet-improved score-based generative model for medical imaging  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/10288274)

  * 基于深度能量模型的低剂量CT重建  
[<font size=5>**[Paper]**</font>](http://cttacn.org.cn/cn/article/doi/10.15953/j.ctta.2021.077)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EBM-LDCT)  

  * Universal Generative Modeling for Calibration-free Parallel MR Imaging  
[<font size=5>**[Paper]**</font>](https://biomedicalimaging.org/2022/)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/UGM-PI)   [<font size=5>**[Poster]**</font>](https://github.com/yqx7150/UGM-PI/blob/main/paper%20%23160-Poster.pdf)     
     
* Progressive Colorization via Interative Generative Models  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/9258392)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/iGM)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)
 
* Joint Intensity-Gradient Guided Generative Modeling for Colorization
[<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2012.14130)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/JGM)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)

 * Wavelet Transform-assisted Adaptive Generative Modeling for Colorization
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/9782538)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/WACM)   [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

 * Diffusion Models for Medical Imaging
[<font size=5>**[Paper]**</font>](https://github.com/yqx7150/Diffusion-Models-for-Medical-Imaging)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/Diffusion-Models-for-Medical-Imaging)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HKGM/tree/main/PPT) 
