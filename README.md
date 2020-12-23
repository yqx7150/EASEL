# EASEL: Iterative Reconstruction for Low-Dose CT using Deep Gradient Priors of Generative Model
The Code is created based on the method described in the following paper:        
Iterative Reconstruction for Low-Dose CT using Deep Gradient Priors of Generative Model.      
Zhuonan He, Yikun Zhang, Yu Guan, Shanzhou Niu, Yi Zhang,  Yang Chen, Qiegen Liu.


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

  * Homotopic Gradients of Generative Density Priors for MR Image Reconstruction  
[<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2008.06284)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/HGGDP)

  * REDAEP: Robust and Enhanced Denoising Autoencoding Prior for Sparse-View CT Reconstruction  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/9076295)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/REDAEP)

* Progressive Colorization via Interative Generative Models  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/9258392)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/iGM)
 
