## Team Objective

#### Move along PSNR - Per curve

#### 1. Undergrad's tips and tricks: 
  - 1. [ ] Mean shift per patch 
  - 2. [ ] RGB shuffle 
  
#### 2. DownMSE loss 

#### 3. Architecture adjustment 

#### 4. Laplace form

#### 5. Handle faces directly 


## Overall Progress 

#### 08/21 Tuesday: 
  ##### A. RGBshuffle 
  - 1. [x] Fix directory saving for model testing 
  - 2. [x] Implement rgb random shuffle, random in channels
  - 3. [x] Test & confirm the shuffling 
  - 4. [x] save the shuffle img patches for sanity check 
      - pre-shuffle & post-shuffle for both lr, hr 
  - 5. [x] Check the rgb training results, compare with the previous results
  - 6. [ ] understand the function __get_patch()
  - 7. [ ] Analysis and understand RGB results difference in paper
  
  ##### Mean shift per patch 
  
  ##### B. DownMSE loss 
  - 1. [ ] Save the LR images from SR, outside program for sanity check 
      - Decide to use BiCubic Downsampling 
  - 2. [ ] Integrate to program, bicubic down-sampling  
  - 3. [ ] Implement downMSE
  
  ##### C. others: 
  - 1. [ ] Display previous running results F_per vs. PSNR score 
  - 2. [ ] Reply Cynthia 
  - 3. [ ] Draft office desk email 
  
  ##### Readings & theoretical analysis
  - 1. [ ] Read about PSNR theory 
  - 2. [ ] Analysis GAN loss in different scalers 

####
  - save and load the approximator 
  - 

# PDSR Prior work
Submission to the PIRM2018 Super Resolution Contest 

The models are to big to fit on github, so they can be found here: https://drive.google.com/drive/folders/13eSV6d5cIFG67MMZ1by54uJcaExd6FHK?usp=sharing



