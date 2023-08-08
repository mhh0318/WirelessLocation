from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
import warnings
warnings.filterwarnings("ignore")
import scipy.io
import json

def rss_noise_add(image):
    dB_image = -127+(image/255.) * 80.
    noise = np.random.normal(0, 10, image.shape)
    dB_image += noise
    image = ((dB_image+127.)/ 80.)* 255.
    return image

class locDL(Dataset):
    def __init__(self,maps_inds=np.zeros(1), phase="train",
                 ind1=0,ind2=0, 
                 dir_dataset="dataset/",
                 numTx=5,   
                 numTrials=50,
                 numRx=200,
                 simulation="DPM",
                 cityMap=False, 
                 carsMap=False, 
                 TxMaps=False,
                 transform= transforms.ToTensor(),
                 return_dict=False,
                 rss_noise_flag=False,
                 ):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom". 
            dir_dataset: directory of the dataset.
            numTx: Number of transmitters per map. Default and maximum numTx = 5.  
            numTrials: Number of sets of numTx transmitters per map
            simulation:"DPM", "IRT2", "DPMtoIRT2", "DPMcars". Default= "DPM"
            cityMap: . Default cityMap="false"
            TxMaps: Images of Tx. Defaul TxMaps="false"
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())
                 
        Output:
            inputs: The LocUNet inputs.  
            RXlocr: Pixel row of true location
            RXlocc: Pixel column of true location
            
        """
        self.return_dic = return_dict

        
        if phase=="train":
            self.ind1=0
            self.ind2=68
            self.maps_inds=np.arange(0,99,1,dtype=np.int16)
            self.matfile = '/home/hu/uwlc/dataset/ToATest/mat/TxRxRandTrainnoTx5.npy'
            self.dir_dataset = '/home/hu/uwlc/dataset/RadioLocSeer'
            self.dir_gainTrue= '/home/hu/uwlc/dataset/RadioLocSeer/gain/DPM/true'
            self.dir_toa_true = '/home/hu/uwlc/dataset/RadioToAImage/True'
            self.dir_toa_est = '/home/hu/uwlc/dataset/RadioToAImage/Est'
        elif phase=="val":
            self.ind1=69
            self.ind2=83
            self.maps_inds=np.arange(0,99,1,dtype=np.int16)
            self.matfile = '/home/hu/uwlc/dataset/ToATest/mat/TxRxRandTrainnoTx5.npy'
            self.dir_dataset = '/home/hu/uwlc/dataset/RadioLocSeer'
            self.dir_gainTrue= '/home/hu/uwlc/dataset/RadioLocSeer/gain/DPM/true'
            self.dir_toa_true = '/home/hu/uwlc/dataset/RadioToAImage/True'
            self.dir_toa_est = '/home/hu/uwlc/dataset/RadioToAImage/Est'
        elif phase=="test":
            self.ind1=0
            self.ind2=83
            self.matfile = '/home/hu/uwlc/dataset/ToATest/mat/TxRxRandTestnoTx5.npy'
            self.maps_inds = np.arange(0,84,1,dtype=np.int16)
            self.dir_dataset = '/home/hu/uwlc/dataset/ToATest'
            self.dir_gainTrue=self.dir_dataset+"/gain147/"
            self.dir_toa_true = self.dir_dataset+"/ToA/True/"
            self.dir_toa_est = self.dir_dataset+"/ToA/Noise/"
        else: # custom range
            self.ind1=ind1
            self.ind2=ind2
            
        np.random.seed(42)
        np.random.shuffle(self.maps_inds)    
        self.numTx = numTx                
        self.numTrials =  numTrials
        self.numRx = numRx
        self.simulation=simulation
        self.cityMap=cityMap
        self.carsMap=carsMap
        self.TxMaps=TxMaps
        self.transform= transform
        
        self.height = 256
        self.width = 256
        


        self.rss_noise_flag = rss_noise_flag
        
    def __len__(self):
        return (self.ind2-self.ind1+1)*self.numTrials*self.numRx
    
    def __getitem__(self, idx):
        numMapPhase = self.ind2-self.ind1+1
        idxMap,idxTrial,idxRx = np.unravel_index(idx,(numMapPhase,self.numTrials,self.numRx))
        dataset_map_ind=self.maps_inds[idxMap+self.ind1]

        mat = np.load(self.matfile,allow_pickle=True).item()
        rxx = mat['rxx']
        rxy = mat['rxy']
        RXr = rxx[dataset_map_ind,idxRx] - 1
        RXc = rxy[dataset_map_ind,idxRx] - 1
        antList = mat['antList']
        TXlist = antList[idxRx,dataset_map_ind,:]
            
        
        inputEstMaps = []
        for m in range(self.numTx):

            name2 = str(dataset_map_ind) + "_" + str(TXlist[m]-1) + ".png"
            img_name_gainTrue = os.path.join(self.dir_gainTrue, name2)  
            image_gainTrue = np.asarray(io.imread(img_name_gainTrue))/255
            image_gainEst = rss_noise_add(image_gainTrue)/255.  
            inputEstMaps.append(image_gainEst) 
            gainEST = image_gainTrue[RXr,RXc]
            imgGainTrue_Noise = gainEST*np.ones(np.shape(image_gainEst))
            inputEstMaps.append(imgGainTrue_Noise) 


        inputs = inputEstMaps    
                    
        if self.TxMaps:    
            # antX = mat['antX']
            # antY = mat['antY']
            # TXr = antX[idxRx,:,dataset_map_ind]
            # TXc = antY[idxRx,:,dataset_map_ind]
            for m in range(self.numTx):
                imTx = np.zeros((256,256))
                # imTx[TXr[m],TXc[m]] = 1
                inputs.append(imTx)

                
        inputs = np.asarray(inputs, dtype=np.float32)   
        inputs = np.transpose(inputs, (1, 2, 0))

        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)      
  
        #True coordinates      
        RXlocr = torch.from_numpy(np.asarray(RXr, dtype=np.float32))
        RXlocc = torch.from_numpy(np.asarray(RXc, dtype=np.float32))

        if self.return_dic:
            return {
                'feat': inputs, 
                'gt': torch.stack((RXlocr, RXlocc), dim=0)
            }

        return [inputs, torch.stack((RXlocr, RXlocc), dim=0)]

    