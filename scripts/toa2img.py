import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


f = h5py.File('/home/hu/uwlc/dataset/RadioToASeer/LocDBDelay_Noise20m.mat', 'r')

print('loading all image now...')
# data = f['matDelayLocReshaped'][:,:,:,:]
data = f['TOAs_Noise20m'][:,:,:,:]
print('complete loading all image!!')

a,b,w,h = data.shape
min = data.min()
max = data.max()
print(f'dataset shape is {a},{b},{w},{h}.')
print(f'dataset min max are {min}, {max}!')

data_norm = (data - min) / (max - min)
print(f'normed dataset min max are {data_norm.min()}, {data_norm.max()}!')

for i in tqdm(range(80)):
    for j in range(99):
        arr_t = data_norm[i,j,:,:] * 255.
        arr_t = arr_t.astype(np.uint8)
        img = Image.fromarray(arr_t.T, mode='L')
        img.save('./dataset/RadioToAImage/Est/{}_{}.png'.format(j,i))
        # print('saved {}_{}.png'.format(j,i))
