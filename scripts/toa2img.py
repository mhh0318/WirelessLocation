import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


f = h5py.File('/home/hu/uwlc/dataset/ToATest/ToA/LocDBDelayReshapedTest.mat', 'r')

print('loading all image now...')
# data = f['matDelayLocReshaped'][:,:,:,:]
data = f['matDelay'][:,:,:,:]
# data = f['TOAs_Noise20m'][:,:,:,:]
print('complete loading all image!!')

a,b,w,h = data.shape

min = data.min()
max = data.max()

# noise_level = 10
# data_mask = data!=0
# noise = np.random.normal(0, noise_level, data.shape)
# data = data + data_mask*noise # IF WE NEED NOISE
print('added noise to data!')

print(f'dataset shape is {a},{b},{w},{h}.')
print(f'dataset min max are {min}, {max}!')

data_norm = (data - min) / (max - min)
data_norm[data_norm<0] = 0  
print(f'normed dataset min max are {data_norm.min()}, {data_norm.max()}!')

for i in tqdm(range(84)):
    # for j in range(99):
    for j in range(80):
        arr_t = data_norm[j,i,:,:] * 255.
        arr_t = arr_t.astype(np.uint8)
        img = Image.fromarray(arr_t.T, mode='L')
        # img.save('./dataset/ToATest/ToA/Noise/{}_{}.png'.format(i,j))
        img.save('./dataset/ToATest/ToA/True/{}_{}.png'.format(i,j))
        # print('saved {}_{}.png'.format(j,i))
