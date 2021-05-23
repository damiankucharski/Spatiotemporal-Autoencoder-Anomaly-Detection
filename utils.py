from torchvision.transforms import Resize, Compose, ToPILImage, ToTensor
from PIL import Image
import torch
import tqdm
from diskcache import FanoutCache
from torch.utils.data import Dataset

cache = FanoutCache('tmp')


transform = Compose([ToPILImage(),Resize((227,227)), ToTensor()]) 


@cache.memoize(typed=True, tag='stride')
def generate_stride_set(video_array, stride_size = 1, window_length = 10, name = ''):

    if name:
        print('Generating strides for {}'.format(name))

    end = video_array.shape[-1] - window_length
    windows = []
    for i in tqdm.tqdm(range(0, end, stride_size)):
        x = video_array[..., i:i+window_length]
        transformed_x = []
        for j in range(x.shape[-1]):
            temp = x[..., j]
            temp_transformed = transform(temp)
            transformed_x.append(temp_transformed)
        x = torch.cat(transformed_x)
        windows.append(torch.unsqueeze(x,0))
        # shape  = (10,227,227) 
    
    windows = torch.cat(windows)
    return windows
    

class AnomalyDataset(Dataset):
    
    def __init__(self, X, train=True, fraction = 0.8):

        pivot = int(fraction * len(X))
        if train:
            self.X = X[:pivot]
        else:
            self.X = X[pivot:]

    def __len__(self):

        return len(self.X)
    
    def __getitem__(self, ndx):
        return self.X[ndx, ...]


