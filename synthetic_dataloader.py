# Dataloader with compatible format to main.py, but with synthetic data. 

from torch.utils import data
from gen_img_txt import get_img_text 
from lib.utils.labelmaps import get_vocabulary, labels2strs
import numpy as np 

class SyntheticDataset(data.Dataset):
    def __init__(self, num_samples, max_len=11, initial_seed=10, transform=None, voc_type="ALLCASES"): # transform will be Image.Image -> Image.Image 
        super(SyntheticDataset, self).__init__()
        
        self.num_samples = num_samples 
        self.max_len = max_len # NOTE: this max len includes EOS token 
        self.initial_seed = initial_seed 
        self.transform = transform 
        self.voc_type = voc_type


        assert voc_type in ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
        self.EOS = 'EOS'
        self.PADDING = 'PADDING'
        self.UNKNOWN = 'UNKNOWN'
        self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
        self.char2id = dict(zip(self.voc, range(len(self.voc))))
        self.id2char = dict(zip(range(len(self.voc)), self.voc))

        self.rec_num_classes = len(self.voc)
        self.lowercase = (voc_type == 'LOWERCASE')

    
    def __len__(self): 
        return self.num_samples 

    def __getitem__(self, idx): 
        seed = self.initial_seed + idx 
        img, text = get_img_text(seed, self.max_len-1) # since there's an EOS token 
        assert img.mode == 'RGB', "get_img_text image mode is not RGB, but {}".format(image.mode)

        if self.transform:  
            img = self.transform(img) 
        


        ## convert text to numpy array of classes 

        if self.lowercase:
            text = text.lower()
        
        ## fill with the padding token
        label = np.full((self.max_len,), self.char2id[self.PADDING], dtype=np.int32)
        label_list = []
        for char in text:
            if char in self.char2id:
                label_list.append(self.char2id[char])
            else:
                ## add the unknown token
                print('{0} is out of vocabulary.'.format(char))
                label_list.append(self.char2id[self.UNKNOWN])
        
        ## add a stop token
        label_list = label_list + [self.char2id[self.EOS]]
        assert len(label_list) <= self.max_len
        label[:len(label_list)] = np.array(label_list)


        return img, label_list, len(label_list) # img is an Image.Image 


# image sizing and keep aspect ratio are handled by AlignCollate in lib.datasets.dataset.py 


# test 
if __name__=="__main__": 
  from PIL import Image 
  from lib.datasets.dataset import AlignCollate, labels2strs 
  from lib.utils import to_numpy

  train_dataset = SyntheticDataset(num_samples=1000)
  batch_size = 3
  train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=AlignCollate(imgH=64, imgW=256, keep_ratio=False))

  for i, (images, labels, label_lens) in enumerate(train_dataloader):
    print("NEW BATCH")
    # visualization of input image
    # toPILImage = transforms.ToPILImage()
    images = images.permute(0,2,3,1)
    images = to_numpy(images)
    images = images * 0.5 + 0.5
    images = images * 255
    for id, (image, label, label_len) in enumerate(zip(images, labels, label_lens)):
      image = Image.fromarray(np.uint8(image))
      # image = toPILImage(image)
      image.show()
      print(image.size)
      print(labels2strs(label, train_dataset.id2char, train_dataset.char2id))
      print(label_len.item())
      a = input()

