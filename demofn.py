from __future__ import absolute_import
import sys
sys.path.append('./')

# --- Monkey-patch for scipy.misc.imresize ---
import scipy
try:
    from scipy.misc import imresize
except ImportError:
    # If imresize isn't available (as in recent SciPy versions),
    # define it using Pillow.
    from PIL import Image
    import numpy as np
    def imresize(arr, size, interp='bilinear'):
        """
        Resize an image array to the given size.
        :param arr: numpy array representing the image.
        :param size: output size (width, height).
        :param interp: interpolation method (only 'bilinear' is supported here).
        :return: resized image as a numpy array.
        """
        if isinstance(size, (tuple, list)):
            new_size = size
        else:
            new_size = (size, size)
        return np.array(Image.fromarray(arr).resize(new_size, Image.BILINEAR))
    scipy.misc.imresize = imresize
# --- End monkey-patch for imresize ---

import os
import os.path as osp
import argparse
import numpy as np
import math
import time
from PIL import Image, ImageFile

import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

# --- Begin FakeConfig and Helper Functions ---
class FakeConfig:
    """
    A replacement for command-line configuration.
    All parameters are set to their default values.
    Modify as needed.
    """
    def __init__(self, image_path):
        self.seed = 42
        # Set to True if you want to run on GPU (if available).
        self.cuda = False  
        # If height or width is None, main() will default to (32, 100).
        self.height = 64 
        self.width = 256 
        self.voc_type = "ALLCASES_SYMBOLS"
        self.arch = "ResNet_ASTER"
        self.with_lstm = True 
        self.max_len = 10 
        self.STN_ON = True 
        self.beam_width = 5 
        self.tps_inputsize = [32, 64]
        self.tps_outputsize = [32, 100]
        self.tps_margins = [0.05, 0.05]
        self.stn_activation = "none"
        self.num_control_points = 20
        self.resume = "editdistance7333_model.pth.tar"  # Path to the checkpoint.

        self.decoder_sdim = 512
        self.attDim = 512 
        self.image_path = image_path

def image_process(image_path, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
    """
    Loads an image, converts it to RGB, resizes it (with optional aspect ratio preservation),
    then transforms it to a torch tensor.
    """
    img = Image.open(image_path).convert('RGB')
    if keep_ratio:
        w, h = img.size
        ratio = w / float(h)
        imgW = int(np.floor(ratio * imgH))
        imgW = max(imgH * min_ratio, imgW)
    else:
        imgW = imgW
    img = img.resize((imgW, imgH), Image.BILINEAR)
    img = transforms.ToTensor()(img)
    img.sub_(0.5).div_(0.5)
    return img

class DataInfo(object):
    """
    Saves information about the dataset.
    Code snippet adapted from your original dataset.py.
    """
    def __init__(self, voc_type):
        super(DataInfo, self).__init__()
        self.voc_type = voc_type
        assert voc_type in ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
        self.EOS = 'EOS'
        self.PADDING = 'PADDING'
        self.UNKNOWN = 'UNKNOWN'
        from lib.utils.labelmaps import get_vocabulary
        self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
        self.char2id = dict(zip(self.voc, range(len(self.voc))))
        self.id2char = dict(zip(range(len(self.voc)), self.voc))
        self.rec_num_classes = len(self.voc)
# --- End FakeConfig and Helper Functions ---

# --- Begin Library Imports ---
from lib import datasets, evaluation_metrics, models
from lib.models.model_builder import ModelBuilder
from lib.datasets.dataset import LmdbDataset, AlignCollate
from lib.loss import SequenceCrossEntropyLoss
from lib.trainers import Trainer
from lib.evaluators import Evaluator
from lib.utils.logging import Logger, TFLogger
from lib.utils.serialization import load_checkpoint, save_checkpoint
from lib.utils.osutils import make_symlink_if_not_exists
from lib.evaluation_metrics.metrics import get_str_list
# --- End Library Imports ---


args = FakeConfig('') # temporarily do this for model initialization 
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# Set args.cuda flag based on availability.
args.cuda = args.cuda and torch.cuda.is_available()
if args.cuda:
    print('Using CUDA.')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if args.height is None or args.width is None:
    args.height, args.width = (32, 100)

dataset_info = DataInfo(args.voc_type)

# Build the model.
model = ModelBuilder(arch=args.arch, rec_num_classes=dataset_info.rec_num_classes,
                        sDim=args.decoder_sdim, attDim=args.attDim, max_len_labels=args.max_len,
                        eos=dataset_info.char2id[dataset_info.EOS], STN_ON=args.STN_ON)

# Load model checkpoint.
if args.resume:
    if args.cuda:
        checkpoint = load_checkpoint(args.resume)
    else:
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
    # Allow unexpected keys (e.g. for extra RNN parameters) by using strict=False.
    model.load_state_dict(checkpoint['state_dict'], strict=False)

device = torch.device("cuda") if args.cuda else torch.device("cpu")
if args.cuda:
    model = model.to(device)
    model = nn.DataParallel(model)

model.eval()




def main(args):
    """
    Main function for processing the image.
    It sets random seeds, builds the model, loads a checkpoint,
    processes the input image, and prints (and returns) the recognized text.
    """
    

    # --- Begin monkey-patch for index_select ---
    # Patch torch.Tensor.index_select to ensure the index tensor is in int32/int64.
    _orig_index_select = torch.Tensor.index_select
    def patched_index_select(self, dim, index):
        if not index.dtype in (torch.int32, torch.int64):
            index = index.long()
        return _orig_index_select(self, dim, index)
    torch.Tensor.index_select = patched_index_select
    # --- End monkey-patch for index_select ---

    img = image_process(args.image_path, imgH=args.height, imgW=args.width)
    img = img.to(device)
    input_dict = {}
    input_dict['images'] = img.unsqueeze(0)
    rec_targets = torch.IntTensor(1, args.max_len).fill_(1)
    rec_targets[:, args.max_len - 1] = dataset_info.char2id[dataset_info.EOS]
    input_dict['rec_targets'] = rec_targets
    input_dict['rec_lengths'] = [args.max_len]
    
    output_dict = model(input_dict)

    #print("INPUT DICT:", input_dict) 
    #print("OUTPUT DICT:", output_dict)
    
    # Revert the monkey-patch.
    torch.Tensor.index_select = _orig_index_select

    pred_rec = output_dict['output']['pred_rec']
    pred_str, _ = get_str_list(pred_rec, input_dict['rec_targets'], dataset=dataset_info)
    result_str = pred_str[0]
    #print('Recognition result: {0}'.format(result_str))
    return result_str



def run_demo(image_path):
    """
    Entry point for the demo.
    Takes an image path as input, creates a FakeConfig,
    then calls main() to process the image and return the recognized text.
    """
    args = FakeConfig(image_path)
    return main(args)






if __name__ == '__main__':
    # When running from the command line, use a default image path.

    text = run_demo("./data/demos/gPzE.png")

    import time 
    start = time.time() 
    for _ in range(30): run_demo('./data/demos/gPzE.png')
    end = time.time()

    print("Average time for 30 runs: ", (end-start)/30)


    import matplotlib.pyplot as plt 
    import cv2 
    img = cv2.imread("./data/demos/gPzE.png")
    plt.imshow(img)
    plt.title("DETECTED: {}".format(text))

    plt.savefig('./data/demos/gPzE_detection.png')

