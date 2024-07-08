import torch
import utility
import data
import model
import loss
import os
from option import args
from trainer import Trainer
import numpy as np
import random
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="0"
os.environ['CUDA_CACHE_PATH']='~/.cudacache'
torch.cuda.empty_cache()
checkpoint = utility.checkpoint(args)
seed = args.seed
torch.manual_seed(seed) # 为CPU设置随机种子
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.	
os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def main():
    global model
    if checkpoint.ok:
        loader = data.Data(args)
        _model = model.Model(args, checkpoint)
        _loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, _model, _loss, checkpoint)
        while not t.terminate():
            t.train()
            t.test()
        checkpoint.done()

if __name__ == '__main__':
    main()
