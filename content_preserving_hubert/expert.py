"""

SCORE Finetuning for HuBERT

Author: Amit Meghanani

Contact: ameghanani1@sheffield.ac.uk

"""

import math
import os
import random
from pathlib import Path

import s3prl.hub as hub
import torch
import torch.nn as nn
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler

from .dataset import LIBRISPEECH
from .model import Model
from .soft_dtw_cuda import SoftDTW
import wandb
from torch.nn import functional as F

model_ssl = getattr(hub, 'hubert_base')()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model_ssl = model_ssl.to(device)

def SSL_extractor(model_ssl, wavs, device="cuda"):
    
    wavs = [wav.to(device) for wav in wavs]
    with torch.no_grad():
        reps = model_ssl(wavs)["hidden_states"][-1].detach()
    return reps

# define soft dtw

stdw = SoftDTW(use_cuda=False, gamma=0.1)

def normalize_sdtw(x,y):
    return stdw(x,y) - 1/2*(stdw(x,x) + stdw(y,y))

## upstream dimension = 768 for HuBERT base
## upstream rate = 20 ms for HuBERT base

# intialize wandb with project name s3prl-dummy
wandb.init(project="SCORE_Finetuning")

class DownstreamExpert(nn.Module):

    def __init__(self, upstream_dim, upstream_rate, downstream_expert,
                 expdir, **kargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert["datarc"]
        ## modelrc is not used in this expert
        self.modelrc = downstream_expert['modelrc']
        self.expdir = expdir

        self.train_dataset = LIBRISPEECH(root = self.datarc['path'], 
                                                url='train-clean-100', download=True)

        # no model needed for this expert
        print(f"Upstream dim: {upstream_dim}")

        self.connector = nn.Linear(upstream_dim, self.modelrc['input_dim'])

        self.objective = normalize_sdtw

        self.register_buffer("best_score", torch.tensor(float("inf")))

        # set config for wandb
        wandb.config.update(self.datarc)
        wandb.config.update(self.modelrc)


    
    # Interface
    def get_dataloader(self, split, epoch: int = 0):

        if split == "train":
            return self._get_train_dataloader(self.train_dataset, epoch)
        
    def _get_train_dataloader(self, dataset, epoch: int):
        from s3prl.utility.data import get_ddp_sampler
        sampler = get_ddp_sampler(dataset, epoch)

        return DataLoader(
            dataset, batch_size=self.datarc["train_batch_size"],
            shuffle = (sampler is None),
            sampler=sampler,
            num_workers=self.datarc["num_workers"]
        )

    # Interface
    def forward(self, split, features, wav2, records, **kwargs):
        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features]).to(device=device)
        features = pad_sequence(features, batch_first=True)
        features = self.connector(features)

        target_features = SSL_extractor(model_ssl, wav2, device=device)
        target_features_len = torch.IntTensor([len(feat) for feat in target_features]).to(device=device)
        target_features = pad_sequence(target_features, batch_first=True)
        target_features = self.connector(target_features)

        # Note: No model needed for this expert

        # l2 normalization
        features = F.normalize(features, p=2, dim=2)
        target_features = F.normalize(target_features, p=2, dim=2)
        total_seq_len = features.shape[1] + target_features.shape[1]
    
        loss = self.objective(features, target_features).mean()/total_seq_len ## tested only for batch size 1
        records["loss"].append(loss.item())

        return loss

    
    # interface
    def log_records(self, split, records, logger, global_step, **kwargs):
        save_names = []
        for key in ["loss"]:
            values = records[key]
            average = torch.FloatTensor(values).mean().item()
            logger.add_scalar(
                f'content_preserving-{split}/{key}',
                average,
                global_step=global_step
            )
            with open(Path(self.expdir)/ "log.log", 'a') as f:
                if key == 'loss':
                    print(f"\n{split} {key}: {average}")
                    f.write(f'\n{split} at step {global_step}: {average}\n')
                    wandb.log({f'{split}_loss': average}, step=global_step)

                    if split == 'train' and average < self.best_score:
                        self.best_score = torch.ones(1) * average
                        f.write(f'New best on {split} at step {global_step}: {average}\n')
                        save_names.append(f'{split}-best.ckpt')
