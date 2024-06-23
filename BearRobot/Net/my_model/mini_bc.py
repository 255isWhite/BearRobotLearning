import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from BearRobot.Net.basic_net.mlp import MLP, MLPResNet
from BearRobot.Net.basic_net.resnet import ResNet
from BearRobot.Net.my_model.FiLM import FiLM_layer
from BearRobot.Net.encoder.DecisionNCE import DecisionNCE_encoder, DecisionNCE_visual, DecisionNCE_lang
from . import LANG_EMB_DIM

class MiniBC_pretrain(nn.Moudle):
        """
        the mini behavior cloning model that uses pretrained mult-modal backbone
        """
        def __init__(
              self, 
              view_num: int=2,
              output_dim: int=7,  # a dim
              cond_dim: int=768,  # cond dim, if condition on s
              s_dim: int=0,  # qpos_dim, use qpos when > 0 
              hidden_dim: int=256,
              num_blocks: int=3,
              time_dim: int=32,
              time_hidden_dim: int=256,
              mm_encoder: str='DecionNCE-T',
              ft_mmencoder: bool=True,
              ac_fn: str='mish',
              time_embed: str='learned',
              film_fusion: bool=False,
              encode_s: bool=False,
              encode_a: bool=False,
              device: str='cpu',
              *args,
              **kwargs
        ):
            super().__init__()
            
            self.output_dim = output_dim
            self.ft_mmencoder = ft_mmencoder
            self.cond_dim = cond_dim
            self.device = device
            
            assert mm_encoder in ['DecisionNCE-T', 'DecisionNCE-P'], f"mm_encoder {mm_encoder} not supported"
            mm_encoder = DecisionNCE_encoder(mm_encoder, device=device)
            
            if not ft_mmencoder:
                for _, p in mm_encoder.named_parameters():
                    p.requires_grad = False
            else:
                mm_encoder.model.requires_grad_(False)
                mm_encoder.model.model.visual.requires_grad_(True)
                
            self.visual_encoder = DecisionNCE_visual(mm_encoder)
            self.lang_encoder = DecisionNCE_lang(mm_encoder)
            self.visual_dim = 1024
            self.lang_dim = 1024
            self.img_size = 0
            
            self.s_dim = s_dim
            if encode_s and s_dim>0:
                self.state_encoder = nn.Linear(s_dim, hidden_dim) 
                self.s_dim = hidden_dim 
            if encode_a:
                pass
            if film_fusion:
                pass
            
            input_dim = self.visual_dim * view_num + self.lang_dim + self.s_dim
            self.decoder = MLP(input_dim,[hidden_dim,hidden_dim],output_dim)
            
        
        def forward(self, imgs: torch.Tensor, texts: torch.Tensor, state: torch.Tensor=None):
            if self.s_dim>0 :
                state = state.reshape([state.shape[0], -1])
                state_embeddings = self.state_encoder(state) if self.encode_s else state
            else:
                state_embeddings = None
                
            ## Image embeddings
            B, F, V, C, H, W = imgs.shape
            img_embeddings = imgs.view(B*F*V, C, H, W)
            img_embeddings = self.visual_encoder(img_embeddings)
            img_embeddings = img_embeddings.view(B, F*V*self.visual_dim)
            
            ## Text embeddings
            if isinstance(texts, list):
                text_embeddings = self.lang_encoder(texts)
            else:
                raise ValueError(f"Need type [list], but got type [{type(texts)}]")
            
            input_embeddings = torch.concat([img_embeddings, text_embeddings, state_embeddings], dim=-1)\
                if state_embeddings is not None else torch.concat([img_embeddings, text_embeddings], dim=-1)
                
            action_pred = self.decoder(input_embeddings)

            return action_pred