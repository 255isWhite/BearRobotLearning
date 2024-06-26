import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import copy

from PIL import Image

from BearRobot.Agent.base_agent import BaseAgent
from BearRobot.Net.my_model.diffusion_model import VisualDiffusion
from BearRobot.Net.my_model.t5 import T5Encoder
from BearRobot.Net.encoder.DecisionNCE import DecisionNCE_encoder, DecisionNCE_lang

class MiniBC_Agent(BaseAgent):
    def __init__(
        self,
        policy: torch.nn.Module,
        lang_encoder: str="DecisionNCE-T",
        ac_num: int=1,
        **kwargs
    ):
        super().__init__(
            policy, None, None, ac_num=ac_num
        )

        assert lang_encoder in ['DecisionNCE-T', 'DecisionNCE-P'], f"lang_encoder {lang_encoder} not supported"
        mm_encoder = DecisionNCE_encoder(lang_encoder, device=self.device)
        self.lang_encoder = DecisionNCE_lang(mm_encoder)
    
    
    def forward(self, images: torch.Tensor, texts: list, action_gt: torch.Tensor, state=None):
        '''
        calculate bc loss
        Input:
        images: [B, F, V, C, H, W] batch of frames of different views
        texts: [B, D] batch of instruction embbeding
        action_gt: [B, D_a] ground truth action
        state: [B, D_s] batch of robot arm x,y,z, gripper state
        
        return: loss
        '''
        text_emb = self.lang_encoder.embed_text(texts).to(images.device).detach() if self.lang_encoder is not None else texts
        loss = self.policy_loss(action_gt, images, text_emb, state)
        loss_dict = {"policy_loss": loss}
        return loss_dict
    
    
    def policy_loss(self, action_gt: torch.Tensor, imgs: torch.Tensor, texts: torch.Tensor, state: torch.Tensor=None):
        '''
        calculate bc loss
        Input:
        action_gt: [B, D_a] ground truth action
        imgs: [B, F, V, C, H, W] batch of frames of different views
        texts: [B, D] batch of instruction embbeding
        state: [B, D_s] batch of robot arm x,y,z, gripper state
        
        return: loss
        '''
        action_pred = self.predict_action(imgs, texts, state)

        # normilize action_gt
        B, D_a = action_pred.shape
        action_pred = action_pred.view(B, -1, 7)
        B, N, D_a = action_pred.shape
        a_max = self.a_max.repeat(B, N, 1).to(action_pred.device)
        a_min = self.a_min.repeat(B, N, 1).to(action_pred.device)

        action_gt = action_gt.view(B,-1,7)
        action_gt = action_gt.to(action_pred.device)
        a_mid = (a_max+a_min)/2
        action_gt = (action_gt-a_mid)/(a_mid-a_min) # [-1,1]

        loss = (((action_pred - action_gt) ** 2).sum(axis = -1)).mean()
        
        return loss
    
    
    def predict_action(self, imgs: torch.Tensor, texts: torch.Tensor, state: torch.Tensor=None):
        '''
        Predict the action
        
        Input:
        imgs: [B, F, V, C, H, W] batch of frames of different views
        texts: [B, D] batch of instruction embbeding
        state: [B, D_s] batch of robot arm x,y,z, gripper state
        
        Return: [B, D_a] predicted action
        '''
        state = ((state - self.s_mean.to(state.device)) / self.s_std.to(state.device)) if state is not None else None
        action_pred = self.policy(imgs, texts, state)
        return action_pred
    
    
    @torch.no_grad()
    def get_action(self, imgs, lang, state=None, t=1, k=0.25):
        if not isinstance(imgs, torch.Tensor):
            # transform lists to torch.Tensor
            imgs = torch.stack([self.transform(Image.fromarray(frame).convert("RGB")) for frame in imgs]).unsqueeze(0).unsqueeze(0).to('cuda')
        else:
            imgs = imgs.to('cuda')

        B, F, V, C, H, W = imgs.shape
        try:
            s_dim = self.policy.s_dim
        except:
            s_dim = self.policy.module.s_dim
        state = torch.from_numpy(state.astype(np.float32)).view(-1, s_dim) if state is not None else None
        state = ((state - self.s_mean) / self.s_std).to('cuda') if state is not None else None
        
        try:
            output_dim = self.policy.output_dim
        except:
            output_dim = self.policy.module.output_dim
        action = self.predict_action(imgs,[lang]*B,state).detach().cpu()
        action = action.view(B, -1, 7)
        
        B, N, D_a = action.shape
        a_max = self.a_max.repeat(B, N, 1)
        a_min = self.a_min.repeat(B, N, 1)
        a_mean = self.a_mean.repeat(B, N, 1)
        a_std = self.a_std.repeat(B, N, 1)
        
        action = (action + 1) * (a_max - a_min) / 2 + a_min
        action = self.get_ac_action(action.numpy(), t, k)
        # action = action * a_std + a_mean
        return action  