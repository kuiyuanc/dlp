from typing import Callable
import torch
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])

        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    def save_transformer_checkpoint(self, save_ckpt_path):
        torch.save(self.transformer.state_dict(), save_ckpt_path)

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True)
        model = model.eval()
        return model

##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x: torch.Tensor) -> torch.Tensor:
        _, z_indices, _ = self.vqgan.encode(x)  # z_indices.shape: (batch_size * height * width, )
        batch_size = x.size(0)
        return z_indices.view(batch_size, -1)

##TODO2 step1-2:
    def gamma_func(self, mode: str = "cosine") -> Callable[[float], np.float32]:
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1].
        During training, the input ratio is uniformly sampled;
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.

        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            return lambda r: np.float32(1 - r)
        elif mode == "cosine":  # concave
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":  # concave
            return lambda r: 1 - np.square(r)
        elif mode == "sqrt":  # convex
            return lambda r: 1 - np.sqrt(r)
        elif mode == "log":
            return lambda r: 1 - np.log((np.e - 1) * r + 1)
        else:
            raise NotImplementedError

##TODO2 step1-3:
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        z_indices: torch.Tensor=self.encode_to_z(x) #ground truth
        mask_ratio = torch.full_like(z_indices, np.random.uniform(0, 1), dtype=torch.float32, device=x.device)
        mask = torch.bernoulli(mask_ratio).bool()
        z_masked = z_indices.masked_fill(mask, self.mask_token_id)
        logits = self.transformer(z_masked)  #transformer predict the probability of tokens

        return logits, z_indices

##TODO3 step1-1: define one iteration decoding
    @torch.no_grad()
    def inpainting(self, z_indices: torch.Tensor, mask: torch.Tensor, mask_num: int, r: float, mask_func: str) -> tuple[torch.Tensor, torch.Tensor]:
        z_masked = z_indices.masked_fill(mask, self.mask_token_id)
        logits: torch.Tensor = self.transformer(z_masked)
        #Apply softmax to convert logits into a probability distribution across the last dimension.
        probs: torch.Tensor = logits.softmax(dim=-1)

        #FIND MAX probability for each token value
        z_indices_predict_prob, z_indices_predict = probs.max(dim=-1)
        z_indices_predict_prob = z_indices_predict_prob.masked_fill(~mask, np.inf)

        ratio=self.gamma_func(mask_func)(r)
        #predicted probabilities add temperature annealing gumbel noise as confidence
        g = torch.distributions.gumbel.Gumbel(0, 1).sample(z_indices_predict_prob.size()).to(z_indices.device)  # gumbel noise
        temperature = self.choice_temperature * (1 - ratio)
        confidence: torch.Tensor = z_indices_predict_prob + temperature * g

        #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        #sort the confidence for the rank
        sorted_confidence, _ = confidence.sort(dim=-1)
        n = math.floor(mask_num * ratio)
        threshold = sorted_confidence[:, n].unsqueeze(-1)
        #define how much the iteration remain predicted tokens by mask scheduling
        mask_bc: torch.Tensor = confidence < threshold
        ##At the end of the decoding process, add back the original(non-masked) token values
        z_indices_predict = torch.where(mask, z_indices_predict, z_indices)

        return z_indices_predict, mask_bc

__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
