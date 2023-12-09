# Third-party library imports
import clip
from einops import rearrange
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


def norm1(prompt):
    return prompt / prompt.square().sum(dim=-1, keepdim=True).sqrt()


class MakeCutouts(torch.nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.0):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, x):
        sideY, sideX = x.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = x[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)
    

class CLIP(object):
    def __init__(self, 
                 device: torch.device, 
                 cutn: int = 1):
        self.device = device
        self.make_cutouts = MakeCutouts(224, cutn, 0.5)

        clip_model_name = "ViT-B/32"
        self.model, _ = clip.load(clip_model_name, device=device)
        self.model = self.model.requires_grad_(False)
        self.model.eval()
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                              std=[0.26862954, 0.26130258, 0.27577711])

    @torch.no_grad()
    def tokenize(self, prompt):
        return clip.tokenize(prompt)
        
    def embed_text(self, prompt):
        return norm1(self.model.encode_text(self.tokenize(prompt)
               .to(self.device)).float())

    def embed_cutout(self, image):
        return norm1(self.model.encode_image(self.normalize(image)))

    def embed_image(self, image):
        n = image.shape[0]
        cutouts = self.make_cutouts(image)
        embeds = self.embed_cutout(cutouts)
        embeds = rearrange(embeds, '(cc n) c -> cc n c', n=n)
        return embeds
