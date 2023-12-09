import pickle
import torch

class StyleGAN2(object):
    def __init__(self, 
                 device: torch.device):
        self.latent_shape = (-1, 512)
        self.device = device

        # Load model
        model_filename = 'facial_recognition/pretrained/stylegan2-ffhq-256x256.pkl'
        with open(model_filename, 'rb') as fp:
            self.model = pickle.load(fp)['G_ema'].to(device)
            self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Init w stats
        self.init_stats()

    def init_stats(self) -> None:
        # Compute initial std(w)
        zs = torch.randn([10000, self.model.mapping.z_dim], device=self.device)
        ws = self.model.mapping(zs, None)
        self.w_stds = ws.std(0)
        # Compute initial qs
        qs = ((ws - self.model.mapping.w_avg) / self.w_stds).reshape(10000, -1)
        self.q_norm = torch.norm(qs, dim=1).mean() * 0.5  # 0.2, 0.15

    def gen_random_ws(self, 
                      num_latents: int) -> torch.tensor:
        # Randomly initialize Zs
        zs = torch.randn([num_latents, self.model.mapping.z_dim], device=self.device)
        # Map to w values
        ws = self.model.mapping(zs, None)
        return ws