import torch
import sys
import cv2
import os
import math
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from ribs.schedulers import Scheduler
from ribs.archives import GridArchive
from ribs.emitters import (GaussianEmitter, GradientArborescenceEmitter,
                           EvolutionStrategyEmitter, IsoLineEmitter)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    STRIKETHROUGH = '\u0336'
    
def tensor_to_pil_img(img: torch.tensor) -> Image:
    img = (img.clamp(-1, 1) + 1) / 2.0
    img = img[0].permute(1, 2, 0).detach().cpu().numpy() * 255
    img = Image.fromarray(img.astype('uint8'))
    return img

class Generator():
    def __init__(self,
                 task: str,
                 device: torch.device):
        self.task = task
        self.device = device

        if task == 'shapes':
            from shapes.generator import VariationalAutoencoder
            self.backbone = VariationalAutoencoder(latent_dims=6)
            self.latent_shape = (-1, 6)
            self.backbone.load_state_dict(torch.load('shapes/pretrained/vae-shapes.pt'))
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.to(self.device)
        elif task == 'facial_recognition':
            from facial_recognition.generator import StyleGAN2
            self.backbone = StyleGAN2(device=device)
            self.latent_shape = (-1, 512)

    def randn(self,
              n: int) -> torch.Tensor:
        if self.task == 'shapes':
            sols = torch.randn(n, 6).to(self.device)
        elif self.task == 'facial_recognition':
            G = self.backbone.model
            w_stds = self.backbone.w_stds
            sols = (G.mapping(torch.randn([n, G.mapping.z_dim], device=self.device),
                    None, truncation_psi=0.7) - G.mapping.w_avg) / w_stds
            
        return sols

    def preprocess_synthesis(self, 
                             sols):
        # Transforms solutions from the q-space (normalized) to w-space
        sols_with_grad = []
        sols_for_gen = []
        for cur_code in sols:
            if torch.is_tensor(sols):
                sol = cur_code.view(self.latent_shape)
            else:
                sol = torch.tensor(
                        cur_code.reshape(self.latent_shape), 
                        device=self.device,
                        requires_grad=True,
                        dtype=torch.float
                    )
            sols_with_grad.append(sol)
            if self.task == 'facial_recognition':
                w = sol * self.backbone.w_stds + self.backbone.model.mapping.w_avg
                sols_for_gen.append(w)

        if self.task == 'shapes':
            return sols_with_grad
        elif self.task == 'facial_recognition':
            sols_for_gen = torch.stack(sols_for_gen, dim=0)
            return sols_with_grad, sols_for_gen
    
    def synthesis(self,
                  sols: torch.tensor):
        if self.task == 'shapes':
            sols = self.preprocess_synthesis(sols)
            images = self.backbone.decoder(torch.cat(sols))
        elif self.task == 'facial_recognition':
            sols, sols_for_gen = self.preprocess_synthesis(sols)
            images = self.backbone.model.synthesis(sols_for_gen, noise_mode='const')
            images = images.add(1).div(2)
            
        return images, sols
    

def create_scheduler(algorithm: str, 
                     discriminator,
                     grid_dims: List[int],
                     batch_size: int,
                     seed: int) -> Tuple[Scheduler, GridArchive]:
    """Creates an scheduler based on the algorithm name."""
    num_emitters = 1
    # Compute starting solutions and archive
    initial_sol = discriminator.find_good_start_latent(num_batches=32).cpu().detach().numpy()
    solution_dim = len(initial_sol)

    ranges = [(-0.1, 0.1) for _ in grid_dims]
    # Maintain a passive elitist archive
    passive_archive = GridArchive(solution_dim=solution_dim, 
                                  dims=grid_dims, 
                                  ranges=ranges, 
                                  seed=seed)

    if algorithm in [
            "map_elites", "map_elites_line", 
            "cma_me", "cma_mega", 
    ]:
        archive = GridArchive(
                solution_dim=solution_dim,
                dims=grid_dims, 
                ranges=ranges,
                seed=seed
        )
    elif algorithm in ["cma_mae", "cma_maega"]:
        archive = GridArchive(
                solution_dim=solution_dim,
                dims=grid_dims, 
                ranges=ranges, 
                learning_rate=0.02,
                threshold_min=0.0,
                seed=seed,
        )
    else:
        raise ValueError(f"Algorithm `{algorithm}` is not recognized")

    # Create emitters. Each emitter needs a different seed, so that they do not
    # all do the same thing.
    emitter_seeds = [None] * num_emitters if seed is None else list(
        range(seed, seed + num_emitters))
    if algorithm in ["map_elites"]:
        emitters = [
            GaussianEmitter(archive,
                            initial_sol,
                            0.1,
                            batch_size=batch_size,
                            seed=s) for s in emitter_seeds
        ]
    elif algorithm in ["map_elites_line"]:
        emitters = [
            IsoLineEmitter(archive,
                           initial_sol,
                           iso_sigma=0.1,
                           line_sigma=0.2,
                           batch_size=batch_size,
                           seed=s) for s in emitter_seeds
        ]
    elif algorithm in ["cma_mega"]:
        emitters = [
            GradientArborescenceEmitter(
                    archive=archive,
                    x0=initial_sol,
                    sigma0=0.5, # 0.01, 0.05, 0.1
                    lr=0.5, # 0.5
                    ranker='2imp',
                    grad_opt="adam",
                    normalize_grad=True,
                    restart_rule='no_improvement',
                    selection_rule='filter',
                    bounds=None,
                    batch_size=batch_size - 1,
                    seed=s) for s in emitter_seeds
        ]
    elif algorithm in ["cma_maega"]:
        emitters = [
            GradientArborescenceEmitter(
                    archive=archive,
                    x0=initial_sol,
                    sigma0=0.5, # 0.01, 0.05, 0.1
                    lr=0.5, # 0.5
                    ranker='imp',
                    grad_opt="adam",
                    normalize_grad=True,
                    restart_rule='basic',
                    bounds=None,
                    batch_size=batch_size - 1,
                    seed=s) for s in emitter_seeds
        ]
    elif algorithm in ["cma_me"]:
        emitters = [
            EvolutionStrategyEmitter(archive,
                                     initial_sol,
                                     0.1,
                                     ranker='2imp',
                                     batch_size=batch_size,
                                     seed=s) for s in emitter_seeds
        ]
    elif algorithm in ["cma_mae"]:
        emitters = [
            EvolutionStrategyEmitter(archive=archive,
                                     x0=initial_sol,
                                     sigma0=0.1,
                                     ranker="imp",
                                     restart_rule='basic',
                                     batch_size=batch_size,
                                     seed=s) for s in emitter_seeds
        ]

    return Scheduler(archive, emitters), passive_archive


def embed_archive(archive_df,
                  gen_model,
                  clip_model,
                  batch_size = 100):
    print("> Begin embedding images...")
    clip_features = []
    with torch.no_grad():
        for i_batch in tqdm(range((len(archive_df) + 1) // batch_size)):
            ## GENERATE IMAGE ###
            sol = archive_df.loc[i_batch * batch_size : (i_batch + 1) * batch_size, "solution_0":].values
            images, _ = gen_model.synthesis(sol)      
            clip_features.append(clip_model.embed_image(images)[0].cpu())
    clip_features = torch.cat(clip_features).numpy()
    archive_embedded = pd.DataFrame(clip_features)

    return archive_embedded


def write_data(outdir,
               task,
               measure_bins_list,
               archive_df,
               gen_model,
               device):
    if task == 'facial_recognition':
        sys.path.insert(0, 'facial_recognition/AdaFace')
        from face_alignment import align

    label_iterator = -1

    for measure_bins in measure_bins_list:
        # Subset the measure criteria
        archive_bin = archive_df
        for j, measure_bin in enumerate(measure_bins):
            crit = (archive_bin[f"measure_{j}"] >= measure_bin[0]) & \
                (archive_bin[f"measure_{j}"] < measure_bin[1])
            archive_bin = archive_bin[crit]
        
        sols_in_cell = torch.tensor(archive_bin.loc[:, 'solution_0':].values, 
                                    dtype=torch.float, 
                                    device=device)
        if len(sols_in_cell) == 0:
            continue

        if task == 'shapes':
            label_iterator += 1
            ### MAKE DIRECTORY ###
            subjectdir = Path(f'{outdir}/{label_iterator}')
            if not subjectdir.is_dir():
                subjectdir.mkdir()
            ### GENERATE IMAGE ###
            for img_id, sol in enumerate(sols_in_cell):
                sols = sol.unsqueeze(0)
                img, _ = gen_model.synthesis(sols)
                img = img[0].clamp(0, 1).mul(255)
                # Resize image
                img = img.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
                img = Image.fromarray(img)
                img.save(os.path.join(f'{outdir}/{label_iterator}', f"{int(img_id):03d}.jpg"))
        elif task == 'facial_recognition':
            imgs_per_subject = 2
            latent_augmentations = torch.load(f'facial_recognition/pretrained/augmentations.pt').to(device)

            # Cluster similar solutions
            subject_idx_list = {}
            n_clusters = math.ceil(len(sols_in_cell) / imgs_per_subject)
            clusters = KMeans(n_clusters, n_init=10, max_iter=2000).fit(sols_in_cell).labels_
            for i in range(n_clusters):
                subject_idx_list[i] = np.where(clusters == i)[0]

            for subject_idx in tqdm(subject_idx_list):
                img_id = 0
                label_iterator += 1
                ### MAKE DIRECTORY ###
                subjectdir = Path(f'{outdir}/{label_iterator}')
                if not subjectdir.is_dir():
                    subjectdir.mkdir()
                ### GENERATE IMAGE ###
                for neighbor_idx in subject_idx_list[subject_idx]:
                    sols = sols_in_cell[neighbor_idx].unsqueeze(0).to(device)
                    sols = sols.repeat_interleave(len(latent_augmentations), axis=0) + latent_augmentations
                    images, _ = gen_model.synthesis(sols)
                    images = images.clamp(0, 1)
                    for img in images:
                        img_id += 1
                        # Resize image
                        img = cv2.resize(img.permute(1, 2, 0).detach().cpu().numpy() * 255, (256, 256)).astype('uint8')
                        img = Image.fromarray(img)
                        aligned_img = align.get_aligned_face("", img)
                        aligned_img.save(os.path.join(f'{outdir}/{label_iterator}', f"{int(img_id):03d}.jpg"))