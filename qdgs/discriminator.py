# Standard library imports
import random
from typing import Tuple, List

# Third-party library imports
import numpy as np
import torch
import torch.nn.functional as F

def prompts_dist_loss(x, targets, loss):
    distances = [loss(x, target) for target in targets]
    return torch.stack(distances, dim=-1).sum(dim=-1)

def cos_sim_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().mul(2)

def memory_dist_loss(x, targets, loss, device):
    distances = [loss(x, target.to(device)) for target in targets]
    return torch.stack(distances, dim=-1).min(dim=-1)[0]
        
class Discriminator(object):
    def __init__(self, 
                 task: str, 
                 objective_prompts: Tuple[str],
                 measure_prompts: List[Tuple],
                 gen_model,
                 clip_model,
                 device: torch.device):
        self.task = task
        self.gen_model = gen_model
        self.clip_model = clip_model
        self.device = device
        self.objective_prompts = self.embed_prompt(*objective_prompts)
        self.measure_prompts = [self.embed_prompt(*prompt) for prompt in measure_prompts]
        self.archive_memory = []

        if self.task == 'shapes':
            self.objective_weights = [1, 0, 0]
        elif self.task == 'facial_recognition':
            self.objective_weights = [1, 0.5, 0.2]

    def embed_prompt(self, 
                     positive_text, 
                     negative_text) -> Tuple[torch.tensor]:
        texts = [frase.strip() for frase in positive_text.split("|") if frase]
        positive_targets = [self.clip_model.embed_text(text) for text in texts]

        texts = [frase.strip() for frase in negative_text.split("|") if frase]
        negative_targets = [self.clip_model.embed_text(text) for text in texts]
        
        return (positive_targets, negative_targets)

    def find_good_start_latent(self, 
                               num_batches: int) -> torch.tensor:
        with torch.inference_mode():
            sols_list = []
            loss_list = []
            for _ in range(num_batches):
                sols = self.gen_model.randn(16)
                images, sols = self.gen_model.synthesis(sols)
                loss = self.compute_objective_loss(images, torch.cat(sols))
                i = torch.argmin(loss)
                sols_list.append(sols[i])
                loss_list.append(loss[i])
            sols_list = torch.stack(sols_list)
            loss_list = torch.stack(loss_list)

            i = torch.argmin(loss_list)
            sols = sols_list[i].unsqueeze(0)

        return sols.flatten()

    def transform_obj(self, 
                      objs: torch.tensor) -> torch.tensor:
        # Remap the objective from minimizing [0, 10] to maximizing [0, 100]
        return objs.mul(-5).add(10).mul(10)

    def compute_objective_loss(self, 
                               images: torch.tensor, 
                               sols: torch.tensor, 
                               dim: Tuple = None) -> torch.tensor:
        ### PROMPT LOSS ###
        embeds = self.clip_model.embed_image(images)
        prompt_loss = prompts_dist_loss(embeds, self.objective_prompts[0], cos_sim_loss).mean(0)
        prompt_loss -= prompts_dist_loss(embeds, self.objective_prompts[1], cos_sim_loss).mean(0)
        loss = self.objective_weights[0] * prompt_loss
        ### REGULARIZATION LOSS ###
        if self.objective_weights[1] > 0:
            diff = torch.max(torch.norm(sols, dim=dim), self.gen_model.backbone.q_norm)
            reg_loss = (diff - self.gen_model.backbone.q_norm).pow(2)
            loss += self.objective_weights[1] * reg_loss
        ### UNIQUENESS LOSS ###
        if self.objective_weights[2] > 0:
            if len(self.archive_memory) > 0:
                uniqueness = memory_dist_loss(embeds, self.archive_memory, cos_sim_loss, self.device).mean(0)
                uniqueness = uniqueness.mul(100).div(len(self.archive_memory))
            else:
                uniqueness = torch.ones(images.shape[0], device=self.device)
            loss += self.objective_weights[2] * uniqueness.mul(-1).add(1)

            for i in random.sample(range(len(images)), len(images)):
                if (loss[i] < 1):
                    self.archive_memory.append(embeds[:, i:i + 1, :].cpu().detach())
                    if (len(self.archive_memory) > 100):
                        self.archive_memory = self.archive_memory[1:]
                    break

        return loss

    def compute_objective(self, sols):
        images, sols = self.gen_model.synthesis(sols)
        loss = self.compute_objective_loss(images, sols[0])
        loss.backward()

        jacobian = -sols[0].grad.cpu().detach().numpy()
        return self.transform_obj(loss), jacobian.flatten()

    def compute_measure(self, index, sols):
        images, sols = self.gen_model.synthesis(sols)

        embeds = self.clip_model.embed_image(images)
        measure_targets = self.measure_prompts[index]
        pos_loss = prompts_dist_loss(embeds, measure_targets[0], cos_sim_loss).mean(0)
        neg_loss = prompts_dist_loss(embeds, measure_targets[1], cos_sim_loss).mean(0)
        loss = pos_loss - neg_loss
        loss.backward()

        value = loss.cpu().detach().numpy()
        jacobian = sols[0].grad.cpu().detach().numpy()
        return value, jacobian.flatten()

    def compute_measures(self, sols):
        values = []
        jacobian = []
        for i in range(len(self.measure_prompts)):
            value, jac = self.compute_measure(i, sols)
            values.append(value)
            jacobian.append(jac)

        return np.stack(values, axis=0), np.stack(jacobian, axis=0)

    def compute_all(self, 
                    sols: torch.tensor,
                    use_grad: bool) -> np.array:
        with torch.set_grad_enabled(use_grad):
            images, sols = self.gen_model.synthesis(sols)
            sols = torch.stack(sols, dim=0)

            embeds = self.clip_model.embed_image(images)
            obj_loss = self.compute_objective_loss(images, sols, dim=(1,2))
            measures = []
            for measure_targets in self.measure_prompts:
                pos_loss = prompts_dist_loss(embeds, measure_targets[0], cos_sim_loss).mean(0)
                neg_loss = prompts_dist_loss(embeds, measure_targets[1], cos_sim_loss).mean(0)
                loss = pos_loss - neg_loss
                measures.append(loss)
        return self.transform_obj(obj_loss), torch.stack(measures, axis=1)