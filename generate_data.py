# Standard library imports
import itertools

# Third-party library imports
import fire
import numpy as np
import torch
from alive_progress import alive_bar

# External package imports
from qdgs import (Generator, Discriminator, Writer, CLIP)
from qdgs.utils import (bcolors, create_scheduler, write_data)

from ribs.archives import GridArchive
from ribs.schedulers import Scheduler


def init_device() -> torch.device:
    """
    Initialize and return the appropriate Torch device based on GPU availability.

    Returns:
        torch.device: The selected device (GPU if available, otherwise CPU).
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"> Using device: {bcolors.OKGREEN}{device}{bcolors.ENDC}")
    return device


def compute_iter(itr: int,
                 best: int,
                 algorithm: str,
                 scheduler: Scheduler,
                 passive_archive: GridArchive,
                 discriminator: Discriminator,
                 writer: Writer) -> int:
    """
    Perform one iteration of solution branching and evaluation.

    Args:
        itr (int): The current iteration number.
        best (int): The overall best objective value.
        algorithm (str): The name of the quality-diversity algorithm.
        scheduler (Scheduler): The quality-diversity scheduler.
        passive_archive (GridArchive): The passive elitist archive.
        discriminator (Discriminator): The discriminator model.
        writer (Writer): The writer for logging.

    Returns:
        int: Updated overall best objective value.
    """
    is_dqd = algorithm in ['cma_mega', 'cma_mega_adam', 'cma_maega']

    if is_dqd:
        # Get solution from scheduler
        sols = scheduler.ask_dqd()

        # Compute objectives and measures
        objs, jacobian_obj = discriminator.compute_objective(sols)
        objs = objs.cpu().detach().numpy()
        measures, jacobian_measure = discriminator.compute_measures(sols)
        measures = np.transpose(measures) 

        jacobian_obj = np.expand_dims(jacobian_obj, axis=0) 
        jacobian = np.concatenate((jacobian_obj, jacobian_measure), axis=0)
        jacobian = np.expand_dims(jacobian, axis=0)

        objs = objs.astype(np.float32)
        measures = measures.astype(np.float32)
        jacobian = jacobian.astype(np.float32)

        best_gen = max(objs)
        best = max(best, best_gen)

        # Update scheduler
        scheduler.tell_dqd(objs, measures, jacobian_batch=jacobian)

        # Update the passive elitist archive.
        passive_archive.add(sols, objs, measures)
    
    #################### ASK SOLUTIONS ####################
    sols = scheduler.ask()
    objs, measures = discriminator.compute_all(sols, use_grad=False)

    best_gen = objs.max().item()
    best = max(best, best_gen)
    #################### TELL SOLUTIONS ###################
    # Detach first, then tell
    objs = objs.cpu().detach().numpy()
    measures = measures.cpu().detach().numpy()
    scheduler.tell(objs, measures)
    #######################################################

    # Update the passive elitist archive.
    passive_archive.add(sols, objs, measures)

    # Log iteration
    writer.log_iter(itr=itr,
                    sols=sols,
                    objs=objs,
                    gen_model=discriminator.gen_model,
                    best=best,
                    best_gen=best_gen,
                    passive_archive=passive_archive)
    return best
    

def start_generation(itrs: int,
                     algorithm: str,
                     scheduler: Scheduler,
                     passive_archive: GridArchive,
                     discriminator: Discriminator,
                     writer: Writer) -> None:
    """
    Starts the data generation with QDGS.

    Args:
        itrs (int): The number of iterations to run.
        algorithm (str): The name of the quality-diversity algorithm.
        scheduler (Scheduler): The quality-diversity scheduler.
        passive_archive (Union[GridArchive, CVTArchive]): The passive elitist archive.
        discriminator (Discriminator): The discriminator model.
        writer (Writer): The writer for logging.

    Returns:
        None
    """
    best = -1000

    print(f"> Initialized algorithm: {bcolors.OKGREEN}{algorithm}{bcolors.ENDC}")
    # Begin optimizations
    with alive_bar(itrs) as progress:
        for itr in range(1, itrs + 1):
            best = compute_iter(itr=itr,
                                best=best,
                                algorithm=algorithm,
                                scheduler=scheduler,
                                passive_archive=passive_archive,
                                discriminator=discriminator,
                                writer=writer)
            progress()


def generate_data(task: str,
                  outdir: str = 'logs',
                  log_freq: int = 1,
                  log_arch_freq: int = 5000,
                  image_monitor: bool = False,
                  image_monitor_freq: int = 100,
                  seed: int = None) -> None:
    """Generates data and writes to the respective task directory.

    Args:
        task (str): Which task to run. Options are {'shapes', 'facial_recognition'}.
        outdir (str): QDGS logging directory.
        log_freq (int): Number of iterations between computing QD metrics and updating logs.
        log_arch_freq (int): Number of iterations between saving an archive and generating heatmaps.
        image_monitor (bool): Flags if images should be saved every few iterations.
        image_monitor_freq (int): Number of iterations between saving images.
        seed (int): Seed for the algorithm. By default, there is no seed.

    Returns:
        None
    """
    # Parse args
    assert task in ["shapes", "facial_recognition"]
    assert image_monitor_freq >= 1, "image_monitor_freq must be larger or equal to 1"

    # Initialization
    device = init_device()

    if task == 'shapes':
        itrs = 70000
        objective_prompts = (
            "A regular square or a triangle.", 
            "Splatters of colors."
        )
        measure_prompts = [
            ("A red shape.", "A blue shape."),
            ("A triangle with 3 edges.", "A square or diamond with 4 edges.")
        ]
        grid_dims = [50, 50]
        measure_bins_list = itertools.product([(-100, 100)], 
                                              [(-100, -0.01), (0, 100)])
    elif task == 'facial_recognition':
        itrs = 60000
        objective_prompts = (
            "A detailed photo of an individual with diverse features.",
            "An obscure, fake, or discolored photo of a person."
        )
        measure_prompts = [
            ("A person with dark skin.", "A person with light skin."),
            ("A masculine person.", "A feminine person."),
            ("A person with long hair.", "A person with short hair."),
            ("A person in their 20s.", "A person in their 50s.")
        ]
        grid_dims = [50, 25, 5, 25]
        measure_bins_list = itertools.product([(-100, -0.03), (-0.03, 0.03), (0.03, 100)], 
                                              [(-100, -0.03), (-0.03, 0.03), (0.03, 100)], 
                                              [(-100, 100)],
                                              [(-100, 100)])
    else:
        assert False, f"Task '{task}' not defined"
    
    algorithm = 'cma_maega'
    measure_dim = len(measure_prompts)

    # Initialize models
    gen_model = Generator(task=task, device=device)
    clip_model = CLIP(device=device)
    
    # Initialize quality-diversity
    discriminator = Discriminator(task=task,
                                objective_prompts=objective_prompts,
                                measure_prompts=measure_prompts,
                                gen_model=gen_model, 
                                clip_model=clip_model,
                                device=device
                                )
    scheduler, passive_archive = create_scheduler(algorithm=algorithm, 
                                                  discriminator=discriminator,
                                                  grid_dims=grid_dims,
                                                  batch_size=4, 
                                                  seed=seed)
    writer = Writer(task=task, 
                    algorithm=algorithm, 
                    measure_dim=measure_dim,
                    itrs=itrs,
                    outdir=outdir,
                    log_freq=log_freq,
                    log_arch_freq=log_arch_freq,
                    image_monitor=image_monitor,
                    image_monitor_freq=image_monitor_freq)

    # Start running experiment
    start_generation(itrs=itrs,
                     algorithm=algorithm,
                     scheduler=scheduler,
                     passive_archive=passive_archive,
                     discriminator=discriminator,
                     writer=writer)
    
    # Write dataset to respective task directory.
    archive_df = passive_archive.as_pandas(include_solutions=True).reset_index(drop=True)
    outdir_path = writer.mkdr_recursive([task, 'data', writer.trial_str])
    with torch.no_grad():
        write_data(outdir=outdir_path,
                task=task,
                measure_bins_list=measure_bins_list,
                archive_df=archive_df,
                gen_model=gen_model,
                device=device)

     
if __name__ == '__main__':
    fire.Fire(generate_data)
