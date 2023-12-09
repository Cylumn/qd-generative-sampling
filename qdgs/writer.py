from typing import List, Union, Tuple

import os
import csv
from pathlib import Path

import torch
import numpy as np


from .utils import bcolors
from .utils import tensor_to_pil_img

import matplotlib.pyplot as plt
from datetime import datetime
from ribs.archives import CVTArchive, GridArchive


class Writer():
    def __init__(self,
                 task: str,
                 algorithm: str,
                 measure_dim: int,
                 itrs: int,
                 outdir: str,
                 log_freq: int,
                 log_arch_freq: int,
                 image_monitor: bool,
                 image_monitor_freq: int):
        # Logging preferences
        self.measure_dim = measure_dim
        self.itrs = itrs
        self.log_freq = log_freq
        self.log_arch_freq = log_arch_freq
        self.image_monitor = image_monitor
        self.image_monitor_freq = image_monitor_freq

        # Open and initialize directories
        trial_str, logdir_str, gen_logdir_str = self.open(task, outdir, algorithm)
        self.trial_str = trial_str
        self.logdir_str = logdir_str
        self.gen_logdir_str = gen_logdir_str
        print(f"> Created logdir: {bcolors.OKCYAN}{self.logdir_str}{bcolors.ENDC}")

        # Create writer
        columns = ['Iteration', 'QD-Score', 'Coverage', 'Maximum Objective', 'Average Objective']
        s_summary_file = self.open_summary(columns)
        self.s_summary_file = s_summary_file

    def mkdr_recursive(self,
                        dir_list):
        cur_path = ""
        for dir in dir_list:
            cur_path = os.path.join(cur_path, dir)
            path = Path(cur_path)
            if not path.is_dir():
                path.mkdir()

        return cur_path

    def open(self,
             task: str,
             outdir: str,
             algorithm: str) -> Tuple[str, str]:
        # Create a shared logging directory for the experiments for this algorithm.
        trial_str = f"trial_{datetime.now().strftime('%m.%d_%I:%M:%S_%p')}"
        logdir_str = self.mkdr_recursive([task, outdir, algorithm, trial_str])

        # Create a directory for logging intermediate images if the monitor is on.
        gen_logdir_str = None
        if self.image_monitor:
            gen_logdir_str = os.path.join(logdir_str, 'generations')
            logdir = Path(gen_logdir_str)
            if not logdir.is_dir():
                logdir.mkdir()

        return trial_str, logdir_str, gen_logdir_str
    

    def open_summary(self,
                     columns: List[str]) -> csv.writer:
        # Create a new summary file
        s_summary_file = os.path.join(self.logdir_str, f"summary.csv")
        if os.path.exists(s_summary_file):
            os.remove(s_summary_file)
        with open(s_summary_file, 'w') as summary_file:
            writer = csv.writer(summary_file)
            writer.writerow(columns)

        return s_summary_file


    def _log_image(self,
                   itr: int,
                   sols: torch.tensor,
                   objs: np.array,
                   gen_model) -> None:
        best_index = np.argmax(objs)
        latent_code = sols[best_index]

        img, _ = gen_model.synthesis(latent_code)
        img = tensor_to_pil_img(img)
        img.save(os.path.join(self.gen_logdir_str, f'{itr}.png'))


    def _save_heatmap(self,
                      archive: Union[GridArchive, CVTArchive], 
                      x_measure_dim: int,
                      y_measure_dim: int,
                      heatmap_path: str) -> None:
        """Saves a heatmap of the archive to the given path."""
        plt.figure(figsize=(8, 6))
        grid_archive_heatmap(archive, x_measure_dim=x_measure_dim, y_measure_dim=y_measure_dim, vmin=0, vmax=100, cmap="viridis")
        plt.tight_layout()
        plt.savefig(heatmap_path)
        plt.close(plt.gcf())


    def log_iter(self,
                 itr: int,
                 sols: np.array,
                 objs: np.array,
                 gen_model,
                 best: Tuple[int],
                 best_gen: Tuple[int],
                 passive_archive: Union[GridArchive, CVTArchive]) -> None:
        print(f'Overall best: {best} | Iteration best: {best_gen}')

        # Log image
        if self.image_monitor and itr % self.image_monitor_freq == 0:
            self._log_image(itr=itr,
                            sols=sols,
                            objs=objs,
                            gen_model=gen_model)

        # Save the archive at the given frequency.
        # Always save on the final iteration.
        final_itr = itr == self.itrs
        if (itr > 0 and itr % self.log_arch_freq == 0) or final_itr:
            # Save a full archive for analysis.
            df = passive_archive.as_pandas(include_solutions = final_itr)
            df.to_pickle(os.path.join(self.logdir_str, f"archive_{itr:08d}.pkl"))

            # Save a heatmap image to observe how the trial is doing.
            for i in range(self.measure_dim):
                for j in range(self.measure_dim):
                    if i >= j:
                        continue
                    self._save_heatmap(passive_archive, 
                                       x_measure_dim=i,
                                       y_measure_dim=j,
                                       heatmap_path=os.path.join(self.logdir_str, f"heatmap[{i},{j}]_{itr:08d}.png"))

        # Update the summary statistics for the archive
        if (itr > 0 and itr % self.log_freq == 0) or final_itr:
            with open(self.s_summary_file, 'a') as summary_file:
                writer = csv.writer(summary_file)

                sum_obj = 0
                num_filled = 0
                num_cells = passive_archive.cells
                for elite in passive_archive:
                    num_filled += 1
                    sum_obj += elite.objective
                qd_score = sum_obj / num_cells
                average_obj = sum_obj / num_filled
                coverage = 100.0 * num_filled / num_cells
                data = [itr, qd_score, coverage, best, average_obj]
                writer.writerow(data)



#################### RIBS.VISUALIZE ############################
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
def grid_archive_heatmap(archive,
                         x_measure_dim,
                         y_measure_dim,
                         ax=None,
                         *,
                         transpose_measures=False,
                         cmap="magma",
                         aspect=None,
                         vmin=None,
                         vmax=None,
                         cbar="auto",
                         pcm_kwargs=None,
                         cbar_kwargs=None):
    """Plots heatmap of a :class:`~ribs.archives.GridArchive` with 1D or 2D
    measure space.

    This function creates a grid of cells and shades each cell with a color
    corresponding to the objective value of that cell's elite. This function
    uses :func:`~matplotlib.pyplot.pcolormesh` to generate the grid. For further
    customization, pass extra kwargs to :func:`~matplotlib.pyplot.pcolormesh`
    through the ``pcm_kwargs`` parameter. For instance, to create black
    boundaries of width 0.1, pass in ``pcm_kwargs={"edgecolor": "black",
    "linewidth": 0.1}``.

    Args:
        archive (GridArchive): A 2D :class:`~ribs.archives.GridArchive`.
        ax (matplotlib.axes.Axes): Axes on which to plot the heatmap.
            If ``None``, the current axis will be used.
        transpose_measures (bool): By default, the first measure in the archive
            will appear along the x-axis, and the second will be along the
            y-axis. To switch this behavior (i.e. to transpose the axes), set
            this to ``True``.
        cmap (str, list, matplotlib.colors.Colormap): The colormap to use when
            plotting intensity. Either the name of a
            :class:`~matplotlib.colors.Colormap`, a list of RGB or RGBA colors
            (i.e. an :math:`N \\times 3` or :math:`N \\times 4` array), or a
            :class:`~matplotlib.colors.Colormap` object.
        aspect ('auto', 'equal', float): The aspect ratio of the heatmap (i.e.
            height/width). Defaults to ``'auto'`` for 2D and ``0.5`` for 1D.
            ``'equal'`` is the same as ``aspect=1``.
        vmin (float): Minimum objective value to use in the plot. If ``None``,
            the minimum objective value in the archive is used.
        vmax (float): Maximum objective value to use in the plot. If ``None``,
            the maximum objective value in the archive is used.
        cbar ('auto', None, matplotlib.axes.Axes): By default, this is set to
            ``'auto'`` which displays the colorbar on the archive's current
            :class:`~matplotlib.axes.Axes`. If ``None``, then colorbar is not
            displayed. If this is an :class:`~matplotlib.axes.Axes`, displays
            the colorbar on the specified Axes.
        pcm_kwargs (dict): Additional kwargs to pass to
            :func:`~matplotlib.pyplot.pcolormesh`.
        cbar_kwargs (dict): Additional kwargs to pass to
            :func:`~matplotlib.pyplot.colorbar`.

    Raises:
        ValueError: The archive's dimension must be 1D or 2D.
    """
    if aspect is None:
        # Handles default aspects for different dims.
        aspect = "auto"

    # Try getting the colormap early in case it fails.
    cmap = matplotlib.cm.get_cmap(cmap)

    # Useful to have these data available.
    df = archive.as_pandas()
    objective_batch = df.objective_batch()

    # Retrieve data from archive.
    lower_bounds = archive.lower_bounds
    upper_bounds = archive.upper_bounds
    x_dim = archive.dims[x_measure_dim]
    y_dim = archive.dims[y_measure_dim]
    x_bounds = archive.boundaries[x_measure_dim]
    y_bounds = archive.boundaries[y_measure_dim]

    # Color for each cell in the heatmap.
    colors = np.full((y_dim, x_dim), np.nan)
    grid_index_batch = archive.int_to_grid_index(df.index_batch())
    colors[grid_index_batch[:, y_measure_dim], grid_index_batch[:, x_measure_dim]] = objective_batch

    # Initialize the axis.
    ax = plt.gca() if ax is None else ax
    ax.set_xlim(lower_bounds[x_measure_dim], upper_bounds[x_measure_dim])
    ax.set_ylim(lower_bounds[y_measure_dim], upper_bounds[y_measure_dim])

    ax.set_aspect(aspect)

    # Create the plot.
    pcm_kwargs = {} if pcm_kwargs is None else pcm_kwargs
    vmin = np.min(objective_batch) if vmin is None else vmin
    vmax = np.max(objective_batch) if vmax is None else vmax
    t = ax.pcolormesh(x_bounds,
                        y_bounds,
                        colors,
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        **pcm_kwargs)

    # Create color bar.
    cbar_kwargs = {} if cbar_kwargs is None else cbar_kwargs
    ax.figure.colorbar(t, ax=ax, **cbar_kwargs)
