# Plots trainig trajectories

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from definitions.utils import (
    load_experiments, 
    get_seed_folders,
)

from configs.local_config import EXPERIMENTS_DIR


def plot_trajectories(
    experiments,
    x_min=0,
    x_max=None,
    y_min=0,
    y_max=1,
    plot_source=True,
):

    if plot_source:
        n_plots = 2
    else:
        n_plots = 1

    fig, axes = plt.subplots(
        n_plots,
        1,
        figsize=(8, 2.8 * n_plots),
        sharex=True,
    )


    axes = np.atleast_1d(axes)

    for path, experiment in experiments.items():

        metrics = experiment["metrics"]

        t = metrics["collect_time_stamps"]

        clean_accuracy = metrics["clean_eval_accuracy"]
        source_accuracy = metrics["source_eval_accuracy"]


        if not plot_source:

            axes[0].plot(
                t,
                clean_accuracy,
                linewidth=1.2,
            )


        else:

            axes[0].plot(
                t,
                clean_accuracy,
                linewidth=1.2,
            )

            axes[1].plot(
                t,
                source_accuracy,
                linewidth=1.2,
            )

    if not plot_source:
        axes[0].set_ylabel("Clean accuracy")

    else:
        axes[0].set_ylabel("Clean accuracy")
        axes[1].set_ylabel("Source accuracy")

    axes[-1].set_xlabel("Training step")

    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(alpha=0.2)

    fig.tight_layout()

    return fig, axes



# CNN paths
clean_path_cnn=EXPERIMENTS_DIR/'pc_0_pw_0_sh_1_sw_1'/'cnn_c1_3_c2_3_k1_2_p1_2_k2_2_p2_2_h_4'/'cnn_clean_run'
shortcut_paths_cnn=EXPERIMENTS_DIR/'pc_1_pw_0_sh_1_sw_1'/'cnn_c1_3_c2_3_k1_2_p1_2_k2_2_p2_2_h_4'/'cnn_shortcut_run'

# MLP paths
clean_path_mlp=EXPERIMENTS_DIR/'pc_0_pw_0_sh_1_sw_1'/'mlp_h1_4_h2_2'/'mlp_clean_run'
shortcut_paths_mlp=EXPERIMENTS_DIR/'pc_1_pw_0_sh_1_sw_1'/'mlp_h1_4_h2_2'/'mlp_shortcut_run'

clean_cnn=load_experiments(get_seed_folders(clean_path_cnn))
shortcut_cnn=load_experiments(get_seed_folders(shortcut_paths_cnn))

clean_mlp=load_experiments(get_seed_folders(clean_path_mlp))
shortcut_mlp=load_experiments(get_seed_folders(shortcut_paths_mlp))
   
fig, ax = plot_trajectories(
    clean_cnn,
    x_min=0,
    x_max=12000,
    y_min=0.5,
    y_max=1.02,
    plot_source=False,
)
plt.show()

fig, ax = plot_trajectories(
    clean_mlp,
    x_min=0,
    x_max=12000,
    y_min=0.5,
    y_max=1.02,
    plot_source=False,
)
plt.show()

fig, ax = plot_trajectories(
    shortcut_cnn,
    x_min=0,
    x_max=12000,
    y_min=0.5,
    y_max=1.02,
    plot_source=False,
)
plt.show()

fig, ax = plot_trajectories(
    shortcut_mlp,
    x_min=0,
    x_max=12000,
    y_min=0.5,
    y_max=1.02,
    plot_source=False,
)
plt.show()

# fig.savefig(
#     'figure_name.pdf',
#     bbox_inches='tight',
#     transparent=True,
# )


    
    
