import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# Create the colormap
colors = ["#F4C849", "#FF4F19", "#54AC61", "#84C5DD", "#DCB5FF"]
custom_map = LinearSegmentedColormap.from_list("custom", colors, N=len(colors))

start = 0
end = 1
step = 0.01
possible_bins = np.arange(start, end + step, step)


def save_score_histogram(csv_path: Path):
    df = pd.read_csv(csv_path)

    df = df.rename(columns={"object": "objeto"})
    df = df.replace(
        {"bunny": "conejo", "dragon": "dragón", "cube": "cubo", "sphere": "esfera", "cube_sphere": "cubo_esfera"}
    )

    bins = possible_bins[possible_bins >= (df["ssim"].min() - step)].tolist()
    ax = df.pivot(columns="objeto")["ssim"].plot(
        kind="hist",
        stacked=True,
        bins=bins,
        rwidth=0.9,
        colormap=custom_map,
        title="Distribución del conjunto de evaluación\nagrupado por SSIM y por tipo de objeto",
        rot=0 if len(bins) < 12 else 90,
    )
    ax.set_xlabel("SSIM")
    ax.set_ylabel("Cantidad de instancias")
    ax.set_xticks(bins)

    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, top + 50)

    ax.bar_label(ax.containers[-1], label_type="edge", fontsize="small")

    result_filename = f"hist{'_global' if 'global' in csv_path.stem else ''}"

    ax.get_figure().savefig(csv_path.parent / result_filename, dpi=1200, bbox_inches="tight")
    return ax
