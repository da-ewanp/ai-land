from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def create_two_slope_norm(data):
    # Calculate 5th and 95th percentiles
    vmin, vmax = np.percentile(data, [5, 95])
    # Determine the center based on data distribution
    if vmin >= 0:  # All positive data
        vcenter = vmin + (vmax - vmin) / 3
    elif vmax <= 0:  # All negative data
        vcenter = vmax - (vmin - vmax) / 3
    else:  # Data spans positive and negative
        vcenter = 0
    # Create the TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    return norm


def make_map_val_plot(
    input,
    targ_arr,
    pred_arr,
    targ_diag_arr,
    pred_diag_arr,
    lat_arr,
    lon_arr,
    name_lst,
    diag_name_lst,
):
    rows = len(name_lst) + len(diag_name_lst)
    fig, axes = plt.subplots(
        ncols=5,
        nrows=rows,
        figsize=(24, 4 * rows),
    )
    for i, var in enumerate(name_lst):
        targ_inc = targ_arr[:, i] - input[:, i]
        pred_inc = pred_arr[:, i] - input[:, i]
        c = axes[i, 0].scatter(
            lon_arr,
            lat_arr,
            c=input[:, i],
            s=1,
        )
        plt.colorbar(c, ax=axes[i, 0])
        axes[i, 0].set_title(f"Input {var}")

        c = axes[i, 1].scatter(
            lon_arr,
            lat_arr,
            c=targ_inc,
            cmap="RdBu",
            # vmin=np.quantile(targ_inc, 0.05),
            # vmax=np.quantile(targ_inc, 0.95),
            norm=create_two_slope_norm(targ_inc),
            s=1,
        )
        plt.colorbar(c, ax=axes[i, 1])
        axes[i, 1].set_title(f"Target inc {var}")

        c = axes[i, 2].scatter(
            lon_arr,
            lat_arr,
            c=pred_inc,
            cmap="RdBu",
            # vmin=np.quantile(targ_inc, 0.05),
            # vmax=np.quantile(targ_inc, 0.95),
            norm=create_two_slope_norm(targ_inc),
            s=1,
        )
        plt.colorbar(c, ax=axes[i, 2])
        axes[i, 2].set_title(f"Prediction inc {var}")

        c = axes[i, 3].scatter(
            lon_arr,
            lat_arr,
            c=np.abs(targ_inc - pred_inc),
            vmin=0,
            vmax=np.quantile(np.abs(targ_inc - pred_inc), 0.95),
            s=1,
        )
        plt.colorbar(c, ax=axes[i, 3])
        axes[i, 3].set_title(f"MAE inc {var}")

        mape = (targ_arr[:, i] - pred_arr[:, i]) / targ_arr[:, i]
        c = axes[i, 4].scatter(
            lon_arr,
            lat_arr,
            c=mape,
            # vmin=np.quantile(mape, 0.05),
            # vmax=np.quantile(mape, 0.95),
            cmap="RdBu",
            norm=create_two_slope_norm(mape),
            s=1,
        )
        plt.colorbar(c, ax=axes[i, 4])
        axes[i, 4].set_title(f"MAPE {var}")

    for i, var in enumerate(diag_name_lst):
        axes[i + len(name_lst), 0].set_axis_off()
        c = axes[i + len(name_lst), 1].scatter(
            lon_arr,
            lat_arr,
            c=targ_diag_arr[:, i],
            s=1,
        )
        plt.colorbar(c, ax=axes[i + len(name_lst), 1])
        axes[i + len(name_lst), 1].set_title(f"Target {var}")

        c = axes[i + len(name_lst), 2].scatter(
            lon_arr,
            lat_arr,
            c=pred_diag_arr[:, i],
            s=1,
        )
        plt.colorbar(c, ax=axes[i + len(name_lst), 2])
        axes[i + len(name_lst), 2].set_title(f"Prediction {var}")

        c = axes[i + len(name_lst), 3].scatter(
            lon_arr,
            lat_arr,
            c=np.abs(targ_diag_arr[:, i] - pred_diag_arr[:, i]),
            vmin=0,
            vmax=np.quantile(np.abs(targ_diag_arr[:, i] - pred_diag_arr[:, i]), 0.95),
            s=1,
        )
        plt.colorbar(c, ax=axes[i + len(name_lst), 3])
        axes[i + len(name_lst), 3].set_title(f"MAE {var}")

        mape = (targ_diag_arr[:, i] - pred_diag_arr[:, i]) / targ_diag_arr[:, i]
        c = axes[i + len(name_lst), 4].scatter(
            lon_arr,
            lat_arr,
            c=mape,
            # vmin=np.quantile(mape, 0.05),
            # vmax=np.quantile(mape, 0.95),
            norm=create_two_slope_norm(mape),
            cmap="RdBu",
            s=1,
        )
        plt.colorbar(c, ax=axes[i + len(name_lst), 4])
        axes[i + len(name_lst), 4].set_title(f"MAPE {var}")

    fig.tight_layout()
    return fig


def r2_score_multi(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculated the r-squared score between 2 arrays of values.

    :param y_pred: predicted array :param y_true: "truth" array :return: r-squared
    metric
    """
    return r2_score(y_pred.flatten(), y_true.flatten())
