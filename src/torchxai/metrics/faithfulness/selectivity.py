import numpy as np
import torch


def selectivity(descending_perturbation_fwds: torch.Tensor) -> np.ndarray | float:
    from scipy.integrate import simpson

    return simpson(
        descending_perturbation_fwds.cpu().numpy(),
        x=np.arange(0, descending_perturbation_fwds.shape[0])
        / (descending_perturbation_fwds.shape[0] - 1),
    )
