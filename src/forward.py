import math

def compute_mean_rt(a, v, t):
    """Computes mean reaction time (RT) using EZ diffusion model."""
    if v == 0:
        raise ValueError("Drift rate (v) cannot be zero.")
    return (a / v) + t

def compute_variance_rt(a, v):
    """Computes reaction time variance using EZ diffusion model."""
    if v == 0:
        raise ValueError("Drift rate (v) cannot be zero.")
    return (a ** 2) / (v ** 2)

def compute_accuracy(a, v):
    """Computes accuracy using EZ diffusion model."""
    return 1 / (1 + math.exp(-2 * v * a))

#Developed with the help of AI