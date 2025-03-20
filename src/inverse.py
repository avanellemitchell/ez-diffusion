import math

def recover_parameters(RT_mean, RT_var, accuracy):
    """
    Recover EZ diffusion model parameters (a, v, t) from observed statistics.

    Parameters:
        RT_mean (float): Mean response time
        RT_var (float): Variance of response time
        accuracy (float): Proportion of correct responses

    Returns:
        tuple: (a_est, v_est, t_est) - estimated boundary separation, drift rate, and nondecision time
    """

    # Prevent division or log errors
    accuracy = max(min(accuracy, 0.9999), 0.0001)  # Keep within (0,1)
    RT_var = max(RT_var, 1e-6)  # Prevent zero variance issues

    # Compute drift rate (v)
    ln_term = math.log(accuracy / (1 - accuracy))
    v_est = ln_term / RT_mean

    # Compute boundary separation (a)
    a_est = v_est * RT_mean

    # Compute nondecision time (t)
    t_est = RT_mean - (a_est / v_est)

    return a_est, v_est, t_est

#Developed with th help of AI