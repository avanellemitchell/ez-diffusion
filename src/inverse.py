import math

def recover_parameters(RT_mean, RT_var, accuracy):
    """
    Recover EZ diffusion model parameters (a, v, t) from observed statistics.
    """
    # Prevent division or log errors
    accuracy = max(min(accuracy, 0.9999), 0.0001)  # Keep within (0,1)
    RT_var = max(RT_var, 1e-6)  # Prevent zero variance issues

    # Compute drift rate (v) correctly using variance
    ln_term = math.log(accuracy / (1 - accuracy))
    v_est = ln_term / math.sqrt(RT_var)  # ✅ Use sqrt(RT_var) instead of RT_mean

    # Compute boundary separation (a) correctly
    a_est = math.sqrt(RT_var) * v_est  # ✅ Uses RT variance

    # Compute nondecision time (t)
    t_est = RT_mean - (a_est / v_est)

    return a_est, v_est, t_est


#Developed with th help of AI