import math

def compute_bias(true_values, estimated_values):
    """
    Compute bias as the difference between true and estimated parameters.

    Parameters:
        true_values (tuple): (true_a, true_v, true_t)
        estimated_values (tuple): (est_a, est_v, est_t)

    Returns:
        tuple: Bias values (bias_a, bias_v, bias_t)
    """
    bias_a = true_values[0] - estimated_values[0]
    bias_v = true_values[1] - estimated_values[1]
    bias_t = true_values[2] - estimated_values[2]

    return (bias_a, bias_v, bias_t)

def compute_squared_error(true_values, estimated_values):
    """
    Compute squared error between true and estimated parameters.

    Parameters:
        true_values (tuple): (true_a, true_v, true_t)
        estimated_values (tuple): (est_a, est_v, est_t)

    Returns:
        float: Squared error averaged over the three parameters
    """
    squared_error = (
        (true_values[0] - estimated_values[0]) ** 2 +
        (true_values[1] - estimated_values[1]) ** 2 +
        (true_values[2] - estimated_values[2]) ** 2
    ) / 3

    return squared_error

def mean(values):
    """
    Compute the mean of a list of values.

    Parameters:
        values (list): List of numerical values

    Returns:
        float: Mean value
    """
    return sum(values) / len(values) if values else 0

def variance(values):
    """
    Compute the variance of a list of values.

    Parameters:
        values (list): List of numerical values

    Returns:
        float: Variance
    """
    if len(values) < 2:
        return 0  # Variance is undefined for lists with fewer than 2 values

    mean_value = mean(values)
    return sum((x - mean_value) ** 2 for x in values) / (len(values) - 1)

#Developed with the help of AI