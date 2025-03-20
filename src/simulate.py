import math
import random

def simulate_ez_diffusion(a, v, t, N):
    """
    Simulate N response times and accuracy based on the EZ diffusion model.

    Parameters:
        a (float): Boundary separation
        v (float): Drift rate
        t (float): Nondecision time
        N (int): Number of trials to simulate

    Returns:
        tuple: (RTs, accuracy) - List of response times and overall accuracy
    """

    # Ensure parameters are valid
    if a <= 0 or v <= 0 or t < 0 or N <= 0:
        raise ValueError("Parameters must be positive, and N must be a positive integer.")

    RTs = []  # List to store reaction times
    correct_responses = 0  # Track number of correct responses

    for _ in range(N):
        # Generate a random reaction time using a normal distribution
        # Mean RT = a/v, variance approximated as 0.1 for simplicity
        RT = random.gauss(mu=(a / v), sigma=0.1) + t  # Add nondecision time
        RT = max(RT, 0.001)  # Ensure RT is positive

        RTs.append(RT)

        # Compute probability of a correct response
        prob_correct = 1 / (1 + math.exp(-2 * v * a))
        if random.random() < prob_correct:
            correct_responses += 1  # Count as correct response

    # Compute accuracy as proportion of correct responses
    accuracy = correct_responses / N

    return RTs, accuracy

#Developed with the help of AI