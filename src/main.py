import random
import os
from simulate import simulate_ez_diffusion
from inverse import recover_parameters  # Correct function name

# Define parameter ranges
A_RANGE = (0.5, 2.0)
V_RANGE = (0.5, 2.0)
T_RANGE = (0.1, 0.5)

# Number of simulations per N
ITERATIONS = 1000

# Different values of N
N_VALUES = [10, 40, 4000]

# Prepare results folder
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_simulation():
    """Runs the simulate-and-recover experiment."""
    for N in N_VALUES:
        biases = []
        squared_errors = []
        
        for _ in range(ITERATIONS):
            # Step 1: Randomly generate true parameters
            a_true = random.uniform(*A_RANGE)
            v_true = random.uniform(*V_RANGE)
            t_true = random.uniform(*T_RANGE)

            # Step 2: Simulate reaction times and accuracy
            RTs, accuracy = simulate_ez_diffusion(a_true, v_true, t_true, N)

            # Step 3: Compute observed statistics
            RT_mean = sum(RTs) / len(RTs)
            RT_var = sum((rt - RT_mean) ** 2 for rt in RTs) / len(RTs)

            # Step 4: Recover parameters
            a_est, v_est, t_est = recover_parameters(RT_mean, RT_var, accuracy)  # Correct function call

            # Step 5: Compute bias and squared error
            bias_a = a_true - a_est
            bias_v = v_true - v_est
            bias_t = t_true - t_est
            squared_error = (bias_a ** 2 + bias_v ** 2 + bias_t ** 2) / 3

            biases.append((bias_a, bias_v, bias_t))
            squared_errors.append(squared_error)

        # Step 6: Store results in a file
        result_file = os.path.join(RESULTS_DIR, f"results_N{N}.txt")
        with open(result_file, "w") as f:
            f.write(f"Average Bias (a, v, t): {sum(b[0] for b in biases) / ITERATIONS}, "
                    f"{sum(b[1] for b in biases) / ITERATIONS}, "
                    f"{sum(b[2] for b in biases) / ITERATIONS}\n")
            f.write(f"Average Squared Error: {sum(squared_errors) / ITERATIONS}\n")

        print(f"Completed N={N} simulations. Results saved in {result_file}")

if __name__ == "__main__":
    run_simulation()

#Developed with the help of AI