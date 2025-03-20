import unittest
from src.forward import ez_forward
from src.inverse import ez_inverse
from src.simulate import simulate_ez_diffusion
from src.utils import compute_bias, compute_squared_error, mean, variance

class TestEZDiffusion(unittest.TestCase):
    
    def test_forward_equations(self):
        """Test that the EZ diffusion forward equations produce expected values."""
        a, v, t = 1.0, 1.5, 0.3  # Sample parameters
        mean_RT, var_RT, accuracy = ez_forward(a, v, t)

        # Ensure values are within reasonable ranges
        self.assertGreater(mean_RT, 0)  # Reaction times must be positive
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)  # Accuracy must be between 0 and 1
        self.assertGreater(var_RT, 0)  # Variance must be positive

    def test_inverse_equations(self):
        """Test that the EZ diffusion inverse equations correctly recover parameters."""
        true_a, true_v, true_t = 1.2, 1.0, 0.25
        mean_RT, var_RT, accuracy = ez_forward(true_a, true_v, true_t)

        a_est, v_est, t_est = ez_inverse(mean_RT, var_RT, accuracy)

        self.assertAlmostEqual(a_est, true_a, delta=0.2)
        self.assertAlmostEqual(v_est, true_v, delta=0.2)
        self.assertAlmostEqual(t_est, true_t, delta=0.1)

    def test_simulation(self):
        """Test that simulated data is reasonable."""
        a, v, t = 1.0, 1.5, 0.3
        RTs, accuracy = simulate_ez_diffusion(a, v, t, N=100)

        self.assertTrue(all(rt > 0 for rt in RTs))  # RTs must be positive
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)

    def test_bias_computation(self):
        """Test the bias computation function."""
        true_values = (1.5, 1.2, 0.3)
        estimated_values = (1.4, 1.1, 0.25)
        
        bias = compute_bias(true_values, estimated_values)
        
        self.assertAlmostEqual(bias[0], 0.1, delta=0.01)
        self.assertAlmostEqual(bias[1], 0.1, delta=0.01)
        self.assertAlmostEqual(bias[2], 0.05, delta=0.01)

    def test_squared_error(self):
        """Test the squared error function."""
        true_values = (1.5, 1.2, 0.3)
        estimated_values = (1.4, 1.1, 0.25)
        
        squared_error = compute_squared_error(true_values, estimated_values)
        
        expected_error = (0.1**2 + 0.1**2 + 0.05**2) / 3
        self.assertAlmostEqual(squared_error, expected_error, delta=0.01)

    def test_mean_function(self):
        """Test the mean function."""
        values = [1, 2, 3, 4, 5]
        self.assertAlmostEqual(mean(values), 3.0, delta=0.01)

    def test_variance_function(self):
        """Test the variance function."""
        values = [1, 2, 3, 4, 5]
        expected_variance = sum((x - 3.0) ** 2 for x in values) / 4  # Sample variance
        self.assertAlmostEqual(variance(values), expected_variance, delta=0.01)

if __name__ == "__main__":
    unittest.main()

#Completed with the help of AI