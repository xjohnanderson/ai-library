import numpy as np
from scipy.optimize import curve_fit

def logistic_function(t, beta0, beta1):
    """
    Standard logistic model for AI success probability.
    P(success) = 1 / (1 + exp(-(beta0 + beta1 * ln(t))))
    """
    return 1 / (1 + np.exp(-(beta0 + beta1 * np.log(t))))

def calculate_time_horizon(beta0, beta1):
    """
    Calculates the human task duration (t) where P(success) = 0.5.
    H_m = exp(-beta0 / beta1)
    """
    return np.exp(-beta0 / beta1)

def fit_model_horizon(human_times, success_flags):
    """
    Fits model parameters to empirical task data.
    
    Args:
        human_times (list): Array of human task durations in seconds.
        success_flags (list): Binary outcomes (1 for success, 0 for failure).
        
    Returns:
        float: The calculated 50% success time horizon in seconds.
    """
    # Initial guess for [beta0, beta1]
    p0 = [1.0, -1.0] 
    
    params, _ = curve_fit(logistic_function, human_times, success_flags, p0=p0)
    beta0, beta1 = params
    
    horizon = calculate_time_horizon(beta0, beta1)
    return horizon, (beta0, beta1)

def project_future_capability(current_horizon, years_forward, doubling_time_years=0.58):
    """
    Extrapolates the horizon based on the 7-month (0.58 year) doubling trend.
    """
    return current_horizon * (2 ** (years_forward / doubling_time_years))

# Example usage for testing
if __name__ == "__main__":
    # Mock Data: [time_in_sec], [success_binary]
    # Tasks ranging from 4 mins to 10 hours
    t_data = np.array([240, 600, 1800, 3600, 7200, 14400, 36000])
    s_data = np.array([1, 1, 1, 0, 0, 0, 0]) 
    
    h_m, coeffs = fit_model_horizon(t_data, s_data)
    print(f"Calculated 50% Time Horizon: {h_m:.2f} seconds")
    
    # Project 2 years into the future
    future_h = project_future_capability(h_m, 2)
    print(f"Projected Horizon in 2 years: {future_h / 3600:.2f} human-hours")