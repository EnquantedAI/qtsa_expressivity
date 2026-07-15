# Support functions for data generation
# Author: Jacob Cybulski, ironfrown[at]gmail.com
# Date: 2026

import numpy as np


def mackey_glass(beta=0.17, gamma=0.1, n=10, tau=30, dt=1.5, T=2000):
    """Generate a Mackey-Glass time series."""
    N = int(T / dt)
    delay_steps = int(tau / dt)
    x = np.zeros(N + delay_steps)
    x[0:delay_steps] = 1.2

    for t in range(delay_steps - 1, N + delay_steps - 1):
        x_tau = x[t - delay_steps]
        dxdt = (beta * x_tau / (1 + x_tau**n)) - (gamma * x[t])
        x[t+1] = x[t] + dxdt * dt

    d_min, d_max = np.min(x), np.max(x)
    d = (x - d_min) / (d_max - d_min)

    return d[delay_steps:]

def mackey_glass_jaeger(beta=0.2, gamma=0.1, n=10, tau=30, dt=1, T=3000):
    time_series_mg17_jaeger = mackey_glass(tau=tau, beta=beta, gamma=gamma, n=n, dt=dt, T=T)
    time_series_mg17_jaeger = time_series_mg17_jaeger[0:1333]
    return time_series_mg17_jaeger

def split_time_series(time_series, train_fraction):
    """Split a series into train and test parts."""
    split_point = int(len(time_series) * train_fraction)
    time_series = np.asarray(time_series, dtype=float)
    return time_series[:split_point], time_series[split_point:]


def fit_minmax_scaler(train_series):
    """Fit min-max statistics on the training split."""
    train_series = np.asarray(train_series, dtype=float)
    return {
        "min": float(np.min(train_series)),
        "max": float(np.max(train_series)),
    }


def transform_with_minmax_scaler(time_series, scaler):
    """Apply a fitted min-max scaler to a series."""
    time_series = np.asarray(time_series, dtype=float)
    scale_range = scaler["max"] - scaler["min"]
    if scale_range <= 0:
        return np.zeros_like(time_series, dtype=float)
    return (time_series - scaler["min"]) / scale_range


def split_and_scale_series(time_series, train_fraction):
    """Split a series and scale both parts from the train split."""
    train_series, test_series = split_time_series(time_series, train_fraction)
    scaler = fit_minmax_scaler(train_series)
    return (
        transform_with_minmax_scaler(train_series, scaler),
        transform_with_minmax_scaler(test_series, scaler),
        scaler,
    )


def create_io_pairs(data, window_size, lag=0):
    """Build input-output pairs from a time series."""
    inputs, outputs = [], []
    for i in range(len(data) - window_size - lag):
        input_window = data[i : i + window_size]
        output_point = data[i + window_size + lag]
        inputs.append(input_window)
        outputs.append(output_point)

    return np.array(inputs), np.array(outputs)


def generate_arma_data(n_points=1333, ar_coeffs=[1, -0.7], ma_coeffs=[1, 0.5, -0.3], seed=42):
    """Generate an ARMA time series."""
    from scipy.signal import lfilter

    np.random.seed(seed)
    noise = np.random.normal(0, 1, n_points)
    data = lfilter(ma_coeffs, ar_coeffs, noise)

    d_min, d_max = np.min(data), np.max(data)
    d = (data - d_min) / (d_max - d_min)
    
    return d


def generate_narma_data(n_points=1333, order=10, alpha=0.3, beta=0.05, gamma=1.5, delta=0.1, seed=42):
    """Generate a NARMA time series."""
    np.random.seed(seed)
    s = np.random.uniform(0, 0.5, n_points)
    y = np.zeros(n_points)

    for k in range(order, n_points):
        sum_term = np.sum(y[k-order:k])
        y[k] = (alpha * y[k-1] +
                beta * y[k-1] * sum_term +
                gamma * s[k-order] * s[k] +
                delta)

    d_min, d_max = np.min(y), np.max(y)
    d = (y - d_min) / (d_max - d_min)
    
    return d

def generate_narma_sequence(length, u, alpha=0.3, beta=0.05, gamma=1.5, delta=0.1, n=10, scale_by_n=True):
    """
    Generates a stabilized, double-precision NARMA sequence.
    
    Parameters:
    -----------
    length : int
        The length of the output sequence.
    u : numpy.ndarray
        The input signal (should be bounded, e.g., in [0, 0.5]).
    alpha, beta, gamma, delta : float
        Standard NARMA system parameters.
    n : int
        The system order (lookback window).
    scale_by_n : bool
        If True, scales the beta feedback term by (1/n) to maintain 
        mathematical stability at high orders (n > 10).
    """
    # Force 64-bit double precision floating point representation
    y = np.zeros(length, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    
    # Adjust beta to scale down feedback for high-order structures
    effective_beta = beta / n if scale_by_n else beta
    
    for t in range(n, length):
        # Sum of previous n outputs
        nar_term = np.sum(y[t-n : t])
        
        # Double precision calculation prevents early variable overflow
        y[t] = (alpha * y[t-1] + 
                effective_beta * y[t-1] * nar_term + 
                gamma * u[t-1] * u[t-n] + 
                delta)
                
    return y

