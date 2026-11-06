import numpy as np
from scipy.stats import norm
from .pricing import calculate_d1_d2

def delta_call(S, K, T, r, sigma, q=0.0):
    
    d1, _ = calculate_d1_d2(S, K, T, r, sigma, q)

    return (
        np.exp(-q * T) * norm.cdf(d1)
    )

def delta_put(S, K, T, r, sigma, q=0.0):

    d1, _ = calculate_d1_d2(S, K, T, r, sigma, q)
    
    return (
        np.exp(-q * T) * (norm.cdf(d1) - 1)
    )

def gamma(S, K, T, r, sigma, q=0.0):

    d1, _ = calculate_d1_d2(S, K, T, r, sigma, q)

    return (
        np.exp(-q * T) * norm.pdf(d1)
        / (S * sigma * np.sqrt(T))
    )

def theta_call(S, K, T, r, sigma , q=0.0):

    d1, d2 = calculate_d1_d2(S, K, T, r, sigma , q)

    return (
        - S * np.exp(-q * T) * norm.pdf(d1) * sigma
        / (2 * np.sqrt(T)) 
        - r * K * np.exp(-r * T) * norm.cdf(d2)
        + q * S * np.exp(-q * T) * norm.cdf(d1)
    )


def theta_put(S, K, T, r, sigma , q=0.0):
        
    d1, d2 = calculate_d1_d2(S, K, T, r, sigma , q)

    return (
        - S * np.exp(-q * T) * norm.pdf(d1) * sigma
        / (2 * np.sqrt(T)) 
        + r * K * np.exp(-r * T) * norm.cdf(-d2)
        - q * S * np.exp(-q * T) * norm.cdf(-d1)
    )

def vega(S, K, T, r, sigma , q=0.0):

    d1, _ = calculate_d1_d2(S, K, T, r, sigma , q)
    
    return (
        S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    )

def rho_call(S, K, T, r, sigma , q=0.0):

    _, d2 = calculate_d1_d2(S, K, T, r, sigma , q)
    
    return (
        K * T * np.exp(-r * T) * norm.cdf(d2)
    )

def rho_put(S, K, T, r, sigma , q=0.0):

    _, d2 = calculate_d1_d2(S, K, T, r, sigma , q)
        
    return (
        - K * T * np.exp(-r * T) * norm.cdf(-d2)
    )
