import math
import numbers
import numpy as np
from scipy.stats import norm

def validate_inputs(S, K, T, r, sigma, q):

    parameters = {
        "Stock price": S,
        "Strike price": K,
        "Time to maturity": T,
        "Risk-free rate": r,
        "Volatility": sigma,
        "Dividend yield": q
    }

    for name, value in parameters.items():

        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"{name} must be numeric.")

        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")

    if S <= 0:
        raise ValueError("Stock price must be positive.")
    if K <= 0:
        raise ValueError("Strike price must be positive.")
    if T < 0:
        raise ValueError("Time to maturity cannot be negative.")
    if sigma < 0:
        raise ValueError("Volatility cannot be negative.")

    
def calculate_d1_d2(S, K, T, r, sigma, q=0.0):

    validate_inputs(S, K, T, r, sigma, q)

    if T == 0 or sigma == 0:
        raise ValueError(
            "d1 and d2 are undefined when time or volatility is zero."
        )


    d1 = (np.log(S / K)
        + (r - q + sigma**2 / 2) * T
        ) / (sigma * np.sqrt(T))
    
    d2 = d1 - sigma * np.sqrt(T)

    return d1, d2

def black_scholes_call(S, K, T, r, sigma, q=0.0):

    validate_inputs(S, K, T, r, sigma, q)

    if T == 0 :
        return max(S - K, 0)
    elif sigma == 0 :
        return np.exp(-r * T) * max(S * np.exp((r-q)*T) - K, 0)

    d1, d2 = calculate_d1_d2(S, K, T, r, sigma, q)

    return (
        S * np.exp(-q * T) * norm.cdf(d1)
        - K * np.exp(-r * T) * norm.cdf(d2)
    )

def black_scholes_put(S, K, T, r, sigma, q=0.0):

    validate_inputs(S, K, T, r, sigma, q)

    if T == 0 :
        return max(K - S, 0)
    elif sigma == 0 :
        return np.exp(-r * T) * max(K - S * np.exp((r-q)*T), 0)

    d1, d2 = calculate_d1_d2(S, K, T, r, sigma, q)

    return (
        K * np.exp(-r * T) * norm.cdf(-d2)
        - S * np.exp(-q * T) * norm.cdf(-d1)
    )

def put_call_parity(S, K, T, r, sigma, q=0.0):
    """
    Verify European put-call parity.

    C - P = S * exp(-qT) - K * exp(-rT)

    Returns:
        dict: Call-put difference, theoretical parity value,
              and whether parity holds.
    """

    # Validate inputs
    validate_inputs(S, K, T, r, sigma, q)

    # Calculate call and put prices
    call = black_scholes_call(S, K, T, r, sigma, q)
    put = black_scholes_put(S, K, T, r, sigma, q)

    # Calculate both sides of put-call parity
    lhs = call - put

    rhs = (
        S * np.exp(-q * T)
        - K * np.exp(-r * T)
    )

    # Verify parity
    is_valid = np.isclose(
        lhs,
        rhs,
        rtol=1e-10,
        atol=1e-10
    )

    return {
        "call_minus_put": lhs,
        "parity_value": rhs,
        "is_valid": bool(is_valid)
    }