from .pricing import black_scholes_put, black_scholes_call, validate_inputs
from .greeks import vega
import numpy as np
import numbers
import math
from scipy.optimize import brentq

def calculate_price_error(market_price, theoretical_price):
    return theoretical_price - market_price

def newton_raphson_step(
        current_vol,
        market_price,
        theoretical_price,
        option_vega,
        min_vega=1e-12
):
    if abs(option_vega) < min_vega:
        raise ValueError(
            "Vega is too small for a stable Newton-Raphson Update."
        )
    return (
        current_vol - (theoretical_price - market_price) / option_vega
    )

def newton_raphson_iteration(
        option_type,
        S,
        K,
        T,
        r,
        current_vol,
        market_price,
        q=0.0
):
    tol = 1e-8
    max_iter = 100

    if option_type not in ("call", "put"):
        raise ValueError(
            "option_type must be either 'call' or 'put'."
        )

    if current_vol <= 0:
        raise ValueError(
            "Initial volatility must be positive."
        )
    

    for i in range(max_iter):
        if option_type == "call":
            theoretical_price = black_scholes_call(S, K, T, r, current_vol, q)
        else:
            theoretical_price = black_scholes_put(S, K, T, r, current_vol, q)
        
        option_vega = vega(S, K, T, r, current_vol, q)
        price_error = calculate_price_error(market_price, theoretical_price)

        if abs(price_error) < tol:
            return current_vol

        next_vol = newton_raphson_step(current_vol, market_price, theoretical_price, option_vega)

        if not np.isfinite(next_vol) or next_vol <= 0:
            raise ValueError(
                "Newton-Raphson produced an invalid volatility estimate."
            )
        
        current_vol = next_vol

    raise RuntimeError(
        "Newton-Raphson did not converge within the maximum iterations."
    )

def validate_market_price(
    option_type,
    market_price,
    S,
    K,
    T,
    r,
    q=0.0
):

    if option_type not in ("call", "put"):
        raise ValueError(
            "option_type must be either 'call' or 'put'."
        )


    if isinstance(market_price, bool) or not isinstance(market_price, numbers.Real):
        raise TypeError("Market price must be numeric.")

    if not math.isfinite(market_price):
        raise ValueError("Market price must be finite.")

    if market_price < 0:
        raise ValueError("Market price cannot be negative.")


    validate_inputs(S, K, T, r, 1.0, q)

    if T == 0:
        raise ValueError(
            "Implied volatility is not identifiable at expiration."
        )

  
    discounted_stock = S * np.exp(-q * T)
    discounted_strike = K * np.exp(-r * T)

    if option_type == "call":
        lower_bound = max(
            discounted_stock - discounted_strike,
            0
        )
        upper_bound = discounted_stock

    else:
        lower_bound = max(
            discounted_strike - discounted_stock,
            0
        )
        upper_bound = discounted_strike

  
    if market_price <= lower_bound:
        raise ValueError(
            f"Market price must be greater than the lower bound "
            f"({lower_bound:.10f}) for a finite positive implied volatility."
        )

    if market_price >= upper_bound:
        raise ValueError(
            f"Market price must be less than the upper bound "
            f"({upper_bound:.10f}) for a finite implied volatility."
        )

    return {
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "market_price": market_price,
        "is_valid": True
    }

def implied_volatility(
    option_type,
    S,
    K,
    T,
    r,
    market_price,
    q=0.0,
    initial_vol=0.20
):
   
    validate_market_price(
        option_type,
        market_price,
        S,
        K,
        T,
        r,
        q
    )


    if not np.isfinite(initial_vol) or initial_vol <= 0:
        raise ValueError(
            "Initial volatility must be positive and finite."
        )


    return newton_raphson_iteration(
        option_type,
        S,
        K,
        T,
        r,
        initial_vol,
        market_price,
        q
    )

def implied_volatility_brent(
    option_type,
    S,
    K,
    T,
    r,
    market_price,
    q=0.0,
    min_vol=1e-6,
    max_vol=5.0
):

    validate_market_price(
        option_type,
        market_price,
        S,
        K,
        T,
        r,
        q
    )

    if min_vol <= 0 or max_vol <= min_vol:
        raise ValueError(
            "Invalid volatility search interval."
        )


    if option_type == "call":
        pricing_function = black_scholes_call
    elif option_type == "put":
        pricing_function = black_scholes_put
    else:
        raise ValueError(
            "option_type must be either 'call' or 'put'."
        )


    def objective_function(vol):
        theoretical_price = pricing_function(
            S, K, T, r, vol, q
        )

        return theoretical_price - market_price


    f_min = objective_function(min_vol)
    f_max = objective_function(max_vol)

    if f_min * f_max > 0:
        raise ValueError(
            "The selected volatility interval does not bracket a solution."
        )


    return brentq(
        objective_function,
        min_vol,
        max_vol,
        xtol=1e-12,
        maxiter=100
    )