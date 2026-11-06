from black_scholes import (
    black_scholes_call,
    black_scholes_put,
    put_call_parity,
    delta_call,
    delta_put,
    gamma,
    theta_call,
    theta_put,
    vega,
    rho_call,
    rho_put,
    implied_volatility,
    implied_volatility_brent
)


def main():

    # Option parameters
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.20
    q = 0.0

    print("=" * 45)
    print("BLACK-SCHOLES PRICING ENGINE")
    print("=" * 45)

    # Option pricing
    call_price = black_scholes_call(S, K, T, r, sigma, q)
    put_price = black_scholes_put(S, K, T, r, sigma, q)

    print("\nOPTION PRICES")
    print(f"Call Price: {call_price:.6f}")
    print(f"Put Price:  {put_price:.6f}")

    # Put-call parity
    parity = put_call_parity(S, K, T, r, sigma, q)

    print("\nPUT-CALL PARITY")
    print(f"Call - Put: {parity['call_minus_put']:.6f}")
    print(f"Parity RHS: {parity['parity_value']:.6f}")
    print(f"Parity Valid: {parity['is_valid']}")

    # Greeks
    print("\nCALL GREEKS")
    print(f"Delta: {delta_call(S, K, T, r, sigma, q):.6f}")
    print(f"Gamma: {gamma(S, K, T, r, sigma, q):.6f}")
    print(f"Theta: {theta_call(S, K, T, r, sigma, q):.6f}")
    print(f"Vega:  {vega(S, K, T, r, sigma, q):.6f}")
    print(f"Rho:   {rho_call(S, K, T, r, sigma, q):.6f}")

    print("\nPUT GREEKS")
    print(f"Delta: {delta_put(S, K, T, r, sigma, q):.6f}")
    print(f"Gamma: {gamma(S, K, T, r, sigma, q):.6f}")
    print(f"Theta: {theta_put(S, K, T, r, sigma, q):.6f}")
    print(f"Vega:  {vega(S, K, T, r, sigma, q):.6f}")
    print(f"Rho:   {rho_put(S, K, T, r, sigma, q):.6f}")

    # Implied volatility
    print("\nIMPLIED VOLATILITY")

    iv_newton = implied_volatility(
        "call", S, K, T, r, call_price, q
    )

    iv_brent = implied_volatility_brent(
        "call", S, K, T, r, call_price, q
    )

    print(f"Original Volatility: {sigma:.6f}")
    print(f"Newton-Raphson IV:   {iv_newton:.6f}")
    print(f"Brent IV:            {iv_brent:.6f}")


if __name__ == "__main__":
    main()