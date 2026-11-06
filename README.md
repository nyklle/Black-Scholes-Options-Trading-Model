# Black-Scholes Options Pricing Engine

A Python-based Black-Scholes options pricing engine implementing European call and put pricing, option Greeks, and numerical implied volatility solvers.

## Features

### 1. Black-Scholes Option Pricing

- European call option pricing
- European put option pricing
- Continuous dividend yield support
- Zero-volatility handling
- Expiration payoff handling
- Input validation

### 2. Option Greeks

Analytical calculations of the five primary option Greeks:

| Greek | Description |
|---|---|
| Delta | Sensitivity to underlying asset price |
| Gamma | Sensitivity of Delta to underlying asset price |
| Theta | Sensitivity to the passage of time |
| Vega | Sensitivity to volatility |
| Rho | Sensitivity to the risk-free interest rate |

Both call and put Greeks are supported where applicable.

### 3. Implied Volatility

Two numerical methods are implemented:

**Newton-Raphson Method**
- Uses option Vega as the derivative.
- Iteratively updates volatility estimates.
- Includes convergence and numerical stability safeguards.

**Brent's Method**
- Uses a bracketing approach.
- Does not require Vega.
- Provides a robust alternative to Newton-Raphson.

### 4. Put-Call Parity

Verifies the European put-call parity relationship:

\[
C-P=Se^{-qT}-Ke^{-rT}
\]

### 5. Numerical Verification

The test suite covers:

- Known Black-Scholes pricing values.
- Put-call parity.
- Input validation.
- Edge cases.
- Finite-difference verification of analytical Greeks.
- Implied volatility recovery.
- Agreement between numerical solvers.

---

## Mathematical Background

The Black-Scholes model assumes a European option on an underlying asset following geometric Brownian motion.

The call option price is:

\[
C=Se^{-qT}N(d_1)-Ke^{-rT}N(d_2)
\]

The put option price is:

\[
P=Ke^{-rT}N(-d_2)-Se^{-qT}N(-d_1)
\]

Where:

\[
d_1=
\frac{
\ln(S/K)+(r-q+\frac{1}{2}\sigma^2)T
}{
\sigma\sqrt{T}
}
\]

\[
d_2=d_1-\sigma\sqrt{T}
\]

### Model Parameters

| Parameter | Description |
|---|---|
| S | Current underlying asset price |
| K | Option strike price |
| T | Time to maturity in years |
| r | Continuously compounded risk-free interest rate |
| q | Continuous dividend yield |
| σ | Annualized volatility |
| N(x) | Standard normal cumulative distribution function |

---

## Project Structure

```text
black-scholes-model/
│
├── black_scholes/
│   ├── __init__.py
│   ├── pricing.py
│   ├── greeks.py
│   └── implied_volatility.py
│
├── examples/
│   └── basic_pricing.py
│
├── tests/
│   ├── test_pricing.py
│   ├── test_greeks.py
│   └── test_implied_volatility.py
│
└── requirements.txt
