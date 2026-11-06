import pytest

from black_scholes import (
    black_scholes_call,
    black_scholes_put,
    implied_volatility,
    implied_volatility_brent
)


def test_newton_call_implied_volatility():
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.20

    market_price = black_scholes_call(
        S, K, T, r, sigma
    )

    result = implied_volatility(
        "call", S, K, T, r, market_price
    )

    assert result == pytest.approx(sigma, abs=1e-7)


def test_newton_put_implied_volatility():
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.25

    market_price = black_scholes_put(
        S, K, T, r, sigma
    )

    result = implied_volatility(
        "put", S, K, T, r, market_price
    )

    assert result == pytest.approx(sigma, abs=1e-7)


def test_brent_call_implied_volatility():
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.20

    market_price = black_scholes_call(
        S, K, T, r, sigma
    )

    result = implied_volatility_brent(
        "call", S, K, T, r, market_price
    )

    assert result == pytest.approx(sigma, abs=1e-8)


def test_brent_put_implied_volatility():
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.25

    market_price = black_scholes_put(
        S, K, T, r, sigma
    )

    result = implied_volatility_brent(
        "put", S, K, T, r, market_price
    )

    assert result == pytest.approx(sigma, abs=1e-8)


def test_newton_and_brent_agree():
    S = 100
    K = 105
    T = 0.75
    r = 0.04
    sigma = 0.30

    market_price = black_scholes_call(
        S, K, T, r, sigma
    )

    newton_result = implied_volatility(
        "call", S, K, T, r, market_price
    )

    brent_result = implied_volatility_brent(
        "call", S, K, T, r, market_price
    )

    assert newton_result == pytest.approx(
        brent_result,
        abs=1e-7
    )


def test_invalid_option_type():
    with pytest.raises(ValueError):
        implied_volatility(
            "invalid", 100, 100, 1, 0.05, 10
        )


def test_negative_market_price():
    with pytest.raises(ValueError):
        implied_volatility(
            "call", 100, 100, 1, 0.05, -1
        )


def test_call_market_price_above_upper_bound():
    with pytest.raises(ValueError):
        implied_volatility(
            "call", 100, 100, 1, 0.05, 120
        )


def test_put_market_price_above_upper_bound():
    with pytest.raises(ValueError):
        implied_volatility(
            "put", 100, 100, 1, 0.05, 120
        )


def test_call_market_price_at_lower_bound():
    lower_bound = max(
        100 - 100 * 2.718281828459045 ** (-0.05),
        0
    )

    with pytest.raises(ValueError):
        implied_volatility(
            "call", 100, 100, 1, 0.05, lower_bound
        )


def test_brent_invalid_volatility_interval():
    with pytest.raises(ValueError):
        implied_volatility_brent(
            "call",
            100,
            100,
            1,
            0.05,
            10,
            min_vol=0.5,
            max_vol=0.6
        )


def test_brent_invalid_volatility_bounds():
    with pytest.raises(ValueError):
        implied_volatility_brent(
            "call",
            100,
            100,
            1,
            0.05,
            10,
            min_vol=0.0,
            max_vol=5.0
        )


@pytest.mark.parametrize(
    "sigma",
    [
        0.10,
        0.15,
        0.20,
        0.30,
        0.40,
        0.60
    ]
)
def test_brent_recovers_different_volatilities(sigma):

    S = 100
    K = 100
    T = 1
    r = 0.05

    market_price = black_scholes_call(
        S, K, T, r, sigma
    )

    result = implied_volatility_brent(
        "call", S, K, T, r, market_price
    )

    assert result == pytest.approx(
        sigma,
        abs=1e-7
    )