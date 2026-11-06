import pytest

from black_scholes import (
    delta_call,
    delta_put,
    gamma,
    theta_call,
    theta_put,
    vega,
    rho_call,
    rho_put,
    black_scholes_call,
    black_scholes_put
)

def test_call_delta_known_value():
    result = delta_call(100, 100, 1, 0.05, 0.20)

    assert result == pytest.approx(0.636831, abs=1e-6)


def test_put_delta_known_value():
    result = delta_put(100, 100, 1, 0.05, 0.20)

    assert result == pytest.approx(-0.363169, abs=1e-6)


def test_call_delta_positive():
    result = delta_call(100, 100, 1, 0.05, 0.20)

    assert result > 0


def test_put_delta_negative():
    result = delta_put(100, 100, 1, 0.05, 0.20)

    assert result < 0


def test_call_put_delta_relationship():
    call_delta = delta_call(100, 100, 1, 0.05, 0.20, 0.03)
    put_delta = delta_put(100, 100, 1, 0.05, 0.20, 0.03)

    assert call_delta - put_delta == pytest.approx(
        __import__("math").exp(-0.03),
        abs=1e-10
    )


def test_gamma_known_value():
    result = gamma(100, 100, 1, 0.05, 0.20)

    assert result == pytest.approx(0.018762, abs=1e-6)


def test_gamma_positive():
    result = gamma(100, 100, 1, 0.05, 0.20)

    assert result > 0


def test_call_theta_known_value():
    result = theta_call(100, 100, 1, 0.05, 0.20)

    assert result == pytest.approx(-6.414028, abs=1e-5)


def test_put_theta_known_value():
    result = theta_put(100, 100, 1, 0.05, 0.20)

    assert result == pytest.approx(-1.657880, abs=1e-5)


def test_vega_known_value():
    result = vega(100, 100, 1, 0.05, 0.20)

    assert result == pytest.approx(37.524034, abs=1e-5)


def test_vega_positive():
    result = vega(100, 100, 1, 0.05, 0.20)

    assert result > 0


def test_call_rho_known_value():
    result = rho_call(100, 100, 1, 0.05, 0.20)

    assert result == pytest.approx(53.232482, abs=1e-5)


def test_put_rho_known_value():
    result = rho_put(100, 100, 1, 0.05, 0.20)

    assert result == pytest.approx(-41.890461, abs=1e-5)


def test_call_delta_finite_difference():
    S = 100
    h = 0.001

    numerical_delta = (
        black_scholes_call(S + h, 100, 1, 0.05, 0.20)
        - black_scholes_call(S - h, 100, 1, 0.05, 0.20)
    ) / (2 * h)

    analytical_delta = delta_call(100, 100, 1, 0.05, 0.20)

    assert numerical_delta == pytest.approx(
        analytical_delta,
        rel=1e-6
    )


def test_gamma_finite_difference():
    S = 100
    h = 0.01

    numerical_gamma = (
        black_scholes_call(S + h, 100, 1, 0.05, 0.20)
        - 2 * black_scholes_call(S, 100, 1, 0.05, 0.20)
        + black_scholes_call(S - h, 100, 1, 0.05, 0.20)
    ) / h**2

    analytical_gamma = gamma(100, 100, 1, 0.05, 0.20)

    assert numerical_gamma == pytest.approx(
        analytical_gamma,
        rel=1e-4
    )


def test_vega_finite_difference():
    sigma = 0.20
    h = 1e-5

    numerical_vega = (
        black_scholes_call(100, 100, 1, 0.05, sigma + h)
        - black_scholes_call(100, 100, 1, 0.05, sigma - h)
    ) / (2 * h)

    analytical_vega = vega(100, 100, 1, 0.05, sigma)

    assert numerical_vega == pytest.approx(
        analytical_vega,
        rel=1e-6
    )


def test_call_theta_finite_difference():
    T = 1
    h = 1e-5

    numerical_theta = (
        black_scholes_call(100, 100, T - h, 0.05, 0.20)
        - black_scholes_call(100, 100, T + h, 0.05, 0.20)
    ) / (2 * h)

    analytical_theta = theta_call(100, 100, T, 0.05, 0.20)

    assert numerical_theta == pytest.approx(
        analytical_theta,
        rel=1e-5
    )


def test_call_rho_finite_difference():
    r = 0.05
    h = 1e-5

    numerical_rho = (
        black_scholes_call(100, 100, 1, r + h, 0.20)
        - black_scholes_call(100, 100, 1, r - h, 0.20)
    ) / (2 * h)

    analytical_rho = rho_call(100, 100, 1, r, 0.20)

    assert numerical_rho == pytest.approx(
        analytical_rho,
        rel=1e-6
    )