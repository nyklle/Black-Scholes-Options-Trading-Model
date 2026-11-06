import math
import pytest

from black_scholes import (
    black_scholes_call,
    black_scholes_put,
    calculate_d1_d2,
    put_call_parity
)


def test_call_price_known_value():
    """
    Verify call price against a known Black-Scholes value.
    """
    price = black_scholes_call(
        S=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.20
    )

    assert price == pytest.approx(10.45058357, abs=1e-6)


def test_call_price_positive():
    price = black_scholes_call(100, 100, 1, 0.05, 0.20)

    assert price > 0


def test_call_price_increases_with_stock_price():
    price_1 = black_scholes_call(100, 100, 1, 0.05, 0.20)
    price_2 = black_scholes_call(110, 100, 1, 0.05, 0.20)

    assert price_2 > price_1


def test_call_price_increases_with_volatility():
    price_1 = black_scholes_call(100, 100, 1, 0.05, 0.15)
    price_2 = black_scholes_call(100, 100, 1, 0.05, 0.30)

    assert price_2 > price_1


def test_call_price_at_expiration():
    price = black_scholes_call(110, 100, 0, 0.05, 0.20)

    assert price == pytest.approx(10.0)


def test_call_price_out_of_money_at_expiration():
    price = black_scholes_call(90, 100, 0, 0.05, 0.20)

    assert price == pytest.approx(0.0)


def test_call_price_zero_volatility():
    price = black_scholes_call(100, 95, 1, 0.05, 0)

    expected = max(
        100 * math.exp(0.05) - 95,
        0
    ) * math.exp(-0.05)

    assert price == pytest.approx(expected)


def test_put_price_known_value():
    price = black_scholes_put(
        S=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.20
    )

    assert price == pytest.approx(5.57352602, abs=1e-6)


def test_put_price_positive():
    price = black_scholes_put(100, 100, 1, 0.05, 0.20)

    assert price > 0


def test_put_price_decreases_with_stock_price():
    price_1 = black_scholes_put(100, 100, 1, 0.05, 0.20)
    price_2 = black_scholes_put(110, 100, 1, 0.05, 0.20)

    assert price_2 < price_1


def test_put_price_increases_with_volatility():
    price_1 = black_scholes_put(100, 100, 1, 0.05, 0.15)
    price_2 = black_scholes_put(100, 100, 1, 0.05, 0.30)

    assert price_2 > price_1


def test_put_price_at_expiration():
    price = black_scholes_put(90, 100, 0, 0.05, 0.20)

    assert price == pytest.approx(10.0)


def test_put_price_out_of_money_at_expiration():
    price = black_scholes_put(110, 100, 0, 0.05, 0.20)

    assert price == pytest.approx(0.0)


def test_put_price_zero_volatility():
    price = black_scholes_put(100, 105, 1, 0.05, 0)

    expected = max(
        105 - 100 * math.exp(0.05),
        0
    ) * math.exp(-0.05)

    assert price == pytest.approx(expected)


def test_calculate_d1_d2():
    d1, d2 = calculate_d1_d2(
        S=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.20
    )

    expected_d1 = 0.35
    expected_d2 = 0.15

    assert d1 == pytest.approx(expected_d1)
    assert d2 == pytest.approx(expected_d2)


def test_d1_greater_than_d2():
    d1, d2 = calculate_d1_d2(100, 100, 1, 0.05, 0.20)

    assert d1 > d2


def test_d1_d2_reject_zero_maturity():
    with pytest.raises(ValueError):
        calculate_d1_d2(100, 100, 0, 0.05, 0.20)


def test_d1_d2_reject_zero_volatility():
    with pytest.raises(ValueError):
        calculate_d1_d2(100, 100, 1, 0.05, 0)


def test_put_call_parity():
    result = put_call_parity(
        S=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.20
    )

    assert result["is_valid"] is True

    assert result["call_minus_put"] == pytest.approx(
        result["parity_value"],
        abs=1e-10
    )


def test_put_call_parity_with_dividend_yield():
    result = put_call_parity(
        S=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.20,
        q=0.03
    )

    assert result["is_valid"] is True


@pytest.mark.parametrize(
    "invalid_input",
    [
        0,
        -100
    ]
)
def test_reject_invalid_stock_price(invalid_input):
    with pytest.raises(ValueError):
        black_scholes_call(
            invalid_input, 100, 1, 0.05, 0.20
        )


@pytest.mark.parametrize(
    "invalid_input",
    [
        0,
        -100
    ]
)
def test_reject_invalid_strike_price(invalid_input):
    with pytest.raises(ValueError):
        black_scholes_call(
            100, invalid_input, 1, 0.05, 0.20
        )


def test_reject_negative_maturity():
    with pytest.raises(ValueError):
        black_scholes_call(100, 100, -1, 0.05, 0.20)


def test_reject_negative_volatility():
    with pytest.raises(ValueError):
        black_scholes_call(100, 100, 1, 0.05, -0.20)


@pytest.mark.parametrize(
    "invalid_input",
    [
        float("nan"),
        float("inf"),
        float("-inf")
    ]
)
def test_reject_non_finite_stock_price(invalid_input):
    with pytest.raises(ValueError):
        black_scholes_call(
            invalid_input, 100, 1, 0.05, 0.20
        )


@pytest.mark.parametrize(
    "invalid_input",
    [
        "100",
        None,
        True
    ]
)
def test_reject_non_numeric_stock_price(invalid_input):
    with pytest.raises(TypeError):
        black_scholes_call(
            invalid_input, 100, 1, 0.05, 0.20
        )