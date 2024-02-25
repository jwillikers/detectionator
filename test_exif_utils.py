import pytest

from exif_utils import (
    degrees_decimal_to_degrees_minutes_seconds,
    number_to_exif_rational,
)


def test_degrees_decimal_to_degrees_minutes_seconds():
    longitude = degrees_decimal_to_degrees_minutes_seconds(42.15188)
    assert longitude[0] == 42
    assert longitude[1] == 9
    assert longitude[2] == pytest.approx(6.768)


def test_degrees_decimal_to_degrees_minutes_seconds_negative():
    longitude = degrees_decimal_to_degrees_minutes_seconds(-55.751112)
    assert longitude[0] == -55
    assert longitude[1] == -45
    assert longitude[2] == pytest.approx(-4.0032)


def test_number_to_exif_rational():
    assert number_to_exif_rational(3) == (3, 1)


def test_number_to_exif_rational_float():
    assert number_to_exif_rational(0.1) == (1, 10)
