from fractions import Fraction
from numbers import Real


def degrees_decimal_to_degrees_minutes_seconds(
    degrees_decimal: float,
) -> tuple[int, int, float]:
    sign = -1 if degrees_decimal < 0 else 1
    degrees = int(abs(degrees_decimal))
    decimal_minutes = (abs(degrees_decimal) - degrees) * 60
    minutes = int(decimal_minutes)
    seconds = (decimal_minutes - minutes) * 60
    return degrees * sign, minutes * sign, seconds * sign


def number_to_exif_rational(number: Real):
    fraction = Fraction(number).limit_denominator(1000)
    return fraction.numerator, fraction.denominator
