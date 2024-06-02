from fractions import Fraction
from numbers import Real


def degrees_decimal_to_degrees_minutes_seconds_xmp(
    degrees_decimal: Real,
) -> str:
    degrees = int(abs(degrees_decimal))
    minutes_decimal = (abs(degrees_decimal) - degrees) * 60
    minutes = int(minutes_decimal)
    seconds = (minutes_decimal - minutes) * 60
    return str(degrees) + "," + str(minutes) + "," + str(seconds)


def number_to_xmp_rational(number: Real) -> str:
    fraction = Fraction(number).limit_denominator(1000)
    return str(fraction.numerator) + "/" + str(fraction.denominator)


def xmp_xml(altitude: Real, latitude: Real, longitude: Real) -> str:
    altitude_xmp = number_to_xmp_rational(altitude)
    latitude_xmp = degrees_decimal_to_degrees_minutes_seconds_xmp(latitude) + (
        "N" if latitude > 0 else "S"
    )
    longitude_xmp = degrees_decimal_to_degrees_minutes_seconds_xmp(longitude) + (
        "E" if longitude > 0 else "W"
    )
    return f"""<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?>
    <x:xmpmeta xmlns:x='adobe:ns:meta/'>
    <rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>
      <rdf:Description rdf:about='' xmlns:exif='http://ns.adobe.com/exif/1.0/'>
        <exif:GPSAltitude>{altitude_xmp}</exif:GPSAltitude>
        <exif:GPSLatitude>{latitude_xmp}</exif:GPSLatitude>
        <exif:GPSLongitude>{longitude_xmp}</exif:GPSLongitude>
      </rdf:Description>
    </rdf:RDF>
    </x:xmpmeta>
    <?xpacket end='w'?>"""
