import pytest

from xmp_utils import (
    degrees_decimal_to_degrees_minutes_seconds_xmp,
    number_to_xmp_rational,
    xmp_xml,
)


def test_degrees_decimal_to_degrees_minutes_seconds_xmp():
    longitude = degrees_decimal_to_degrees_minutes_seconds_xmp(42.15188).split(",")
    assert int(longitude[0]) == 42
    assert int(longitude[1]) == 9
    assert float(longitude[2]) == pytest.approx(6.768)


def test_degrees_decimal_to_degrees_minutes_seconds_negative_xmp():
    longitude = degrees_decimal_to_degrees_minutes_seconds_xmp(-55.751112).split(",")
    assert int(longitude[0]) == 55
    assert int(longitude[1]) == 45
    assert float(longitude[2]) == pytest.approx(4.0032)


def test_number_to_exif_rational():
    assert number_to_xmp_rational(3) == "3/1"


def test_number_to_exif_rational_float():
    assert number_to_xmp_rational(0.1) == "1/10"


def test_xmp_xml():
    assert (
        xmp_xml(332.2, 44.1234, -75.1234)
        == """<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?>
    <x:xmpmeta xmlns:x='adobe:ns:meta/'>
    <rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>
      <rdf:Description rdf:about='' xmlns:exif='http://ns.adobe.com/exif/1.0/'>
        <exif:GPSAltitude>1661/5</exif:GPSAltitude>
        <exif:GPSLatitude>44,7,24.239999999987845N</exif:GPSLatitude>
        <exif:GPSLongitude>75,7,24.240000000013424W</exif:GPSLongitude>
      </rdf:Description>
    </rdf:RDF>
    </x:xmpmeta>
    <?xpacket end='w'?>"""
    )
