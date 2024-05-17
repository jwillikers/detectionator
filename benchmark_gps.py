#!/usr/bin/env python3
from functools import partial
from timeit import timeit

import gps

from detectionator import get_gps_exif_metadata

gps_session = gps.gps(mode=gps.WATCH_ENABLE)

time_get_gps_exif_metadata = timeit(
    partial(get_gps_exif_metadata.__wrapped__, gps_session)
)

print("get_gps_exif_metadata:", time_get_gps_exif_metadata, sep="\t")
