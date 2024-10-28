#!/usr/bin/env python3
import argparse
import asyncio
import bisect
import datetime
from dateutil import parser
from functools import partial
import logging
from numbers import Real
import os
import pathlib
import signal
import sys
import time

import configargparse
import cv2
import gps.aiogps
from libcamera import controls
import numpy as np
from picamera2 import MappedArray, Picamera2
from picamera2.devices import Hailo
from picamera2.encoders import H264Encoder, Quality
from picamera2.outputs import FfmpegOutput
import piexif
import sdnotify
import tflite_runtime.interpreter as tflite

from exif_utils import (
    degrees_decimal_to_degrees_minutes_seconds,
    number_to_exif_rational,
)


from xmp_utils import (
    xmp_xml,
)

logger = logging.getLogger(__name__)

rectangles = []


def draw_bounding_boxes(
    request,
    resolution_scale: tuple[Real, Real],
    resolution: tuple[Real, Real],
    scale_factor: Real = 1,
):
    with MappedArray(request, "main") as image:
        for rectangle in rectangles:
            box = cast_int(
                clamp(
                    dilate(scale(rectangle[0:4], resolution_scale), scale_factor),
                    resolution,
                )
            )
            x_min, y_min, _, _ = box
            cv2.rectangle(
                image.array, (box[0], box[1]), (box[2], box[3]), (0, 255, 0, 0)
            )
            if len(rectangle) == 5:
                text = rectangle[4]
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    image.array,
                    text,
                    (x_min, y_min - 10),
                    font,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )


def read_label_file(file_path):
    with open(file_path, "r", encoding="UTF-8") as f:
        lines = f.readlines()
    ret = {}
    for line in lines:
        pair = line.strip().split(maxsplit=1)
        ret[int(pair[0])] = pair[1].strip()
    return ret


def yolo_class_filter(classdata):
    return [c.argmax() for c in classdata]


# Determine if two rectangles are duplicates according to the given percentage of overlap.
def rectangles_are_duplicates(r1, r2, min_overlap: float):
    return percentage_intersecting(r1, r2) >= min_overlap


def intersection_area(r1, r2):
    dx = min(r1[2], r2[2]) - max(r1[0], r2[0])
    dy = min(r1[3], r2[3]) - max(r1[1], r2[1])
    if dx >= 0 and dy >= 0:
        return dx * dy
    return None


def union_area(r1, r2):
    intersection = intersection_area(r1, r2)
    if intersection is None:
        return None
    return rectangle_area(r1) + rectangle_area(r2) - intersection


def rectangle_area(rectangle):
    return (rectangle[2] - rectangle[0]) * (rectangle[3] - rectangle[1])


def percentage_intersecting(r1, r2):
    intersection = intersection_area(r1, r2)
    if intersection is None:
        return 0
    return intersection / union_area(r1, r2)


def combine_rectangles(r1, r2):
    min_x = min(r1[0], r2[0])
    min_y = min(r1[1], r2[1])
    max_x = max(r1[2], r2[2])
    max_y = max(r1[3], r2[3])
    return min_x, min_y, max_x, max_y


def combine_detections(d1, d2):
    if d1 is None and d2 is None:
        return None
    if d1 is None:
        return d2
    if d2 is None:
        return d1
    most_confident_class = d1[2] if d1[0] >= d2[0] else d2[2]
    return max(d1[0], d2[0]), combine_rectangles(d1[1], d2[1]), most_confident_class


def detections_are_duplicates(d1, d2, min_overlap: float):
    return rectangles_are_duplicates(d1[1], d2[1], min_overlap)


# Combine detections at overlapping coordinates.
# The min_overlap parameter is used to determine the minimum percentage of overlapping rectangles that is considered a duplicate.
# Duplicate detections are combined to a single detection.
# The coordinates for the combined bounding box are the minimum coordinates that encapsulate all of the individual bounding boxes.
def combine_duplicate_detections(detections, min_overlap: float):
    if len(detections) <= 1:
        return detections

    deduplicated_detections = []
    sorted_detections = sorted(detections, key=lambda y: (y[1][0], y[1][1], y[1][2], y[1][3]))
    i = 0
    while i < len(sorted_detections) - 1:
        current = sorted_detections[i]
        j = 0
        while i + j < len(sorted_detections) and detections_are_duplicates(current, sorted_detections[j], min_overlap):
            current = combine_detections(current, sorted_detections[j])
            j += 1
        deduplicated_detections.append(current)
        i += j + 1
    return deduplicated_detections


# Combine rectangles that overlap each other.
# The coordinates for the combined bounding box are the minimum coordinates that encapsulate all of the individual bounding boxes.
def combine_intersecting_rectangles(rectangles):
    if len(rectangles) <= 1:
        return rectangles

    combined_rectangles = []
    sorted_rectangles = sorted(rectangles, key=lambda y: (y[0], y[1], y[2], y[3]))
    i = 0
    while i < len(sorted_rectangles) - 1:
        current = sorted_rectangles[i]
        j = 0
        while i + j < len(sorted_rectangles) and intersection_area(current, sorted_rectangles[j]) > 0:
            current = combine_rectangles(current, sorted_rectangles[j])
            j += 1
        combined_rectangles.append(current)
        i += j + 1
    return combined_rectangles


def inference_hailo(
        image,
        hailo,
        labels: list,
        match_labels: list,
        threshold: float,
):
    output = hailo.run(image)[0]
    height, width, _ = hailo.get_input_shape()

    # detection:
    #   score
    #   rectangle
    #   label
    results = list()
    for class_id, detections in enumerate(output):
        if match_labels and labels[class_id] not in match_labels:
            continue
        for detection in detections:
            score = detection[4]
            if score >= threshold:
                bottom, left, top, right = detection[:4]
                x_min = int(left * width)
                y_min = int(bottom * height)
                x_max = int(right * width)
                y_max = int(top * height)
                box = [x_min, y_min, x_max, y_max]
                result = (score, box)
                if labels:
                    result = (*result, labels[class_id])
                results.append(result)
    return results


def inference_tensorflow(
    image, interpreter, labels: dict, match_labels: list, threshold: float, is_yolo: bool
):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]["shape"][1]
    width = input_details[0]["shape"][2]
    floating_model = False
    if input_details[0]["dtype"] == np.float32:
        floating_model = True

    initial_height, initial_width, _channels = image.shape
    if (initial_width, initial_height) != (width, height):
        image = cv2.resize(image, (width, height))

    input_data = np.expand_dims(image, axis=0)
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    interpreter.set_tensor(input_details[0]["index"], input_data)

    interpreter.invoke()

    detected_boxes = []
    detected_classes = []
    detected_scores = []
    num_boxes = 0
    if is_yolo:
        output_data = interpreter.get_tensor(output_details[0]["index"])[0]
        boxes = np.squeeze(output_data[..., :4])
        detected_scores = np.squeeze(
            output_data[..., 4:5]
        )  # confidences  [25200, 1]
        detected_classes = yolo_class_filter(output_data[..., 5:])
        x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]  # xywh
        # detected_boxes = [y - h / 2, x - w / 2, y + h / 2, x + w / 2]  # xywh to yxyx
        detected_boxes = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
        num_boxes = len(detected_scores)
    else:
        detected_boxes = interpreter.get_tensor(output_details[0]["index"])[0]
        detected_classes = interpreter.get_tensor(output_details[1]["index"])[0]
        detected_scores = interpreter.get_tensor(output_details[2]["index"])[0]
        num_boxes = int(interpreter.get_tensor(output_details[3]["index"]).item())

    detections = list()
    # detection:
    #   score
    #   rectangle
    #   label
    for i in range(num_boxes):
        class_id = int(detected_classes[i])
        if match_labels and labels[class_id] not in match_labels:
            continue
        score = detected_scores[i]
        if score > threshold and score <= 1.0:
            bottom, left, top, right = (0, 0, 0, 0)
            if is_yolo:
                bottom = detected_boxes[0][i]
                left = detected_boxes[1][i]
                top = detected_boxes[2][i]
                right = detected_boxes[3][i]
            else:
                bottom, left, top, right = detected_boxes[i]
            x_min = left * initial_width
            y_min = bottom * initial_height
            x_max = right * initial_width
            y_max = top * initial_height
            box = [x_min, y_min, x_max, y_max]
            detection = (score, box)
            if labels:
                detection = (*detection, labels[class_id])
            detections.append(detection)
    return combine_duplicate_detections(detections, 0.8)


def inference(
    image,
    interpreter,
    labels,
    match_labels: list,
    threshold: float,
    is_yolo: bool,
):
    if isinstance(interpreter, Hailo):
        return inference_hailo(image, interpreter, labels, match_labels, threshold)
    return inference_tensorflow(image, interpreter, labels, match_labels, threshold, is_yolo)


# Convert a rectangle defined by two coordinates to a string representation.
def rectangle_coordinates_to_string(rectangle):
    return f"(x1: {rectangle[0]}, y1: {rectangle[1]}), (x2: {rectangle[2]}, y2: {rectangle[3]})"


# Convert a rectangle defined by one coordinate, a width, and a height, to a string representation.
def rectangle_coordinate_width_height_to_string(rectangle):
    return f"x: {rectangle[0]}, y: {rectangle[1]}, width: {rectangle[2]}, height: {rectangle[3]}"


# Convert a detection to a string representation.
def detection_to_string(detection):
    # detection:
    #   score
    #   rectangle
    #   label
    rectangle = rectangle_coordinates_to_string(detection[1])
    if len(detection) == 3:
        return f"Score: {detection[0]}, Box: {rectangle}, Label: {detection[2]}"
    else:
        return f"Score: {detection[0]}, Box: ({rectangle})"


# todo Add some unit tests for this.
def rectangle_coordinates_to_coordinate_width_height(rectangle):
    min_x = min(rectangle[0], rectangle[2])
    min_y = min(rectangle[1], rectangle[3])
    return (
        min_x,
        min_y,
        abs(rectangle[2] - rectangle[0]),
        abs(rectangle[3] - rectangle[1]),
    )


async def update_gps_mp4_metadata(gpsd: gps.aiogps.aiogps, gps_mp4_metadata: dict):
    try:
        while True:
            await gpsd.read()
            if not (gps.PACKET_SET & gpsd.valid):
                logger.warning("Failed to receive package from gpsd")
                await asyncio.sleep(0)
                continue

            if not (gps.MODE_SET & gpsd.valid):
                logger.debug("GPS session invalid")
                await asyncio.sleep(0)
                continue

            fix_mode = gpsd.fix.mode
            if fix_mode in [0, gps.MODE_NO_FIX]:
                logger.warning("No GPS fix")
                await asyncio.sleep(10)
                continue
            if gps.isfinite(gpsd.fix.latitude):
                gps_mp4_metadata["latitude"] = gpsd.fix.latitude
            if gps.isfinite(gpsd.fix.longitude):
                gps_mp4_metadata["longitude"] = gpsd.fix.longitude
            if gps.isfinite(gpsd.fix.altitude):
                gps_mp4_metadata["altitude"] = gpsd.fix.altitude
            logger.debug("Updated MP4 GPS data.")
            await asyncio.sleep(600)
    except asyncio.IncompleteReadError:
        logger.info("Connection closed by server")
    except asyncio.TimeoutError:
        logger.error("Timeout waiting for gpsd to respond")
    except Exception as exc:  # pylint: disable=W0703
        logger.error(f"Error: {exc}")
    except asyncio.CancelledError:
        return {}


async def update_gps_exif_metadata(gpsd: gps.aiogps.aiogps, gps_exif_metadata: dict):
    try:
        while True:
            await gpsd.read()
            if not (gps.PACKET_SET & gpsd.valid):
                logger.warning("Failed to receive package from gpsd")
                await asyncio.sleep(0)
                continue

            if not (gps.MODE_SET & gpsd.valid):
                logger.debug("GPS session invalid")
                await asyncio.sleep(0)
                continue

            fix_mode = gpsd.fix.mode
            if fix_mode in [0, gps.MODE_NO_FIX]:
                logger.warning("No GPS fix")
                await asyncio.sleep(10)
                continue

            # if gps.ALTITUDE_SET & gpsd.valid
            altitude = gpsd.fix.altHAE
            speed = gpsd.fix.speed

            gps_exif_metadata[piexif.GPSIFD.GPSAltitude] = number_to_exif_rational(
                abs(altitude if gps.isfinite(altitude) else 0)
            )
            gps_exif_metadata[piexif.GPSIFD.GPSAltitudeRef] = (
                1 if gps.isfinite(altitude) and altitude <= 0 else 0
            )
            gps_exif_metadata[piexif.GPSIFD.GPSProcessingMethod] = "GPS".encode("ASCII")
            gps_exif_metadata[piexif.GPSIFD.GPSSatellites] = str(gpsd.satellites_used)
            gps_exif_metadata[piexif.GPSIFD.GPSSpeed] = (
                number_to_exif_rational(speed * 3.6)
                if gps.isfinite(speed)
                # Convert m/sec to km/hour
                else number_to_exif_rational(0)
            )
            gps_exif_metadata[piexif.GPSIFD.GPSSpeedRef] = "K"
            gps_exif_metadata[piexif.GPSIFD.GPSVersionID] = (2, 3, 0, 0)

            if gps.isfinite(gpsd.fix.latitude):
                latitude = degrees_decimal_to_degrees_minutes_seconds(gpsd.fix.latitude)
                gps_exif_metadata[piexif.GPSIFD.GPSLatitude] = (
                    number_to_exif_rational(abs(latitude[0])),
                    number_to_exif_rational(abs(latitude[1])),
                    number_to_exif_rational(abs(latitude[2])),
                )
                gps_exif_metadata[piexif.GPSIFD.GPSLatitudeRef] = (
                    "N" if latitude[0] > 0 else "S"
                )

            if gps.isfinite(gpsd.fix.longitude):
                longitude = degrees_decimal_to_degrees_minutes_seconds(
                    gpsd.fix.longitude
                )
                gps_exif_metadata[piexif.GPSIFD.GPSLongitude] = (
                    number_to_exif_rational(abs(longitude[0])),
                    number_to_exif_rational(abs(longitude[1])),
                    number_to_exif_rational(abs(longitude[2])),
                )
                gps_exif_metadata[piexif.GPSIFD.GPSLongitudeRef] = (
                    "E" if longitude[0] > 0 else "W"
                )

            gps_exif_metadata[piexif.GPSIFD.GPSMeasureMode] = str(fix_mode)

            fix_time = parser.parse(str(gpsd.fix.time))
            gps_exif_metadata[piexif.GPSIFD.GPSDateStamp] = fix_time.strftime(
                "%Y:%m:%d"
            )
            gps_exif_metadata[piexif.GPSIFD.GPSTimeStamp] = (
                number_to_exif_rational(fix_time.hour),
                number_to_exif_rational(fix_time.minute),
                number_to_exif_rational(fix_time.second),
            )
            logger.debug("Updated EXIF GPS data.")
            await asyncio.sleep(600)
    except asyncio.IncompleteReadError:
        logger.info("Connection closed by server")
    except asyncio.TimeoutError:
        logger.error("Timeout waiting for gpsd to respond")
    except Exception as exc:  # pylint: disable=W0703
        logger.error(f"Error: {exc}")
    except asyncio.CancelledError:
        return {}


def captured_file(filename: str, detections, job):
    if job:
        logger.info(f"Captured image '{filename}': {detections}")
    else:
        logger.error(f"Failed to capture image '{filename}': {detections}")


# Sort detections from the lowest confidence rating to the highest confidence rating.
def sort_detections_by_confidence(detections: list):
    return sorted(detections, key=lambda x: x[0], reverse=True)


# Widen a bounding box rectangle.
# The rectangle should be represented by the minimum and maximum coordinates.
def pad(rectangle, padding: Real):
    x_min, y_min, x_max, y_max = rectangle
    return x_min - padding, y_min - padding, x_max + padding, y_max + padding


# Expand or contract a rectangle based on the given factor.
# The factor should be positive for expansion and negative for contraction.
# The rectangle should be represented by the minimum and maximum coordinates.
def dilate(rectangle, factor: Real):
    x_min, y_min, x_max, y_max = rectangle
    width = x_max - x_min
    height = y_max - y_min
    additional_width = width * factor
    additional_height = height * factor
    additional_width_per_side = additional_width / 2
    additional_height_per_side = additional_height / 2
    return (
        x_min - additional_width_per_side,
        y_min - additional_height_per_side,
        x_max + additional_width_per_side,
        y_max + additional_height_per_side,
    )


# Convert the components of a rectangle to integers.
def cast_int(rectangle):
    x_min, y_min, x_max, y_max = rectangle
    return int(x_min), int(y_min), int(x_max), int(y_max)


# Move a rectangle by the given x and y amounts.
# The rectangle should be represented by the minimum and maximum coordinates.
def translate(rectangle, offset: tuple[Real, Real]):
    x_min, y_min, x_max, y_max = rectangle
    return x_min + offset[0], y_min + offset[1], x_max + offset[0], y_max + offset[1]


# Scale a rectangle by the given width and height ratios.
# The rectangle should be represented by the minimum and maximum coordinates.
def scale(rectangle, ratio: tuple[Real, Real]):
    x_min, y_min, x_max, y_max = rectangle
    return x_min * ratio[0], y_min * ratio[1], x_max * ratio[0], y_max * ratio[1]


# Clamp a rectangle within the given max and min ranges.
# The rectangle should be represented by the minimum and maximum coordinates.
def clamp(
    rectangle, max_extent: tuple[Real, Real], min_extent: tuple[Real, Real] = (0, 0)
):
    x_min, y_min, x_max, y_max = rectangle
    x_min = max(min_extent[0], x_min)
    y_min = max(min_extent[1], y_min)
    x_max = min(max_extent[0], x_max)
    y_max = min(max_extent[1], y_max)
    return x_min, y_min, x_max, y_max


# Split a sorted list of detections according to a given amount of confidence.
#
# Requires the list of detections to be sorted by the confidence rating.
#
# The first list is the list with values less than the confidence rating.
# The second list is the list with values greater than or equal to the confidence rating.
def split_detections_based_on_confidence(
    detections: list, confidence: float
) -> tuple[list, list]:
    index = bisect.bisect_left(detections, confidence, key=lambda x: x[0])
    return detections[:index], detections[index:]


async def detect_and_capture(
    picam2,
    interpreter,
    labels: dict,
    match: list,
    gps_exif_metadata: dict,
    scaler_crop_maximum_ratio: tuple[Real, Real],
    scaler_crop_maximum_offset: tuple[Real, Real],
    main_resolution: tuple[int, int],
    has_autofocus,
    burst: bool,
    output_directory: pathlib.Path,
    frame: int,
    gap: float,
    detection_threshold: float,
    focal_detection_threshold: float,
    is_yolo: bool,
    bounding_box_scale_factor: Real,
):
    global rectangles
    _, max_window, _ = picam2.camera_controls["ScalerCrop"]
    while True:
        image = picam2.capture_array("lores")
        detections = sort_detections_by_confidence(
            inference(
                image, interpreter, labels, match, focal_detection_threshold, is_yolo
            )
        )
        if len(detections) == 0:
            # Take a quick breather to give the CPU a break.
            # 1/5 of a second results in about 50% CPU usage.
            # 1/10 of a second results in about 80% CPU usage.
            # 1/20 of a second results in about 130% CPU usage.
            # todo Increase / decrease this wait based on recent detections.
            await asyncio.sleep(0)
            time.sleep(0.2)
            continue

        rectangles = [d[1] for d in detections]
        possible_detections, detections = split_detections_based_on_confidence(
            detections, detection_threshold
        )
        if len(possible_detections) > 0 and len(detections) == 0:
            for possible_detection in reversed(possible_detections):
                logger.info(
                    f"Possible detection: {detection_to_string(possible_detection)}"
                )

            bounding_boxes = combine_intersecting_rectangles([d[1] for d in reversed(possible_detections)])
            scaled_bounding_boxes = [
                cast_int(
                    rectangle_coordinates_to_coordinate_width_height(
                        clamp(
                            dilate(
                                scale(
                                    translate(bounding_box, scaler_crop_maximum_offset),
                                    scaler_crop_maximum_ratio,
                                ),
                                bounding_box_scale_factor,
                            ),
                            main_resolution,
                        )
                    )
                )
                for bounding_box in bounding_boxes
            ]
            for scaled_bounding_box in scaled_bounding_boxes:
                logger.info(
                    f"Scaled bounding box: {rectangle_coordinate_width_height_to_string(scaled_bounding_box)}"
                )
            # todo Combine overlapping bounding boxes when setting AfWindows.
            picam2.set_controls({"AfWindows": scaled_bounding_boxes})
            focus_cycle_job = picam2.autofocus_cycle(wait=False)
            await asyncio.sleep(0)
            if not picam2.wait(focus_cycle_job):
                picam2.set_controls({"AfWindows": [max_window]})
                focus_cycle_job = picam2.autofocus_cycle(wait=False)
                logger.warning("Autofocus cycle failed.")
                await asyncio.sleep(0)
                if not picam2.wait(focus_cycle_job):
                    logger.warning("Autofocus cycle failed.")
            time.sleep(0.15)
            continue

        for detection in reversed(detections):
            logger.info(f"Detection: {detection_to_string(detection)}")

        bounding_boxes = combine_intersecting_rectangles([d[1] for d in reversed(detections)])
        scaled_bounding_boxes = [
            cast_int(
                rectangle_coordinates_to_coordinate_width_height(
                    clamp(
                        dilate(
                            scale(
                                translate(bounding_box, scaler_crop_maximum_offset),
                                scaler_crop_maximum_ratio,
                            ),
                            bounding_box_scale_factor,
                        ),
                        main_resolution,
                    )
                )
            )
            for bounding_box in bounding_boxes
        ]
        for scaled_bounding_box in scaled_bounding_boxes:
            logger.info(
                f"Scaled bounding box: {rectangle_coordinate_width_height_to_string(scaled_bounding_box)}"
            )
        picam2.set_controls({"AfWindows": scaled_bounding_boxes})
        focus_cycle_job = None
        if has_autofocus:
            focus_cycle_job = picam2.autofocus_cycle(wait=False)

        exif_metadata = {}
        if gps_exif_metadata:
            exif_metadata["GPS"] = gps_exif_metadata
            logger.debug(f"Exif GPS metadata: {gps_exif_metadata}")
        else:
            logger.warning("No GPS fix")

        detection_names = "detection"
        if labels:
            detection_names = "-".join([i[2] for i in detections])
        filename = os.path.join(output_directory, f"{detection_names}-{frame}.jpg")
        if has_autofocus:
            if not picam2.wait(focus_cycle_job):
                picam2.set_controls({"AfWindows": [max_window]})
                focus_cycle_job = picam2.autofocus_cycle(wait=False)
                logger.warning("Autofocus cycle failed.")
                if not picam2.wait(focus_cycle_job):
                    logger.warning("Autofocus cycle failed.")
        picam2.capture_file(
            filename,
            exif_data=exif_metadata,
            format="jpeg",
            signal_function=partial(captured_file, filename, detections),
        )
        frame += 1

        # Capture burst photographs.
        for _ in range(burst - 1):
            focus_cycle_job = None
            if has_autofocus:
                focus_cycle_job = picam2.autofocus_cycle(wait=False)
            filename = os.path.join(output_directory, f"{detection_names}-{frame}.jpg")
            if has_autofocus:
                if not picam2.wait(focus_cycle_job):
                    picam2.set_controls({"AfWindows": [max_window]})
                    focus_cycle_job = picam2.autofocus_cycle(wait=False)
                    logger.warning("Autofocus cycle failed.")
                    if not picam2.wait(focus_cycle_job):
                        logger.warning("Autofocus cycle failed.")
            picam2.capture_file(
                filename,
                exif_data=exif_metadata,
                format="jpeg",
                signal_function=partial(captured_file, filename, detections),
            )
            frame += 1
        await asyncio.sleep(gap)


async def detect_and_record(
    picam2,
    interpreter,
    labels,
    match,
    gps_mp4_metadata: dict,
    scaler_crop_maximum_ratio: tuple[Real, Real],
    scaler_crop_maximum_offset: tuple[Real, Real],
    main_resolution: tuple[int, int],
    has_autofocus,
    output_directory: pathlib.Path,
    frame: int,
    gap: float,
    encoder,
    encoder_quality: Quality,
    audio: bool,
    detection_threshold: float,
    focal_detection_threshold: float,
    is_yolo: bool,
    bounding_box_scale_factor: Real,
):
    global rectangles
    _, max_window, _ = picam2.camera_controls["ScalerCrop"]
    while True:
        image = picam2.capture_array("lores")
        detections = sort_detections_by_confidence(
            inference(
                image, interpreter, labels, match, focal_detection_threshold, is_yolo
            )
        )
        if len(detections) == 0:
            time.sleep(0.2)
            await asyncio.sleep(0)
            continue

        rectangles = [d[1] for d in detections]
        possible_detections, detections = split_detections_based_on_confidence(
            detections, detection_threshold
        )
        if len(possible_detections) > 0 and len(detections) == 0:
            for possible_detection in reversed(possible_detections):
                logger.info(
                    f"Possible detection: {detection_to_string(possible_detection)}"
                )

            bounding_boxes = combine_intersecting_rectangles([d[1] for d in reversed(possible_detections)])
            scaled_bounding_boxes = [
                cast_int(
                    rectangle_coordinates_to_coordinate_width_height(
                        clamp(
                            dilate(
                                scale(
                                    translate(bounding_box, scaler_crop_maximum_offset),
                                    scaler_crop_maximum_ratio,
                                ),
                                bounding_box_scale_factor,
                            ),
                            main_resolution,
                        )
                    )
                )
                for bounding_box in bounding_boxes
            ]
            for scaled_bounding_box in scaled_bounding_boxes:
                logger.info(
                    f"Scaled bounding box: {rectangle_coordinate_width_height_to_string(scaled_bounding_box)}"
                )
            picam2.set_controls({"AfWindows": scaled_bounding_boxes})
            focus_cycle_job = picam2.autofocus_cycle(wait=False)
            await asyncio.sleep(0)
            if not picam2.wait(focus_cycle_job):
                picam2.set_controls({"AfWindows": [max_window]})
                focus_cycle_job = picam2.autofocus_cycle(wait=False)
                logger.warning("Autofocus cycle failed.")
                await asyncio.sleep(0)
                if not picam2.wait(focus_cycle_job):
                    logger.warning("Autofocus cycle failed.")
            time.sleep(0.15)
            continue

        for detection in reversed(detections):
            logger.info(f"Detection: {detection_to_string(detection)}")

        bounding_boxes = combine_intersecting_rectangles([d[1] for d in reversed(detections)])
        scaled_bounding_boxes = [
            cast_int(
                rectangle_coordinates_to_coordinate_width_height(
                    clamp(
                        dilate(
                            scale(
                                translate(bounding_box, scaler_crop_maximum_offset),
                                scaler_crop_maximum_ratio,
                            ),
                            bounding_box_scale_factor,
                        ),
                        main_resolution,
                    )
                )
            )
            for bounding_box in bounding_boxes
        ]
        for scaled_bounding_box in scaled_bounding_boxes:
            logger.info(
                f"Scaled bounding box: {rectangle_coordinate_width_height_to_string(scaled_bounding_box)}"
            )
        picam2.set_controls({"AfWindows": scaled_bounding_boxes})
        focus_cycle_job = None
        if has_autofocus:
            focus_cycle_job = picam2.autofocus_cycle(wait=False)

        detection_names = "detection"
        if labels:
            detection_names = "-".join([i[2] for i in detections])
        file = os.path.join(output_directory, f"{detection_names}-{frame}.mp4")
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ffmpeg_command = f"-metadata:g creation_time={now} "
        if gps_mp4_metadata:
            ffmpeg_command += f"-metadata:g location={gps_mp4_metadata['longitude']}+{gps_mp4_metadata['latitude']}+{gps_mp4_metadata['altitude']} -metadata:g location-eng={gps_mp4_metadata['longitude']}+{gps_mp4_metadata['latitude']}+{gps_mp4_metadata['altitude']} "
            with open(file + ".xmp", "w") as xmp_file:
                xmp_file.write(
                    xmp_xml(
                        gps_mp4_metadata["altitude"],
                        gps_mp4_metadata["latitude"],
                        gps_mp4_metadata["longitude"],
                    )
                )
            logger.debug(f"MP4 GPS metadata: {gps_mp4_metadata}")
        else:
            logger.warning("No GPS fix")
        ffmpeg_command += file
        output = FfmpegOutput(ffmpeg_command, audio=audio)
        encoder_outputs = encoder.output
        if not isinstance(encoder_outputs, list):
            encoder_outputs = [encoder_outputs]
        encoder.output = [output] + encoder_outputs
        encoder_running = encoder.running
        if not encoder_running:
            picam2.start_encoder(encoder, quality=encoder_quality)

        logger.info(f"Recording video '{file}'")
        output.start()
        if has_autofocus:
            if not picam2.wait(focus_cycle_job):
                picam2.set_controls({"AfWindows": [max_window]})
                focus_cycle_job = picam2.autofocus_cycle(wait=False)
                logger.warning("Autofocus cycle failed.")
                if not picam2.wait(focus_cycle_job):
                    logger.warning("Autofocus cycle failed.")

        time.sleep(0.125)

        if has_autofocus:
            focus_cycle_job = picam2.autofocus_cycle(wait=False)
            if not picam2.wait(focus_cycle_job):
                picam2.set_controls({"AfWindows": [max_window]})
                focus_cycle_job = picam2.autofocus_cycle(wait=False)
                logger.warning("Autofocus cycle failed.")
                if not picam2.wait(focus_cycle_job):
                    logger.warning("Autofocus cycle failed.")

        minimum_record_seconds = 6
        consecutive_failed_detections = 0
        consecutive_failed_detections_to_stop = 10
        last_detection_time = datetime.datetime.now()
        while (
            (datetime.datetime.now() - last_detection_time).seconds
            <= minimum_record_seconds
        ) or consecutive_failed_detections < consecutive_failed_detections_to_stop:
            image = picam2.capture_array("lores")
            detections = sort_detections_by_confidence(
                inference(
                    image,
                    interpreter,
                    labels,
                    match,
                    focal_detection_threshold,
                    is_yolo,
                )
            )
            if len(detections) == 0:
                if has_autofocus:
                    focus_cycle_job = picam2.autofocus_cycle(wait=False)
                    if not picam2.wait(focus_cycle_job):
                        picam2.set_controls({"AfWindows": [max_window]})
                        focus_cycle_job = picam2.autofocus_cycle(wait=False)
                        logger.warning("Autofocus cycle failed.")
                        if not picam2.wait(focus_cycle_job):
                            logger.warning("Autofocus cycle failed.")
                consecutive_failed_detections += 1
                time.sleep(0.125)
                continue

            rectangles = [d[1] for d in detections]
            possible_detections, detections = split_detections_based_on_confidence(
                detections, detection_threshold
            )
            if len(possible_detections) > 0 and len(detections) == 0:
                for possible_detection in reversed(possible_detections):
                    logger.info(
                        f"Possible detection: {detection_to_string(possible_detection)}"
                    )

                bounding_boxes = combine_intersecting_rectangles([d[1] for d in reversed(possible_detections)])
                scaled_bounding_boxes = [
                    cast_int(
                        rectangle_coordinates_to_coordinate_width_height(
                            clamp(
                                dilate(
                                    scale(
                                        translate(
                                            bounding_box, scaler_crop_maximum_offset
                                        ),
                                        scaler_crop_maximum_ratio,
                                    ),
                                    bounding_box_scale_factor,
                                ),
                                main_resolution,
                            )
                        )
                    )
                    for bounding_box in bounding_boxes
                ]
                for scaled_bounding_box in scaled_bounding_boxes:
                    logger.info(
                        f"Scaled bounding box: {rectangle_coordinate_width_height_to_string(scaled_bounding_box)}"
                    )
                picam2.set_controls({"AfWindows": scaled_bounding_boxes})
                focus_cycle_job = picam2.autofocus_cycle(wait=False)
                if not picam2.wait(focus_cycle_job):
                    picam2.set_controls({"AfWindows": [max_window]})
                    focus_cycle_job = picam2.autofocus_cycle(wait=False)
                    logger.warning("Autofocus cycle failed.")
                    if not picam2.wait(focus_cycle_job):
                        logger.warning("Autofocus cycle failed.")
                time.sleep(0.125)
                # todo Should this set consecutive failed detections to zero, increment failed detections, or do nothing?
                # For now, err on the side of recording a lengthier video and reset everything as if there was a confident detection.
                consecutive_failed_detections = 0
                last_detection_time = datetime.datetime.now()
                continue

            for detection in reversed(detections):
                logger.info(f"Detection: {detection_to_string(detection)}")
            bounding_boxes = combine_intersecting_rectangles([d[1] for d in reversed(detections)])
            scaled_bounding_boxes = [
                cast_int(
                    rectangle_coordinates_to_coordinate_width_height(
                        clamp(
                            dilate(
                                scale(
                                    translate(bounding_box, scaler_crop_maximum_offset),
                                    scaler_crop_maximum_ratio,
                                ),
                                bounding_box_scale_factor,
                            ),
                            main_resolution,
                        )
                    )
                )
                for bounding_box in bounding_boxes
            ]
            for scaled_bounding_box in scaled_bounding_boxes:
                logger.info(
                    f"Scaled bounding box: {rectangle_coordinate_width_height_to_string(scaled_bounding_box)}"
                )
            picam2.set_controls({"AfWindows": scaled_bounding_boxes})
            if has_autofocus:
                focus_cycle_job = picam2.autofocus_cycle(wait=False)
                if not picam2.wait(focus_cycle_job):
                    picam2.set_controls({"AfWindows": [max_window]})
                    focus_cycle_job = picam2.autofocus_cycle(wait=False)
                    logger.warning("Autofocus cycle failed.")
                    if not picam2.wait(focus_cycle_job):
                        logger.warning("Autofocus cycle failed.")
            consecutive_failed_detections = 0
            time.sleep(0.125)
            last_detection_time = datetime.datetime.now()
        output.stop()
        if not encoder_running:
            picam2.stop_encoder(encoder)
        encoder.output = encoder_outputs
        logger.info(f"Finished recording video '{file}'")
        frame += 1


async def main():
    parser = configargparse.ArgParser(
        default_config_files=[
            "detectionator.toml",
            "~/.config/detectionator/config.toml",
            "/usr/local/etc/detectionator/config.toml",
            "/etc/detectionator/config.toml",
        ],
        config_file_parser_class=configargparse.TomlConfigParser(["detectionator"]),
    )
    parser.add_argument(
        "--align-resolutions",
        help="Adjust the camera resolutions to the optimal alignment.",
        action=argparse.BooleanOptionalAction,
    )
    # todo Add an option to set the lense position manually.
    # todo Add an option to disable autofocus.
    parser.add_argument(
        "--audio",
        help="Include audio with video stream.",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--autofocus-mode",
        choices=["auto", "continuous"],
        help="The autofocus mode.",
        default="auto",
    )
    parser.add_argument(
        "--autofocus-range",
        choices=["normal", "macro", "full"],
        help="The range of lens positions for which to attempt to autofocus.",
        default="full",
    )
    parser.add_argument(
        "--autofocus-speed",
        choices=["normal", "fast"],
        help="The speed with which to autofocus the lens.",
        default="fast",
    )
    parser.add_argument(
        "--bounding-box-scale-factor",
        help="The amount to scale the bounding box.",
        type=float,
        default=1.1,
    )
    parser.add_argument(
        "--buffers",
        help="The number of buffers to use.",
        type=int,
    )
    parser.add_argument(
        "--burst",
        help="The number of pictures to take after a successful detection.",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--capture-mode",
        choices=["still", "video"],
        help="Capture still images or video.",
        default="still",
    )
    parser.add_argument(
        "-c",
        "--config",
        is_config_file=True,
        help="The path to the config file to use.",
    )
    parser.add_argument(
        "--detection-threshold",
        help="The percentage confidence required for a detection.",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--draw-bounding-boxes",
        action=argparse.BooleanOptionalAction,
        help="Draw rectangles around detected objects",
    )
    parser.add_argument(
        "--encoder-quality",
        choices=["very-low", "low", "medium", "high", "very-high"],
        help="The quality preset to use for the video encoder.",
        default="very-high",
    )
    parser.add_argument(
        "--focal-detection-threshold",
        help="The percentage confidence required for the camera to focus on an object. Must be less than the value for --detection-threshold.",
        type=float,
    )
    parser.add_argument(
        "--frame-rate",
        help="The frame rate for video.",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--gap",
        help="The time to wait in between a successful detection and looking for the next detection. This gap is helpful for not capturing too many photographs of a detection, like when a capybara decides to take a nap in front of your camera.",
        default=0.05,
        type=float,
    )
    parser.add_argument(
        "--hailo",
        help="Use the Raspberry Pi AI accelerator.",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--hdr-mode",
        choices=["off", "single", "night"],
        help="The HDR mode.",
        default="single",
    )
    parser.add_argument("--label", help="Path of the labels file.", type=pathlib.Path)
    parser.add_argument(
        "--log-level", help="The log level, i.e. debug, info, warn etc.", default="warn"
    )
    parser.add_argument(
        "--low-resolution-width",
        help="The width to use for the low resolution size.",
        type=int,
    )
    parser.add_argument(
        "--low-resolution-height",
        help="The height to use for the low resolution size.",
        type=int,
    )
    parser.add_argument(
        "--main-resolution-width",
        help="The width to use for the main resolution size.",
        type=int,
    )
    parser.add_argument(
        "--main-resolution-height",
        help="The height to use for the main resolution size.",
        type=int,
    )
    parser.add_argument(
        "--match",
        help="A label for which to capture photographs. May be specified multiple times.",
        action="append",
        type=str,
    )
    parser.add_argument(
        "--model", help="Path of the detection model.", required=True, type=pathlib.Path
    )
    parser.add_argument(
        "--output", help="Directory path for the output images.", type=pathlib.Path
    )
    parser.add_argument(
        "--rtsp-tcp",
        action=argparse.BooleanOptionalAction,
        help="Stream to the RTSP server using TCP",
    )
    parser.add_argument(
        "--startup-capture",
        help="Take sample photographs when starting the program.",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--stream",
        help="The web address to which to stream video via RTSP like 'mediamtx.jwillikers.io:8554/detectionator'",
    )
    parser.add_argument(
        "--systemd-notify",
        help="Enable systemd-notify support for running as a systemd service.",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--threads",
        help="The number of threads to use to inference with TensorFlow.",
        default=len(os.sched_getaffinity(0)),
        type=int,
    )
    args = parser.parse_args()

    numeric_log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_log_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(numeric_log_level)
    logging.getLogger("picamera2").setLevel(numeric_log_level)

    if args.burst < 1:
        logger.warning(
            f"The burst value must be at least 1. Ignoring the provided burst value of '{args.burst}'."
        )
        args.burst = 1

    output_directory = os.path.join(os.getenv("HOME"), "Pictures")
    if args.output:
        output_directory = os.path.expanduser(args.output)
    if not os.path.isdir(output_directory):
        logger.info(f"The output directory '{output_directory}' does not exist")
        logger.info(f"Creating the output directory '{output_directory}'")
        try:
            os.mkdir(output_directory)
        except FileExistsError:
            pass

    args.model = os.path.expanduser(args.model)
    is_yolo = False
    if "yolov5" in pathlib.Path(args.model).stem:
        logger.info("Using YOLO model")
        is_yolo = True

    label_file = None
    if args.label:
        label_file = os.path.expanduser(args.label)

    labels = None
    if label_file:
        if args.hailo:
            with open(label_file, 'r', encoding="utf-8") as f:
                # class names
                labels = f.read().splitlines()
        else:
            labels = read_label_file(label_file)

    match = []
    if args.match:
        match = args.match

    if labels is not None:
        for m in match:
            if isinstance(labels, dict):
                if m not in labels.values():
                    logger.error(
                        f"The match '{m}' does not appear in the labels file {label_file}"
                    )
                    sys.exit(1)
            elif isinstance(labels, list):
                if m not in labels:
                    logger.error(
                        f"The match '{m}' does not appear in the labels file {label_file}"
                    )
                    sys.exit(1)

    logger.info(f"Will take photographs of: {match}")

    if not (args.detection_threshold > 0.0 and args.detection_threshold <= 1.0):
        logger.error("The detection threshold must be a value between 0.0 and 1.0.")
        sys.exit(1)

    focal_detection_threshold = args.focal_detection_threshold
    if args.focal_detection_threshold is None:
        focal_detection_threshold = args.detection_threshold

    if not (
        focal_detection_threshold > 0.0
        and focal_detection_threshold <= args.detection_threshold
    ):
        logger.error(
            f"The focal detection threshold must be a value between 0.0 and the detection threshold, {args.detection_threshold}."
        )
        sys.exit(1)

    if not (
        args.focal_detection_threshold > 0.0 and args.focal_detection_threshold <= 1.0
    ):
        logger.warning(
            "The focal detection threshold must be a value between 0.0 and 1.0."
        )

    autofocus_mode = (
        controls.AfModeEnum.Auto
        if args.autofocus_mode == "auto"
        else controls.AfModeEnum.Continuous
    )

    autofocus_speed = (
        controls.AfSpeedEnum.Fast
        if args.autofocus_speed == "fast"
        else controls.AfSpeedEnum.Normal
    )

    autofocus_range = controls.AfRangeEnum.Full
    if args.autofocus_range == "normal":
        autofocus_range = controls.AfRangeEnum.Normal
    elif args.autofocus_range == "macro":
        autofocus_range = controls.AfRangeEnum.Macro

    encoder_quality = Quality.VERY_HIGH
    if args.encoder_quality == "very-low":
        encoder_quality = Quality.VERY_LOW
    elif args.encoder_quality == "low":
        encoder_quality = Quality.LOW
    elif args.encoder_quality == "medium":
        encoder_quality = Quality.MEDIUM
    elif args.encoder_quality == "high":
        encoder_quality = Quality.HIGH

    hdr_mode = controls.HdrModeEnum.SingleExposure
    if args.hdr_mode == "off":
        hdr_mode = controls.HdrModeEnum.Off
    elif args.hdr_mode == "night":
        hdr_mode = controls.HdrModeEnum.Night

    number_of_cpu_cores = len(os.sched_getaffinity(0))
    if args.threads > number_of_cpu_cores:
        logger.warning(
            f"Requested {args.threads} threads which is more than {number_of_cpu_cores}, the total number of CPU cores available."
        )

    # Camera Module 3 has a full resolution of 4608x2592.
    # A scale of 2, really 1/2, results in a resolution of 2304x1296.
    # A scale of 4, really 1/4, results in a resolution of 1152x648.
    # A scale of 6, really 1/6, results in a resolution of 768x432.
    # A scale of 8, really 1/8, results in a resolution of 576x324 which is still pretty high resolution for close-up detections.
    # A scale of 12, really 1/12, results in a resolution of 384x216.
    # A scale of 16, really 1/16, results in a resolution of 288x162.
    # A scale of 32, really 1/32, results in a resolution of 144x81.
    default_low_resolution_scale = 8

    frame = int(time.time())

    interpreter = None
    if args.hailo:
        interpreter = Hailo(args.model)
    else:
        interpreter = tflite.Interpreter(
            model_path=str(args.model), num_threads=args.threads
        )
        interpreter.allocate_tensors()

    with Picamera2() as picam2:
        low_resolution_height: int = 0
        low_resolution_width: int = 0
        if args.hailo:
            low_resolution_height, low_resolution_width, _ = interpreter.get_input_shape()
        else:
            if args.low_resolution_width:
                low_resolution_width = args.low_resolution_width
            else:
                low_resolution_width = (
                    picam2.sensor_resolution[0] // default_low_resolution_scale
                )

            if args.low_resolution_height:
                low_resolution_height = args.low_resolution_height
            else:
                low_resolution_height = (
                    picam2.sensor_resolution[1] // default_low_resolution_scale
                )

        low_resolution = (low_resolution_width, low_resolution_height)

        if args.main_resolution_width:
            main_resolution_width = args.main_resolution_width
        else:
            if args.capture_mode == "video":
                main_resolution_width = 1920
            else:
                main_resolution_width = picam2.sensor_resolution[0]

        if args.main_resolution_height:
            main_resolution_height = args.main_resolution_height
        else:
            if args.capture_mode == "video":
                main_resolution_height = 1080
            else:
                main_resolution_height = picam2.sensor_resolution[1]

        main_resolution = (main_resolution_width, main_resolution_height)

        buffers = 2
        if args.buffers:
            buffers = args.buffers
        else:
            if args.capture_mode == "video":
                buffers = 4
            else:
                buffers = 2

        picam2.options["quality"] = 95
        picam2.options["compress_level"] = 0

        encoder = None
        config = None
        if args.capture_mode == "video":
            encoder = H264Encoder()
            config = picam2.create_video_configuration(
                # Use more buffers for a smoother video.
                # todo Large number of buffers may require dtoverlay=vc4-kms-v3d,cma-512 in /boot/firmware/config.txt
                buffer_count=buffers,
                # Minimize the time it takes to autofocus by setting the frame rate.
                # https://github.com/raspberrypi/picamera2/issues/884
                controls={
                    # todo Consider using a less stringent framerate for detections?
                    # Possible not good for streaming.
                    "FrameRate": args.frame_rate,
                    "HdrMode": hdr_mode,
                    # "NoiseReductionMode": controls.draft.NoiseReductionMode.Fast,
                    # "NoiseReductionMode": controls.draft.NoiseReductionMode.HighQuality,
                },
                # Don't display anything in the preview window since the system is running headless.
                display=None,
                main={
                    # I think this format needs to be "XRGB8888" for the H264 encoder.
                    # "format": "RGB888",
                    "size": main_resolution,
                    # 720p
                    # "size": (1280, 720),
                },
                lores={
                    # Only Pi 5 and newer can use formats besides YUV here.
                    # This avoids having to convert the image format for OpenCV later.
                    "format": "RGB888",
                    "size": low_resolution,
                },
            )
        else:
            config = picam2.create_still_configuration(
                # Using a buffer seems to reduce the latency between detections.
                buffer_count=buffers,
                # Minimize the time it takes to autofocus by setting the frame rate.
                # https://github.com/raspberrypi/picamera2/issues/884
                # controls={"FrameRate": 30},
                controls={
                    # todo Add config option for this. Likely, Night is also an important configuration choice.
                    # todo Set this to Night based off of GPS coordinates and sunset time or better yet a light sensor.
                    "HdrMode": hdr_mode,
                    # "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.HighQuality,
                },
                # Don't display anything in the preview window since the system is running headless.
                display=None,
                main={
                    "size": main_resolution,
                    "format": "RGB888",
                },
                lores={
                    # Only Pi 5 and newer can use formats besides YUV here.
                    # This avoids having to convert the image format for OpenCV later.
                    "format": "RGB888",
                    "size": low_resolution,
                },
            )
        if args.align_resolutions:
            picam2.align_configuration(config)
        low_resolution = config["lores"]["size"]
        main_resolution = config["main"]["size"]
        logger.info(f"Final low resolution: {low_resolution}")
        logger.info(f"Final main resolution: {main_resolution}")
        has_autofocus = "AfMode" in picam2.camera_controls
        picam2.configure(config)
        # Enable autofocus.
        if has_autofocus:
            picam2.set_controls(
                {
                    # todo Add option to control AfMetering mode.
                    "AfMetering": controls.AfMeteringEnum.Windows,
                    "AfMode": autofocus_mode,
                    "AfRange": autofocus_range,
                    "AfSpeed": autofocus_speed,
                }
            )
        scaler_crop_maximum = picam2.camera_properties["ScalerCropMaximum"]
        logger.info(
            f"ScalerCropMaximum: {rectangle_coordinate_width_height_to_string(scaler_crop_maximum)}"
        )

        scaler_crop_maximum_offset = (scaler_crop_maximum[0], scaler_crop_maximum[1])

        scaler_crop_maximum_ratio = (
            scaler_crop_maximum[2] / low_resolution[0],
            scaler_crop_maximum[3] / low_resolution[1],
        )

        if args.draw_bounding_boxes:
            # Apparently the windows used for AfWindows work best when they are between 1/4 and 1/12 the size of the total sensor area.
            # todo Scale the AfWindows values up or down accordingly.
            picam2.post_callback = partial(
                draw_bounding_boxes,
                resolution_scale=(
                    main_resolution[0] / low_resolution[0],
                    main_resolution[1] / low_resolution[1],
                ),
                resolution=main_resolution,
                scale_factor=args.bounding_box_scale_factor,
            )
        time.sleep(1)
        picam2.start()

        gps_exif_metadata = {}
        gps_mp4_metadata = {}

        # todo Do these handlers need to be setup inside the async context?
        # There seems to be extreme performance issues when triggering a sample video this way.

        # Take a sample photograph with both high and low resolution images for reference.
        async def capture_sample(gps_exif_metadata):
            timestamp = int(time.time())

            focus_cycle_job = None
            if has_autofocus:
                focus_cycle_job = picam2.autofocus_cycle(wait=False)
            exif_metadata = {}
            if gps_exif_metadata:
                exif_metadata["GPS"] = gps_exif_metadata
                logger.debug(f"Exif GPS metadata: {gps_exif_metadata}")
            else:
                logger.warning("No GPS fix")

            if has_autofocus:
                await asyncio.sleep(0)
                if not picam2.wait(focus_cycle_job):
                    logger.warning("Autofocus cycle failed.")
            picam2.capture_file(
                os.path.join(output_directory, f"low-res-sample-{timestamp}.jpg"),
                name="lores",
                exif_data=exif_metadata,
                format="jpeg",
            )
            picam2.capture_file(
                os.path.join(output_directory, f"high-res-sample-{timestamp}.jpg"),
                exif_data=exif_metadata,
                format="jpeg",
            )

        async def capture_sample_signal_handler():
            await capture_sample(gps_exif_metadata)

        # Record a five second video sample.
        async def record_sample(gps_mp4_metadata: dict):
            timestamp = int(time.time())
            focus_cycle_job = None
            if has_autofocus:
                focus_cycle_job = picam2.autofocus_cycle(wait=False)
            await asyncio.sleep(0)
            if has_autofocus:
                if not picam2.wait(focus_cycle_job):
                    logger.warning("Autofocus cycle failed.")
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            ffmpeg_command = f"-metadata:g creation_time={now} "
            file = os.path.join(output_directory, f"sample-recording-{timestamp}.mp4")
            if gps_mp4_metadata:
                ffmpeg_command += f"-metadata:g location={gps_mp4_metadata['longitude']}+{gps_mp4_metadata['latitude']}+{gps_mp4_metadata['altitude']} -metadata:g location-eng={gps_mp4_metadata['longitude']}+{gps_mp4_metadata['latitude']}+{gps_mp4_metadata['altitude']} "
                with open(file + ".xmp", "w") as xmp_file:
                    xmp_file.write(
                        xmp_xml(
                            gps_mp4_metadata["altitude"],
                            gps_mp4_metadata["latitude"],
                            gps_mp4_metadata["longitude"],
                        )
                    )
                logger.debug(f"MP4 GPS metadata: {gps_mp4_metadata}")
            else:
                logger.warning("No GPS fix")
            ffmpeg_command += file
            output = FfmpegOutput(ffmpeg_command, audio=args.audio)
            encoder_outputs = encoder.output
            if not isinstance(encoder_outputs, list):
                encoder_outputs = [encoder_outputs]
            encoder.output = [output] + encoder_outputs
            encoder_running = encoder.running
            if not encoder_running:
                picam2.start_encoder(encoder, quality=Quality.VERY_HIGH)
            output.start()
            await asyncio.sleep(5)
            output.stop()
            if not encoder_running:
                picam2.stop_encoder(encoder)
            encoder.output = encoder_outputs

        async def record_sample_signal_handler():
            await record_sample(gps_mp4_metadata)

        loop = asyncio.get_event_loop()

        async def interrupt_signal_handler():
            logger.info("You pressed Ctrl+C!")
            picam2.stop()
            loop.stop()

        loop.add_signal_handler(
            signal.SIGINT, lambda: asyncio.create_task(interrupt_signal_handler())
        )
        if args.capture_mode == "video":
            loop.add_signal_handler(
                signal.SIGUSR1,
                lambda: asyncio.create_task(record_sample_signal_handler()),
            )
        else:
            loop.add_signal_handler(
                signal.SIGUSR1,
                lambda: asyncio.create_task(capture_sample_signal_handler()),
            )

        if args.startup_capture:
            if args.capture_mode == "video":
                record_sample(gps_mp4_metadata)
            else:
                capture_sample(gps_exif_metadata)

        if args.stream:
            ffmpeg_command = "-f rtsp"
            if args.rtsp_tcp:
                ffmpeg_command += " -rtsp_transport tcp"
            ffmpeg_command += f" rtsp://{args.stream}"
            output = FfmpegOutput(
                output_filename=ffmpeg_command,
                # todo Include audio?
                audio=False,
            )
            encoder_outputs = encoder.output
            if not isinstance(encoder_outputs, list):
                encoder_outputs = [encoder_outputs]
            encoder.output = [output] + encoder_outputs
            if not encoder.running:
                picam2.start_encoder(encoder, quality=encoder_quality)
            output.start()

        try:
            async with gps.aiogps.aiogps(
                connection_timeout=1,
                reconnect=5,
            ) as gpsd:
                systemd_notifier = None
                if args.systemd_notify:
                    systemd_notifier = sdnotify.SystemdNotifier()
                    systemd_notifier.notify("READY=1")
                    systemd_notifier.notify(f"STATUS=Looking for {match}")
                if args.capture_mode == "still":
                    await asyncio.gather(
                        update_gps_exif_metadata(gpsd, gps_exif_metadata),
                        detect_and_capture(
                            picam2=picam2,
                            interpreter=interpreter,
                            labels=labels,
                            match=match,
                            gps_exif_metadata=gps_exif_metadata,
                            scaler_crop_maximum_ratio=scaler_crop_maximum_ratio,
                            scaler_crop_maximum_offset=scaler_crop_maximum_offset,
                            main_resolution=main_resolution,
                            has_autofocus=has_autofocus,
                            burst=args.burst,
                            output_directory=output_directory,
                            frame=frame,
                            gap=args.gap,
                            detection_threshold=args.detection_threshold,
                            focal_detection_threshold=focal_detection_threshold,
                            is_yolo=is_yolo,
                            bounding_box_scale_factor=args.bounding_box_scale_factor,
                        ),
                        return_exceptions=True,
                    )
                else:
                    await asyncio.gather(
                        update_gps_mp4_metadata(gpsd, gps_mp4_metadata),
                        detect_and_record(
                            picam2=picam2,
                            interpreter=interpreter,
                            labels=labels,
                            match=match,
                            gps_mp4_metadata=gps_mp4_metadata,
                            scaler_crop_maximum_ratio=scaler_crop_maximum_ratio,
                            scaler_crop_maximum_offset=scaler_crop_maximum_offset,
                            main_resolution=main_resolution,
                            has_autofocus=has_autofocus,
                            output_directory=output_directory,
                            frame=frame,
                            gap=args.gap,
                            encoder=encoder,
                            encoder_quality=encoder_quality,
                            audio=args.audio,
                            detection_threshold=args.detection_threshold,
                            focal_detection_threshold=focal_detection_threshold,
                            is_yolo=is_yolo,
                            bounding_box_scale_factor=args.bounding_box_scale_factor,
                        ),
                        return_exceptions=True,
                    )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error(f"Error: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
