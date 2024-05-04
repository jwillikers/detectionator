#!/usr/bin/env python3
import argparse
from functools import partial
import logging
import os
import signal
import sys
import time

import gps
import cv2
from libcamera import controls
import numpy as np
from picamera2 import Picamera2
import piexif
import tflite_runtime.interpreter as tflite

from exif_utils import (
    degrees_decimal_to_degrees_minutes_seconds,
    number_to_exif_rational,
)


def read_label_file(file_path):
    with open(file_path, "r", encoding="UTF-8") as f:
        lines = f.readlines()
    ret = {}
    for line in lines:
        pair = line.strip().split(maxsplit=1)
        ret[int(pair[0])] = pair[1].strip()
    return ret


def inference_tensorflow(image, model, labels, match_labels: list):
    interpreter = tflite.Interpreter(model_path=model, num_threads=4)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]["shape"][1]
    width = input_details[0]["shape"][2]
    floating_model = False
    if input_details[0]["dtype"] == np.float32:
        floating_model = True

    picture = cv2.resize(image, (width, height))
    initial_h, initial_w, _channels = picture.shape

    input_data = np.expand_dims(picture, axis=0)
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    interpreter.set_tensor(input_details[0]["index"], input_data)

    interpreter.invoke()

    detected_boxes = interpreter.get_tensor(output_details[0]["index"])
    detected_classes = interpreter.get_tensor(output_details[1]["index"])
    detected_scores = interpreter.get_tensor(output_details[2]["index"])
    num_boxes = interpreter.get_tensor(output_details[3]["index"])

    matches = list()
    # match:
    #   score
    #   rectangle
    #   label
    for i in range(int(num_boxes.item())):
        top, left, bottom, right = detected_boxes[0][i]
        classId = int(detected_classes[0][i])
        if match_labels and labels[classId] not in match_labels:
            continue
        score = detected_scores[0][i]
        if score > 0.5:
            xmin = left * initial_w
            ymin = bottom * initial_h
            xmax = right * initial_w
            ymax = top * initial_h
            box = [xmin, ymin, xmax, ymax]
            match = (score, box)
            if labels:
                match = (*match, labels[classId])
                logging.debug(f"label = {labels[classId]}, score = {score}")
            else:
                logging.debug(f"score = {score}")
            matches.append(match)
    return matches


def scale(coord, scaler_crop_maximum, lores):
    x_offset, y_offset, width, height = coord

    # scaler_crop_maximum represents a larger image than the image we use so we need to account
    x_offset_scm, y_offset_scm, _width_scm, _height_scm = scaler_crop_maximum

    # create a scale so that you can scale the preview to the SCM
    y_scale = scaler_crop_maximum[3] / lores[1]
    x_scale = scaler_crop_maximum[2] / lores[0]
    print("y_scale, x_scale", y_scale, x_scale)

    # scale coords to SCM
    y_offset_scaled = int(y_offset * y_scale)
    height_scaled = int(height * y_scale)
    x_offset_scaled = int(x_offset * x_scale)
    width_scaled = int(width * x_scale)

    return (
        x_offset_scm + x_offset_scaled,
        y_offset_scm + y_offset_scaled,
        width_scaled,
        height_scaled,
    )


# Retrieve and format the data from the GPS for EXIF.
def get_gps_exif_metadata(session: gps.gps) -> dict:
    if session.read() != 0 and not (gps.MODE_SET & session.valid):
        return {}

    fix_mode = session.fix.mode
    if fix_mode in [0, gps.MODE_NO_FIX]:
        return {}

    if gps.isfinite(session.fix.latitude):
        latitude = degrees_decimal_to_degrees_minutes_seconds(session.fix.latitude)
    if gps.isfinite(session.fix.longitude):
        longitude = degrees_decimal_to_degrees_minutes_seconds(session.fix.longitude)

    # if gps.ALTITUDE_SET & session.valid
    altitude = session.fix.altHAE

    gps_ifd = {
        piexif.GPSIFD.GPSAltitude: number_to_exif_rational(
            abs(0 if altitude is None else altitude)
        ),
        piexif.GPSIFD.GPSAltitudeRef: (0 if altitude is None or altitude > 0 else 1),
        piexif.GPSIFD.GPSLatitude: (
            number_to_exif_rational(abs(latitude[0])),
            number_to_exif_rational(abs(latitude[1])),
            number_to_exif_rational(abs(latitude[2])),
        ),
        piexif.GPSIFD.GPSLatitudeRef: "N" if latitude[0] > 0 else "S",
        piexif.GPSIFD.GPSLongitude: (
            number_to_exif_rational(abs(longitude[0])),
            number_to_exif_rational(abs(longitude[1])),
            number_to_exif_rational(abs(longitude[2])),
        ),
        piexif.GPSIFD.GPSLongitudeRef: "E" if longitude[0] > 0 else "W",
        piexif.GPSIFD.GPSProcessingMethod: "GPS".encode("ASCII"),
        piexif.GPSIFD.GPSSatellites: str(session.satellites_used),
        piexif.GPSIFD.GPSSpeed: (
            number_to_exif_rational(0)
            if session.fix.speed is None
            # Convert m/sec to km/hour
            else number_to_exif_rational(session.fix.speed * 3.6)
        ),
        piexif.GPSIFD.GPSSpeedRef: "K",
        piexif.GPSIFD.GPSVersionID: (2, 3, 0, 0),
    }
    gps_ifd[piexif.GPSIFD.GPSMeasureMode] = str(fix_mode)
    if gps.TIME_SET & session.valid:
        gps_ifd[piexif.GPSIFD.GPSDateStamp] = time.strftime(
            "%Y:%m:%d", session.fix.time
        )
        gps_ifd[piexif.GPSIFD.GPSTimeStamp] = (
            number_to_exif_rational(gps.timestamp_utc.tm_hour),
            number_to_exif_rational(gps.timestamp_utc.tm_min),
            number_to_exif_rational(gps.timestamp_utc.tm_sec),
        )
    gps_ifd[piexif.GPSIFD.GPSStatus] = session.fix.status
    return gps_ifd


def captured_file(filename: str, matches, job):
    if job:
        logging.info(f"Captured image '{filename}': {matches}")
    else:
        logging.error(f"Failed to capture image '{filename}': {matches}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap", help="The time to wait between pictures.", default=0.2)
    parser.add_argument("--label", help="Path of the labels file.")
    parser.add_argument(
        "--log-level", help="The log level, i.e. debug, info, warn etc.", default="warn"
    )
    parser.add_argument(
        "--low-resolution-width",
        help="The width to use for the low resolution size.",
    )
    parser.add_argument(
        "--low-resolution-height",
        help="The height to use for the low resolution size.",
    )
    parser.add_argument(
        "--match", help="The labels for which to capture photographs", nargs="*"
    )
    parser.add_argument("--model", help="Path of the detection model.", required=True)
    parser.add_argument("--output", help="Directory path for the output images.")
    args = parser.parse_args()

    numeric_log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_log_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    logging.basicConfig(level=numeric_log_level)

    output_directory = os.path.join(os.getenv("HOME"), "Pictures")
    if args.output:
        output_directory = args.output
    if not os.path.isdir(output_directory):
        logging.info(f"The output directory '{output_directory}' does not exist")
        logging.info(f"Creating the output directory '{output_directory}'")
        try:
            os.mkdir(output_directory)
        except FileExistsError:
            pass

    label_file = None
    if args.label:
        label_file = args.label

    labels = None
    if label_file:
        labels = read_label_file(label_file)

    match = []
    if args.match:
        match = args.match

    if labels is not None:
        for m in match:
            if m not in labels.values():
                logging.error(
                    f"The match '{m}' does not appear in the labels file {label_file}"
                )
                sys.exit(1)

    logging.info(f"Will take photographs of: {match}")

    # Initialize the GPS
    gps_session = gps.gps(mode=gps.WATCH_ENABLE)

    frame = int(time.time())
    with Picamera2() as picam2:
        if args.low_resolution_width:
            low_resolution_width = args.low_resolution_width
        else:
            low_resolution_width = picam2.sensor_resolution[0] // 2

        if args.low_resolution_height:
            low_resolution_height = args.low_resolution_height
        else:
            low_resolution_height = picam2.sensor_resolution[1] // 2

        if (
            picam2.sensor_resolution[0] / low_resolution_width
            != picam2.sensor_resolution[1] / low_resolution_height
        ):
            logging.error(
                f"The low resolution width, '{low_resolution_width}', and low resolution height, '{low_resolution_height}' must be a fraction of the resolution, '{picam2.sensor_resolution}'"
            )
            sys.exit(1)

        picam2.options["quality"] = 95
        picam2.options["compress_level"] = 0

        config = picam2.create_still_configuration(
            buffer_count=4,
            # Minimize the time it takes to autofocus by setting the frame rate.
            # https://github.com/raspberrypi/picamera2/issues/884
            # controls={'FrameRate': 30},
            # Don't display anything in the preview window since the system is running headless.
            display=None,
            lores={
                # Only Pi 5 and newer can use formats besides YUV here.
                # This avoids having to convert the image format for OpenCV later.
                "format": "RGB888",
                "size": (low_resolution_width, low_resolution_height),
            },
        )
        picam2.configure(config)
        # Enable autofocus.
        if "AfMode" in picam2.camera_controls:
            # todo Test continuous autofocus.
            # picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
            picam2.set_controls({"AfMode": controls.AfModeEnum.Auto})
        scaler_crop_maximum = picam2.camera_properties["ScalerCropMaximum"]
        time.sleep(1)
        picam2.start()

        def signal_handler(_sig, _frame):
            logging.info("You pressed Ctrl+C!")
            picam2.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        while True:
            # Take a quick breather to give the CPU a break.
            time.sleep(0.1)

            image = picam2.capture_array("lores")
            matches = inference_tensorflow(image, args.model, labels, match)
            if len(matches) == 0:
                continue
            # Autofocus
            best_match = sorted(matches, key=lambda x: x[0], reverse=True)[0]
            adjusted_focal_point = scale(
                best_match[1],
                scaler_crop_maximum,
                (low_resolution_width, low_resolution_height),
            )
            picam2.set_controls({"AfWindows": [adjusted_focal_point]})
            focus_cycle_job = None
            if "AfMode" in picam2.camera_controls:
                focus_cycle_job = picam2.autofocus_cycle(wait=False)

            exif_metadata = {}
            gps_exif_metadata = get_gps_exif_metadata(gps_session)
            if gps_exif_metadata:
                exif_metadata["GPS"] = gps_exif_metadata
                logging.debug(f"Exif GPS metadata: {gps_exif_metadata}")
            else:
                logging.warning("No GPS fix")

            matches_name = "detection"
            if labels:
                matches_name = "-".join([i[2] for i in matches])
            filename = os.path.join(output_directory, f"{matches_name}-{frame}.jpg")
            if "AfMode" in picam2.camera_controls:
                if not picam2.wait(focus_cycle_job):
                    logging.warning("Autofocus cycle failed.")
            picam2.capture_file(
                filename,
                exif_data=exif_metadata,
                format="jpeg",
                signal_function=partial(captured_file, filename, matches),
            )
            frame += 1
            time.sleep(args.gap)


if __name__ == "__main__":
    main()
