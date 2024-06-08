#!/usr/bin/env python3
import argparse
import asyncio
import datetime
from dateutil import parser
from functools import partial
import logging
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
from picamera2 import Picamera2
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


def read_label_file(file_path):
    with open(file_path, "r", encoding="UTF-8") as f:
        lines = f.readlines()
    ret = {}
    for line in lines:
        pair = line.strip().split(maxsplit=1)
        ret[int(pair[0])] = pair[1].strip()
    return ret


def inference_tensorflow(image, model, labels, match_labels: list):
    interpreter = tflite.Interpreter(model_path=str(model), num_threads=4)
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
                logger.debug(f"label = {labels[classId]}, score = {score}")
            else:
                logger.debug(f"score = {score}")
            matches.append(match)
    return matches


def scale(coord, scaler_crop_maximum, lores):
    x_offset, y_offset, width, height = coord

    # scaler_crop_maximum represents a larger image than the image we use so we need to account
    x_offset_scm, y_offset_scm, _width_scm, _height_scm = scaler_crop_maximum

    # create a scale so that you can scale the preview to the SCM
    y_scale = scaler_crop_maximum[3] / lores[1]
    x_scale = scaler_crop_maximum[2] / lores[0]
    logger.debug("y_scale, x_scale", y_scale, x_scale)

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


def captured_file(filename: str, matches, job):
    if job:
        logger.info(f"Captured image '{filename}': {matches}")
    else:
        logger.error(f"Failed to capture image '{filename}': {matches}")


async def detect_and_capture(
    picam2,
    model,
    labels,
    match,
    gps_exif_metadata: dict,
    scaler_crop_maximum,
    low_resolution_width,
    low_resolution_height,
    has_autofocus,
    burst: bool,
    output_directory: pathlib.Path,
    frame: int,
    gap: float,
):
    while True:
        image = picam2.capture_array("lores")
        matches = inference_tensorflow(image, model, labels, match)
        if len(matches) == 0:
            # Take a quick breather to give the CPU a break.
            # 1/5 of a second results in about 50% CPU usage.
            # 1/10 of a second results in about 80% CPU usage.
            # 1/20 of a second results in about 130% CPU usage.
            # todo Increase / decrease this wait based on recent detections.
            await asyncio.sleep(0.075)
            continue

        # Autofocus
        match_boxes = [m[1] for m in sorted(matches, key=lambda x: x[0], reverse=True)]
        adjusted_match_boxes = []
        for match_box in match_boxes:
            adjusted_match_boxes.append(
                scale(
                    (
                        match_box[0],
                        match_box[1],
                        abs(match_box[2] - match_box[0]),
                        abs(match_box[3] - match_box[1]),
                    ),
                    scaler_crop_maximum,
                    (low_resolution_width, low_resolution_height),
                )
            )
        picam2.set_controls({"AfWindows": adjusted_match_boxes})
        focus_cycle_job = None
        if has_autofocus:
            focus_cycle_job = picam2.autofocus_cycle(wait=False)

        exif_metadata = {}
        if gps_exif_metadata:
            exif_metadata["GPS"] = gps_exif_metadata
            logger.debug(f"Exif GPS metadata: {gps_exif_metadata}")
        else:
            logger.warning("No GPS fix")

        matches_name = "detection"
        if labels:
            matches_name = "-".join([i[2] for i in matches])
        filename = os.path.join(output_directory, f"{matches_name}-{frame}.jpg")
        if has_autofocus:
            if not picam2.wait(focus_cycle_job):
                logger.warning("Autofocus cycle failed.")
        picam2.capture_file(
            filename,
            exif_data=exif_metadata,
            format="jpeg",
            signal_function=partial(captured_file, filename, matches),
        )
        frame += 1

        # Capture burst photographs.
        for _ in range(burst - 1):
            focus_cycle_job = None
            if has_autofocus:
                focus_cycle_job = picam2.autofocus_cycle(wait=False)
            filename = os.path.join(output_directory, f"{matches_name}-{frame}.jpg")
            if has_autofocus:
                if not picam2.wait(focus_cycle_job):
                    logger.warning("Autofocus cycle failed.")
            picam2.capture_file(
                filename,
                exif_data=exif_metadata,
                format="jpeg",
                signal_function=partial(captured_file, filename, matches),
            )
            frame += 1
        await asyncio.sleep(gap)


async def detect_and_record(
    picam2,
    model,
    labels,
    match,
    gps_mp4_metadata: dict,
    scaler_crop_maximum,
    low_resolution_width,
    low_resolution_height,
    has_autofocus,
    autofocus_speed,
    output_directory: pathlib.Path,
    frame: int,
    gap: float,
    encoder,
    audio: bool,
):
    while True:
        image = picam2.capture_array("lores")
        matches = inference_tensorflow(image, model, labels, match)
        if len(matches) == 0:
            await asyncio.sleep(0.2)
            continue

        # Autofocus
        match_boxes = [m[1] for m in sorted(matches, key=lambda x: x[0], reverse=True)]
        adjusted_match_boxes = []
        for match_box in match_boxes:
            adjusted_match_boxes.append(
                scale(
                    (
                        match_box[0],
                        match_box[1],
                        abs(match_box[2] - match_box[0]),
                        abs(match_box[3] - match_box[1]),
                    ),
                    scaler_crop_maximum,
                    (low_resolution_width, low_resolution_height),
                )
            )
        picam2.set_controls({"AfWindows": adjusted_match_boxes})
        focus_cycle_job = None
        if has_autofocus:
            focus_cycle_job = picam2.autofocus_cycle(wait=False)

        matches_name = "detection"
        if labels:
            matches_name = "-".join([i[2] for i in matches])
        file = os.path.join(output_directory, f"{matches_name}-{frame}.mp4")
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
            picam2.start_encoder(encoder, quality=Quality.VERY_HIGH)

        if has_autofocus:
            if not picam2.wait(focus_cycle_job):
                logger.warning("Autofocus cycle failed.")

        output.start()

        await asyncio.sleep(0.2)

        minimum_record_seconds = 6
        consecutive_failed_detections = 0
        consecutive_failed_detections_to_stop = 5
        last_detection_time = datetime.datetime.now()
        while (
            (datetime.datetime.now() - last_detection_time).seconds
            <= minimum_record_seconds
        ) or consecutive_failed_detections < consecutive_failed_detections_to_stop:
            # Autofocus
            image = picam2.capture_array("lores")
            matches = inference_tensorflow(image, model, labels, match)
            if len(matches) == 0:
                consecutive_failed_detections += 1
                await asyncio.sleep(0.4)
            else:
                # Autofocus
                match_boxes = [
                    m[1] for m in sorted(matches, key=lambda x: x[0], reverse=True)
                ]
                adjusted_match_boxes = []
                for match_box in match_boxes:
                    adjusted_match_boxes.append(
                        scale(
                            (
                                match_box[0],
                                match_box[1],
                                abs(match_box[2] - match_box[0]),
                                abs(match_box[3] - match_box[1]),
                            ),
                            scaler_crop_maximum,
                            (low_resolution_width, low_resolution_height),
                        )
                    )
                picam2.set_controls({"AfWindows": adjusted_match_boxes})
                focus_cycle_job = None
                if has_autofocus:
                    focus_cycle_job = picam2.autofocus_cycle(wait=False)
                if has_autofocus:
                    if not picam2.wait(focus_cycle_job):
                        logger.warning("Autofocus cycle failed.")
                consecutive_failed_detections = 0
                await asyncio.sleep(0.2)
                last_detection_time = datetime.datetime.now()
        output.stop()
        if not encoder_running:
            picam2.stop_encoder(encoder)
        encoder.output = encoder_outputs
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
        "--audio",
        help="Include audio with video stream.",
        action=argparse.BooleanOptionalAction,
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
        "--gap",
        help="The time to wait in between a successful detection and looking for the next detection. This gap is helpful for not capturing too many photographs of a detection, like when a capybara decides to take a nap in front of your camera.",
        default=0.05,
        type=float,
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
        "--startup-capture",
        help="Take sample photographs when starting the program.",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--stream",
        help="The web address to which to stream video via RTSP like 'mediamtx.jwillikers.io:8554/detectionator'",
    )
    parser.add_argument(
        "--rtsp-tcp",
        action=argparse.BooleanOptionalAction,
        help="Stream to the RTSP server using TCP",
    )
    parser.add_argument(
        "--systemd-notify",
        help="Enable systemd-notify support for running as a systemd service.",
        action=argparse.BooleanOptionalAction,
    )
    args = parser.parse_args()

    numeric_log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_log_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    logger.setLevel(numeric_log_level)
    logging.getLogger("picamera2").setLevel(numeric_log_level)

    if args.burst < 1:
        logger.warn(
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

    label_file = None
    if args.label:
        label_file = os.path.expanduser(args.label)

    labels = None
    if label_file:
        labels = read_label_file(label_file)

    match = []
    if args.match:
        match = args.match

    if labels is not None:
        for m in match:
            if m not in labels.values():
                logger.error(
                    f"The match '{m}' does not appear in the labels file {label_file}"
                )
                sys.exit(1)

    logger.info(f"Will take photographs of: {match}")

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

    # Camera Module 3 has a full resolution of 4608x2592.
    # A scale of 6, really 1/6, results in a resolution of 768x432.
    # A scale of 8, really 1/8, results in a resolution of 576x324 which is still pretty high resolution for close-up detections.
    # A scale of 12, really 1/12, results in a resolution of 384x216.
    # A scale of 16, really 1/16, results in a resolution of 288x162.
    # A scale of 32, really 1/32, results in a resolution of 144x81.
    default_low_resolution_scale = 8

    frame = int(time.time())

    with Picamera2() as picam2:
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

        if (
            picam2.sensor_resolution[0] / low_resolution_width
            != picam2.sensor_resolution[1] / low_resolution_height
        ):
            logger.error(
                f"The low resolution width, '{low_resolution_width}', and low resolution height, '{low_resolution_height}' must be a fraction of the resolution, '{picam2.sensor_resolution}'"
            )
            sys.exit(1)

        picam2.options["quality"] = 95
        picam2.options["compress_level"] = 0

        encoder = None
        config = None
        if args.capture_mode == "video":
            encoder = H264Encoder()
            config = picam2.create_video_configuration(
                # Use more buffers for a smoother video.
                # todo Large number of buffers may require dtoverlay=vc4-kms-v3d,cma-512 in /boot/firmware/config.txt
                buffer_count=8,
                # Minimize the time it takes to autofocus by setting the frame rate.
                # https://github.com/raspberrypi/picamera2/issues/884
                controls={
                    # todo Consider using a less stringent framerate for detections?
                    # Possible not good for streaming.
                    "FrameRate": 30,
                    "HdrMode": controls.HdrModeEnum.SingleExposure,
                    # "NoiseReductionMode": controls.draft.NoiseReductionMode.Fast,
                },
                # Don't display anything in the preview window since the system is running headless.
                display=None,
                main={
                    # I think this format needs to be "XRGB8888" for the H264 encoder.
                    # "format": "RGB888",
                    "size": (1920, 1080),
                    # 720p
                    # "size": (1280, 720),
                },
                lores={
                    # Only Pi 5 and newer can use formats besides YUV here.
                    # This avoids having to convert the image format for OpenCV later.
                    "format": "RGB888",
                    "size": (low_resolution_width, low_resolution_height),
                },
            )
        else:
            config = picam2.create_still_configuration(
                # Using a buffer seems to reduce the latency between detections.
                buffer_count=2,
                # Minimize the time it takes to autofocus by setting the frame rate.
                # https://github.com/raspberrypi/picamera2/issues/884
                # controls={"FrameRate": 30},
                controls={
                    # todo Add config option for this. Likely, Night is also an important configuration choice.
                    # todo Set this to Night based off of GPS coordinates and sunset time or better yet a light sensor.
                    "HdrMode": controls.HdrModeEnum.SingleExposure,
                    # "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.HighQuality,
                },
                # Don't display anything in the preview window since the system is running headless.
                display=None,
                main={
                    "format": "RGB888",
                },
                lores={
                    # Only Pi 5 and newer can use formats besides YUV here.
                    # This avoids having to convert the image format for OpenCV later.
                    "format": "RGB888",
                    "size": (low_resolution_width, low_resolution_height),
                },
            )
        # todo See how forcing the streams into alignment effects performance.
        # todo Make this a configuration option, enabled by default.
        picam2.align_configuration(config)
        low_resolution_size = config["lores"]["size"]
        main_resolution_size = config["main"]["size"]
        logger.info(f"Low resolution: {low_resolution_size}")
        logger.info(f"Main resolution: {main_resolution_size}")
        has_autofocus = "AfMode" in picam2.camera_controls
        picam2.configure(config)
        # Enable autofocus.
        if has_autofocus:
            picam2.set_controls(
                {
                    "AfMetering": controls.AfMeteringEnum.Windows,
                    "AfMode": controls.AfModeEnum.Auto,
                    # todo Test continuous autofocus.
                    # "AfMode": controls.AfModeEnum.Continuous,
                    "AfRange": autofocus_range,
                    "AfSpeed": autofocus_speed
                    if args.capture_mode == "still"
                    else controls.AfSpeedEnum.Normal,
                }
            )
        scaler_crop_maximum = picam2.camera_properties["ScalerCropMaximum"]
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

            await asyncio.sleep(0)
            if has_autofocus:
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
                picam2.start_encoder(encoder, quality=Quality.VERY_HIGH)
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
                            model=args.model,
                            labels=labels,
                            match=match,
                            gps_exif_metadata=gps_exif_metadata,
                            scaler_crop_maximum=scaler_crop_maximum,
                            low_resolution_width=low_resolution_width,
                            low_resolution_height=low_resolution_height,
                            has_autofocus=has_autofocus,
                            burst=args.burst,
                            output_directory=output_directory,
                            frame=frame,
                            gap=args.gap,
                        ),
                        return_exceptions=True,
                    )
                else:
                    await asyncio.gather(
                        update_gps_mp4_metadata(gpsd, gps_mp4_metadata),
                        detect_and_record(
                            picam2=picam2,
                            model=args.model,
                            labels=labels,
                            match=match,
                            gps_mp4_metadata=gps_mp4_metadata,
                            scaler_crop_maximum=scaler_crop_maximum,
                            low_resolution_width=low_resolution_width,
                            low_resolution_height=low_resolution_height,
                            has_autofocus=has_autofocus,
                            autofocus_speed=autofocus_speed,
                            output_directory=output_directory,
                            frame=frame,
                            gap=args.gap,
                            encoder=encoder,
                            audio=args.audio,
                        ),
                        return_exceptions=True,
                    )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error(f"Error: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
