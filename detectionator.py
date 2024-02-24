#!/usr/bin/env python3
import argparse
from fractions import Fraction
import os
import signal
import sys
import time

import adafruit_gps
import cv2
from libcamera import controls
import numpy as np
from picamera2 import Picamera2
import piexif
import serial
import tflite_runtime.interpreter as tflite


# normal_size = (640, 480)
# low_resolution_size = (320, 240)

# Camera Module 3: 4608 x 2592 
normal_size = (4608, 2592)
# low_resolution_size = (576, 324)
# low_resolution_size = (1152, 648)
low_resolution_size = (2304, 1296)

def ReadLabelFile(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    ret = {}
    for line in lines:
        pair = line.strip().split(maxsplit=1)
        ret[int(pair[0])] = pair[1].strip()
    return ret

def InferenceTensorFlow(image, model, labels, match_labels: list):
    interpreter = tflite.Interpreter(model_path=model, num_threads=4)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    floating_model = False
    if input_details[0]['dtype'] == np.float32:
        floating_model = True

    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    picture = cv2.resize(rgb, (width, height))

    input_data = np.expand_dims(picture, axis=0)
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    detected_classes = interpreter.get_tensor(output_details[1]['index'])
    detected_scores = interpreter.get_tensor(output_details[2]['index'])
    num_boxes = interpreter.get_tensor(output_details[3]['index'])

    matches = set()
    for i in range(int(num_boxes)):
        classId = int(detected_classes[0][i])
        if match_labels and labels[classId] not in match_labels:
            continue
        score = detected_scores[0][i]
        if score > 0.5:
            if labels:
                print(labels[classId], 'score = ', score)
            else:
                print('score = ', score)
            matches.add(labels[classId])
    return matches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gap', help='The time to wait between pictures.', default=1)
    parser.add_argument('--label', help='Path of the labels file.')
    parser.add_argument('--match', help='The labels for which to capture photographs', nargs='*')
    parser.add_argument('--model', help='Path of the detection model.', required=True)
    parser.add_argument('--output', help='Directory path for the output images.')
    args = parser.parse_args()

    output_directory = os.path.join(os.getenv("HOME"), "Pictures")
    if (args.output):
        output_directory = args.output
    if not os.path.isdir(output_directory):
        print(f"The output directory '{output_directory}' does not exist")
        print(f"Creating the output directory '{output_directory}'")
        try:
            os.mkdir(output_directory)
        except FileExistsError:
            pass

    label_file = None
    if (args.label):
        label_file = args.label

    labels = None
    if label_file:
        labels = ReadLabelFile(label_file)

    match = []
    if (args.match):
        match = args.match

    if labels is not None:
        for m in match:
            if m not in labels.values():
                print(f"The match '{m}' does not appear in the labels file {label_file}")
                exit(1)

    print(f"Will take photographs of: {match}")

    # Initialize the GPS
    # todo Use a static udev alias name for the GPS serial device.
    uart = serial.Serial("/dev/ttyUSB0", baudrate=9600, timeout=10)
    gps = adafruit_gps.GPS(uart, debug=False)
    gps.send_command(b"PMTK314,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0")
    gps.send_command(b"PMTK220,1000")
    time.sleep(0.5)
    gps.update()
    time.sleep(0.5)
    gps.update()

    frame = int(time.time())
    with Picamera2() as picam2:
        # Enable autofocus.
        # if "AfMode" in picam2.camera_controls:
        picam2.set_controls({"AfMode": controls.AfModeEnum.Auto})
        time.sleep(1)

        picam2.options["quality"] = 95
        picam2.options["compress_level"] = 9
        config = picam2.create_preview_configuration(main={"size": normal_size}, lores={"size": low_resolution_size, "format": "YUV420"})
        picam2.configure(config)
        stride = picam2.stream_configuration("lores")["stride"]
        picam2.start()

        def signal_handler(sig, frame):
            print('You pressed Ctrl+C!')
            picam2.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        while True:
            buffer = picam2.capture_buffer("lores")
            grey = buffer[:stride * low_resolution_size[1]].reshape((low_resolution_size[1], stride))
            matches = InferenceTensorFlow(grey, args.model, labels, match)
            if len(matches) == 0:
                continue
            gps.update()
            # Autofocus
            # if 'AfMode' in picam2.camera_controls:
            for _ in range(5):
                if picam2.autofocus_cycle():
                    break
            exif_dict={}
            if gps.update() and gps.has_fix:
                def decimal_minutes_to_minutes_seconds(minutes_decimal: float) -> tuple[int, Fraction]:
                    minutes = int(minutes_decimal)
                    seconds = Fraction((minutes_decimal - minutes) * 60).limit_denominator(100)
                    return (minutes, seconds)

                latitude_minutes, latitude_seconds = decimal_minutes_to_minutes_seconds(gps.latitude_minutes)
                longitude_minutes, longitude_seconds = decimal_minutes_to_minutes_seconds(gps.longitude_minutes)
                
                def format_fraction(fraction: Fraction) -> tuple[int, int]:
                    return (int(fraction.limit_denominator(100).numerator), int(fraction.limit_denominator(100).denominator))

                gps_ifd = {
                    piexif.GPSIFD.GPSVersionID: (2, 3, 0, 0),
                    piexif.GPSIFD.GPSAltitude: format_fraction(Fraction(abs(0 if gps.altitude_m is None else gps.altitude_m)).limit_denominator(100)),
                    piexif.GPSIFD.GPSAltitudeRef: (0 if gps.altitude_m is None or gps.altitude_m > 0 else 1),
                    piexif.GPSIFD.GPSLatitude: ((abs(gps.latitude_degrees), 1), (latitude_minutes, 1), format_fraction(latitude_seconds)),
                    piexif.GPSIFD.GPSLatitudeRef: "N" if gps.latitude > 0 else "S",
                    piexif.GPSIFD.GPSLongitude: ((abs(gps.longitude_degrees), 1), (longitude_minutes, 1), format_fraction(longitude_seconds)),
                    piexif.GPSIFD.GPSLongitudeRef: "E" if gps.longitude > 0 else "W",
                    piexif.GPSIFD.GPSProcessingMethod: "GPS".encode('ASCII'),
                    piexif.GPSIFD.GPSSatellites: str(gps.satellites),
                    piexif.GPSIFD.GPSSpeed: (0, 1) if gps.speed_knots is None else format_fraction(Fraction(gps.speed_knots)),
                    piexif.GPSIFD.GPSSpeedRef: "N",
                }
                if gps.fix_quality_3d > 0:
                    gps_ifd[piexif.GPSIFD.GPSMeasureMode] = str(gps.fix_quality_3d)
                if gps.timestamp_utc:
                    gps_ifd[piexif.GPSIFD.GPSDateStamp] = time.strftime("%Y:%m:%d", gps.timestamp_utc)
                    gps_ifd[piexif.GPSIFD.GPSTimeStamp] = ((gps.timestamp_utc.tm_hour, 1), (gps.timestamp_utc.tm_min, 1), (gps.timestamp_utc.tm_sec, 1))
                if gps.isactivedata:
                    gps_ifd[piexif.GPSIFD.GPSStatus] = gps.isactivedata
                exif_dict = {"GPS": gps_ifd}
                print(f"Exif GPS metadata: {gps_ifd}")
            else:
                print("No GPS fix")
            matches_name = "-".join(matches)
            filename = os.path.join(output_directory, f"{matches_name}-{frame}.jpg")
            capture_config = picam2.create_still_configuration()
            picam2.switch_mode_and_capture_file(capture_config, filename, delay=10, exif_data=exif_dict, format='jpeg')
            print(f"Image captured: {filename}")
            frame += 1
            time.sleep(args.gap)

if __name__ == '__main__':
    main()
