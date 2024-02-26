default: run

distro := `awk -F= '$1=="ID" { print $2 ;}' /etc/os-release`

alias f := format
alias fmt := format

format:
    venv/bin/ruff format .
    just --fmt --unstable

hdr device="/dev/v4l-subdev2" action="enable":
    systemctl --user stop detectionator.service || true
    pkill detectionator || true
    v4l2-ctl --set-ctrl wide_dynamic_range={{ if action == "enable" { "1" } else { "0" } }} --device {{ device }}
    systemctl --user start detectionator.service || true

init:
    #!/usr/bin/env bash
    set -euxo pipefail
    sudo cp etc/udev/rules.d/99-adafruit-ultimate-gps-usb.rules /etc/udev/rules.d/99-adafruit-ultimate-gps-usb.rules
    distro=$(awk -F= '$1=="ID" { print $2 ;}' /etc/os-release)
    if [ "$distro" = "debian" ]; then
        sudo apt-get --yes install firewalld libatlas-base-dev python3-dev python3-picamera2 python3-venv
    fi
    [ -d venv ] || python -m venv --system-site-packages venv
    venv/bin/python -m pip install -r requirements.txt

init-dev: && sync
    #!/usr/bin/env bash
    set -euxo pipefail
    distro=$(awk -F= '$1=="ID" { print $2 ;}' /etc/os-release)
    if [ "$distro" = "debian" ]; then
        sudo apt-get --yes install libatlas-base-dev python3-dev python3-picamera2 python3-venv
    fi
    [ -d venv ] || python -m venv --system-site-packages venv
    venv/bin/python -m pip install -r requirements-dev.txt
    venv/bin/pre-commit install

install: init
    mkdir --parents {{ config_directory() }}/systemd/user
    ln --force --relative --symbolic systemd/user/* {{ config_directory() }}/systemd/user/
    systemctl --user daemon-reload
    systemctl --user enable --now detectionator.service

alias l := lint

lint:
    venv/bin/yamllint .
    venv/bin/ruff check .

alias r := run

run model="models/mobilenet_v2.tflite" label="models/coco_labels.txt" output=(home_directory() / "Pictures") *args:
    venv/bin/python detectionator.py \
        --label {{ label }} \
        --model {{ model }} \
        --output {{ output }} \
        {{ args }}

sync:
    venv/bin/pip-sync requirements-dev.txt requirements.txt

alias t := test

test:
    venv/bin/pytest

alias u := update
alias up := update

update:
    venv/bin/pip-compile \
        --allow-unsafe \
        --generate-hashes \
        --reuse-hashes \
        --upgrade \
        requirements-dev.in
    venv/bin/pip-compile \
        --allow-unsafe \
        --generate-hashes \
        --reuse-hashes \
        --upgrade \
        requirements.in
    venv/bin/pre-commit autoupdate
