default: run

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
        sudo apt-get --yes install chrony firewalld gpsd gpsd-clients libatlas-base-dev pps-tools python3-dev python3-gps python3-picamera2 python3-venv
    fi
    # todo will socket start gpsd automatically?
    sudo systemctl enable --now chrony gpsd

init-dev: && sync
    #!/usr/bin/env bash
    set -euxo pipefail
    distro=$(awk -F= '$1=="ID" { print $2 ;}' /etc/os-release)
    if [ "$distro" = "debian" ]; then
        sudo apt-get --yes install libatlas-base-dev python3-dev python3-picamera2 python3-venv
    fi
    [ -d venv ] || python -m venv --system-site-packages venv
    venv/bin/python -m pip install --requirement requirements-dev.txt
    venv/bin/pre-commit install

alias install := install-system

install-system: init
    sudo mkdir --parents /etc/security/limits.d
    sudo cp security/limits.d/99-detectionator.conf /etc/security/limits.d
    sudo chown --recursive root:root /etc/security/limits.d
    sudo mkdir --parents /etc/systemd/system
    sudo cp systemd/system/* /etc/systemd/system/
    sudo chown --recursive root:root /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo cp detectionator.py /usr/local/bin/detectionator.py
    sudo mkdir --parents /usr/local/etc/detectionator/
    sudo chown root:root /usr/local/bin/detectionator.py
    sudo cp config/fast-config.toml /usr/local/etc/detectionator/config.toml
    sudo cp models/* /usr/local/etc/detectionator/models
    sudo chown --recursive root:root /usr/local/etc/detectionator
    sudo -H -u detectionator bash -c '[ -d /home/detectionator/venv ] || python -m venv --system-site-packages /home/detectionator/venv'
    sudo -H -u detectionator bash -c '/home/detectionator/venv/bin/python -m pip install --requirement requirements.txt'
    sudo systemctl enable detectionator.service

install-user: init
    [ -d venv ] || python -m venv --system-site-packages venv
    venv/bin/python -m pip install --requirement requirements.txt
    ln --force --relative --symbolic systemd/user/* {{ config_directory() }}/systemd/user/
    mkdir --parents {{ config_directory() }}/detectionator
    cp detectionator.toml.template {{ config_directory() }}/detectionator/config.toml
    mkdir --parents {{ config_directory() }}/systemd/user
    ln --force --relative --symbolic systemd/user/* {{ config_directory() }}/systemd/user/
    systemctl --user daemon-reload
    systemctl --user enable --now detectionator.service

alias l := lint

lint:
    venv/bin/yamllint .
    venv/bin/ruff check --fix .

alias r := run

run model="models/mobilenet_v2.tflite" label="models/coco_labels.txt" output=(home_directory() / "Pictures") *args="":
    sudo -H -u detectionator bash -c 'cd; bash'
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
    sudo -H -u detectionator bash -c 'cd; bash'
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
