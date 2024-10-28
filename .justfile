default: run

export PYTHONPATH := "${PYTHONPATH}:/usr/local/lib/python3/dist-packages"

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

init gpsd_version="release-3.25":
    #!/usr/bin/env bash
    set -euxo pipefail
    sudo cp etc/udev/rules.d/99-adafruit-ultimate-gps-usb.rules /etc/udev/rules.d/99-adafruit-ultimate-gps-usb.rules
    distro=$(awk -F= '$1=="ID" { print $2 ;}' /etc/os-release)
    if [ "$distro" = "debian" ]; then
        sudo apt-get --yes install chrony firewalld libatlas-base-dev pps-tools python3-dev python3-picamera2 python3-venv unattended-upgrades \
            build-essential manpages-dev pkg-config git scons libncurses-dev python3-serial libdbus-1-dev python3-matplotlib
    fi
    mkdir --parents "{{ home_directory() }}/Projects"
    if [ ! -d "{{ home_directory() }}/Projects/gpsd" ]; then
        git -C "{{ home_directory() }}/Projects" clone https://gitlab.com/gpsd/gpsd.git
    fi
    git -C "{{ home_directory() }}/Projects/gpsd" fetch --tags
    git -C "{{ home_directory() }}/Projects/gpsd" switch --detach "{{ gpsd_version }}"
    (cd "{{ home_directory() }}/Projects/gpsd" && scons -c && scons && scons check && sudo scons install)
    sudo cp "{{ home_directory() }}/Projects/gpsd/buildtmp/systemd/gpsdctl@.service" "{{ home_directory() }}/Projects/gpsd/buildtmp/systemd/gpsd.service" "{{ home_directory() }}/Projects/gpsd/buildtmp/systemd/gpsd.socket" /etc/systemd/system
    sudo cp "{{ home_directory() }}/Projects/gpsd/contrib/apparmor/usr.sbin.gpsd" /etc/apparmor.d/usr.local.sbin.gpsd
    sudo sed --in-place 's|/usr/sbin/gpsd|/usr/local/sbin/gpsd|g' /etc/apparmor.d/usr.local.sbin.gpsd
    sudo cp etc/profile.d/pythonpath.sh /etc/profile.d
    sudo cp etc/default/gpsd /etc/default/gpsd
    sudo systemctl enable --now chrony.service gpsd.service

init-dev: && sync
    #!/usr/bin/env bash
    set -euxo pipefail
    distro=$(awk -F= '$1=="ID" { print $2 ;}' /etc/os-release)
    if [ "$distro" = "debian" ]; then
        sudo apt-get --yes install ffmpeg hailo-all libatlas-base-dev python3-dev python3-picamera2 python3-venv
    fi
    [ -d venv ] || python -m venv --system-site-packages venv
    venv/bin/python -m pip install --requirement requirements-dev.txt
    venv/bin/pre-commit install

alias install := install-system

install-system: init
    sudo mkdir --parents /etc/security/limits.d
    sudo cp etc/security/limits.d/99-detectionator.conf /etc/security/limits.d
    sudo chown --recursive root:root /etc/security/limits.d
    sudo mkdir --parents /etc/systemd/system
    sudo cp etc/systemd/system/* /etc/systemd/system/
    sudo chown --recursive root:root /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo cp detectionator.py /usr/local/bin/detectionator.py
    sudo chown root:root /usr/local/bin/detectionator.py
    sudo cp exif_utils.py /usr/local/bin/exif_utils.py
    sudo chown root:root /usr/local/bin/exif_utils.py
    sudo cp xmp_utils.py /usr/local/bin/xmp_utils.py
    sudo chown root:root /usr/local/bin/xmp_utils.py
    sudo mkdir --parents /usr/local/etc/detectionator/
    sudo cp config/hailo-config.toml /usr/local/etc/detectionator/config.toml
    sudo chown --recursive root:root /usr/local/etc/detectionator
    sudo mkdir --parents /usr/local/share/detectionator/
    sudo cp --recursive models /usr/local/share/detectionator/models
    sudo chown --recursive root:root /usr/local/share/detectionator
    sudo -H -u detectionator bash -c '[ -d /home/detectionator/venv ] || python -m venv --system-site-packages /home/detectionator/venv'
    sudo -H -u detectionator bash -c '/home/detectionator/venv/bin/python -m pip install --requirement requirements.txt'
    sudo systemctl enable detectionator.service
    sudo systemctl restart detectionator.service

install-user: init
    [ -d venv ] || python -m venv --system-site-packages venv
    venv/bin/python -m pip install --requirement requirements.txt
    mkdir --parents {{ config_directory() }}/detectionator
    cp config/hailo.toml {{ config_directory() }}/detectionator/config.toml
    mkdir --parents {{ config_directory() }}/systemd/user
    ln --force --relative --symbolic etc/systemd/user/* {{ config_directory() }}/systemd/user/
    systemctl --user daemon-reload
    systemctl --user enable detectionator.service
    systemctl --user restart detectionator.service

alias l := lint

lint:
    venv/bin/yamllint .
    venv/bin/ruff check --fix .

alias r := run

run model="models/mobilenet_v2.tflite" label="models/coco_labels.txt" output=(home_directory() / "Pictures") *args="":
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
