#!/bin/bash
# The classic Fizgig desktop app (Tk). The new look is ./fizgig.sh
cd "$(dirname "$0")"
if [ ! -x venv/bin/python ]; then
    echo "Fizgig's Python environment is missing - run the installer first."
    exit 1
fi
source venv/bin/activate
python lora_trainer_gui.py "$@" &
disown
