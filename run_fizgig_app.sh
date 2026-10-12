#!/bin/bash
# Fizgig in its own app window - the web UI, the new look (the classic app is run_fizgig.sh)
cd "$(dirname "$0")"
source venv/bin/activate
python lora_trainer_gui.py --app "$@" &
disown
