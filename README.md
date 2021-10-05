# Project 2: Ice Cream

<http://www.cs.columbia.edu/~kar/4444f21/node19.html>

## Installation

Requires **python3.6**

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

You can also specify the optional parameters below to disable GUI, disable browser launching, change port and address of server.

```bash
usage: main.py [-h] [--port PORT] [--address ADDRESS] [--no_browser]
               [--no_gui]

optional arguments:
  -h, --help            show this help message and exit
  --port PORT, -p PORT  Port to start
  --address ADDRESS, -a ADDRESS
                        Address
  --no_browser, -nb     Disable browser launching in GUI mode
  --no_gui, -ng         Disable GUI
```

## Debugging

The code generates a `debug.log` on every execution, detailing all the turns and steps in the game.

## Reference

1. <https://remi.readthedocs.io/en/latest/remi.html>
1. <https://github.com/dddomodossola/remi>
