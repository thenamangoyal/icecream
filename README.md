# Project 2: Ice Cream

<http://www.cs.columbia.edu/~kar/4444f21/node19.html>

[Documentation](https://docs.google.com/document/d/1wCQZNEupmkwjPrOVFU3S3sMxz16ph3gBDZg0aunx46Q/edit?usp=sharing)

## Installation

Requires **python3.6** or higher

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

You can also specify the optional parameters below to disable GUI, disable browser launching, change port and address of server.

```bash
usage: main.py [-h] [--automatic] [--seed SEED]
               [--flavors {2,3,4,5,6,8,9,10,12,0}] [--port PORT]
               [--address ADDRESS] [--no_browser] [--no_gui]
               [--log_path LOG_PATH] [--disable_timeout] [--disable_logging]
               [--players PLAYERS [PLAYERS ...]]

optional arguments:
  -h, --help            show this help message and exit
  --automatic           Start playing automatically in GUI mode
  --seed SEED, -s SEED  Seed used by random number generator, specify 0 to use
                        no seed and have different random behavior on each
                        launch
  --flavors {2,3,4,5,6,8,9,10,12,0}, -f {2,3,4,5,6,8,9,10,12,0}
                        Number of flavors, specify 0 to use random number of
                        flavors
  --port PORT           Port to start
  --address ADDRESS, -a ADDRESS
                        Address
  --no_browser, -nb     Disable browser launching in GUI mode
  --no_gui, -ng         Disable GUI
  --log_path LOG_PATH   Directory path to dump log files, filepath if
                        disable_logging is false
  --disable_timeout, -time
                        Disable Timeout in non GUI mode
  --disable_logging     Disable Logging, log_path becomes path to file
  --players PLAYERS [PLAYERS ...], -p PLAYERS [PLAYERS ...]
                        List of players space separated
```

## Debugging

The code generates a `debug.log` on every execution, detailing all the turns and steps in the game.

## Reference

1. <https://remi.readthedocs.io/en/latest/remi.html>
1. <https://github.com/dddomodossola/remi>
