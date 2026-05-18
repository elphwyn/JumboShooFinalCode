#!/usr/bin/env python3
"""
lora_cam_bridge.py — 915 MHz with RGB LED status + low-power shutdown monitor

RECEIVE 0.2  → flash WHITE 3× → run YOLO → if hit, flash BLUE 3× + TX "2.0"
PIN 17 LOW for >1s → graceful filesystem sync + system halt
"""

# ───────────────────────── imports ─────────────────────────
import os, sys, argparse, uuid, time, subprocess, csv

# RPi / LoRa
import busio
from digitalio import DigitalInOut
import board
import adafruit_rfm9x

# CV / AI
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

# GPIO for LED + shutdown pin
import RPi.GPIO as GPIO

# ─────────────── GPIO SETUP ───────────────

GPIO.setmode(GPIO.BCM)

# RGB LED pins
RED_PIN   = 16   # physical pin 36
GREEN_PIN = 20   # physical pin 38
BLUE_PIN  = 21   # physical pin 40

GPIO.setup(RED_PIN,   GPIO.OUT)
GPIO.setup(GREEN_PIN, GPIO.OUT)
GPIO.setup(BLUE_PIN,  GPIO.OUT)

# Shutdown monitor pin
SHUTDOWN_PIN          = 17
SHUTDOWN_DEBOUNCE_S   = 1.0   # seconds pin must stay LOW before shutdown
_shutdown_low_since   = None  # module-level timer

# Pull-up so pin stays HIGH unless battery circuit pulls it LOW
GPIO.setup(SHUTDOWN_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# ─────────────── LED helpers ───────────────

def led_off():
    GPIO.output(RED_PIN,   GPIO.LOW)
    GPIO.output(GREEN_PIN, GPIO.LOW)
    GPIO.output(BLUE_PIN,  GPIO.LOW)

def solid_blue():
    GPIO.output(RED_PIN,   GPIO.LOW)
    GPIO.output(GREEN_PIN, GPIO.LOW)
    GPIO.output(BLUE_PIN,  GPIO.HIGH)

def solid_white():
    GPIO.output(RED_PIN,   GPIO.HIGH)
    GPIO.output(GREEN_PIN, GPIO.HIGH)
    GPIO.output(BLUE_PIN,  GPIO.HIGH)

def flash_blue(times=3, on=1.0, off=0.5):
    for _ in range(times):
        solid_blue();  time.sleep(on)
        led_off();     time.sleep(off)

def flash_white(times=3, on=1.0, off=0.5):
    for _ in range(times):
        solid_white(); time.sleep(on)
        led_off();     time.sleep(off)

# ─────────────── low-power shutdown monitor ───────────────

def check_for_low_power_shutdown():
    """
    Reads Pin 17. If it stays LOW for SHUTDOWN_DEBOUNCE_S seconds,
    syncs the filesystem and halts the Pi.
    Pin is LOW when battery voltage drops below ~10.5 V.
    """
    global _shutdown_low_since
    pin_state = GPIO.input(SHUTDOWN_PIN)

    if pin_state == GPIO.LOW:                          # battery voltage too low
        if _shutdown_low_since is None:
            _shutdown_low_since = time.time()
            print("Warning: Low power signal detected. Debouncing...")
        elif (time.time() - _shutdown_low_since) >= SHUTDOWN_DEBOUNCE_S:
            print("CRITICAL: LOW POWER SHUTDOWN TRIGGERED")
            subprocess.run(["sudo", "shutdown", "-h", "now"])
    else:
        _shutdown_low_since = None                     # power stable — reset


def checked_sleep(duration):
    """Sleeps in 100 ms increments, checking the shutdown pin each tick."""
    end_time = time.time() + duration
    while time.time() < end_time:
        check_for_low_power_shutdown()
        time.sleep(0.1)

# ─────────────── defaults (CLI) ───────────────

DEF_SAVE_DIR   = "/home/a/JumboShoo/logging/ElephantHits"
DEF_MODEL_PATH = "/home/a/JumboShoo/scripts/models/Elephants2x.pt"
DEF_CONF       = 0.50
DEF_WIDTH      = 1280
DEF_HEIGHT     = 720
DEF_CYCLES     = 2
DEF_TRIGGER    = "0.2"    # RECEIVE trigger → flash WHITE
DEF_HIT_REPLY  = "2.0"   # SEND hit       → flash BLUE
DEF_LORA_FREQ  = 915

# ───────────────────────────────────────────────

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def log_hit(csv_path, run_id, count, best_conf, filename):
    with open(csv_path, "a") as f:
        f.write(f"{run_id},{count},{best_conf:.3f},{filename}\n")

# ───────────── YOLO detector ─────────────

def run_detector(model, save_dir, cycles, conf_thres, cam):
    ensure_dir(save_dir)
    csv_log  = os.path.join(save_dir, "detections_log.txt")
    run_id   = str(uuid.uuid4())[:8]

    hit, total_cnt, best_conf = False, 0, 0.0
    elephant_id = [k for k, v in model.names.items() if v == "elephant"][0]

    for idx in range(cycles):
        check_for_low_power_shutdown()          # monitor power mid-detection

        frame = cam.capture_array()
        print("Captured image")
        res = model(frame, classes=[elephant_id],
                    conf=conf_thres, verbose=False)[0]

        if res.boxes and len(res.boxes) > 0:
            hit       = True
            cnt       = len(res.boxes)
            total_cnt += cnt
            best_conf  = max(best_conf, float(res.boxes.conf.max()))

            annotated = res.plot()
            fname     = f"ele_{run_id}_{idx:03}.jpg"
            cv2.imwrite(os.path.join(save_dir, fname), annotated)
            log_hit(csv_log, run_id, cnt, best_conf, fname)
            print(f"[{run_id}] DETECTED {cnt} elephant(s) → saved {fname}")
        else:
            print(f"[{run_id}] no elephants")

    return hit, total_cnt, best_conf

# ───────────────── LoRa setup ─────────────────

def init_lora(freq_mhz):
    cs    = DigitalInOut(board.CE1)
    reset = DigitalInOut(board.D25)
    spi   = busio.SPI(board.SCK, board.MOSI, board.MISO)
    radio = adafruit_rfm9x.RFM9x(spi, cs, reset, freq_mhz)
    radio.tx_power = 23
    return radio

# ───────────── argument parsing ─────────────

def parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir",   default=DEF_SAVE_DIR)
    p.add_argument("--model_path", default=DEF_MODEL_PATH)
    p.add_argument("--conf",       type=float, default=DEF_CONF)
    p.add_argument("--width",      type=int,   default=DEF_WIDTH)
    p.add_argument("--height",     type=int,   default=DEF_HEIGHT)
    p.add_argument("--cycles",     type=int,   default=DEF_CYCLES)
    p.add_argument("--trigger",    default=DEF_TRIGGER)
    p.add_argument("--hit_reply",  default=DEF_HIT_REPLY)
    p.add_argument("--freq",       type=float, default=DEF_LORA_FREQ)
    return p.parse_args()

# ──────────────── MAIN LOOP ────────────────

def main():
    args  = parse_cli()
    radio = init_lora(args.freq)
    model = YOLO(args.model_path)

    print(f"Ready! Listening for '{args.trigger}'…")

    cam = Picamera2()
    cam.configure(cam.create_still_configuration(
        main={"size": (args.width, args.height), "format": "RGB888"}))
    cam.start()
    checked_sleep(0.2)                          # startup delay + power check

    while True:
        # ── 1. Power safety check ──────────────────────────────────────────
        check_for_low_power_shutdown()

        # ── 2. Listen for LoRa packet ──────────────────────────────────────
        pkt = radio.receive()

        if pkt is None:
            checked_sleep(0.1)                  # idle — keep monitoring power
            continue

        try:
            msg = pkt.decode("utf-8").strip()
        except Exception:
            print("Non-UTF8 packet ignored")
            continue

        print(f"LoRa RX: '{msg}'")

        # ── 3. Act on trigger ──────────────────────────────────────────────
        if msg == args.trigger:
            print("Trigger received → flashing WHITE")
            flash_white()                       # WHITE = scanning

            hit, count, best = run_detector(
                model, args.save_dir, args.cycles, args.conf, cam)

            if hit:
                reply = args.hit_reply
                print(f"Sending hit reply '{reply}' …")
                flash_blue()                    # BLUE = elephant confirmed
                radio.send(reply.encode())
                print(f"LoRa TX: '{reply}'  (count={count}, best={best:.2f})")
            else:
                print("No elephants detected → no reply")

# ───────────────── entry point ─────────────────

if __name__ == "__main__":
    try:
        main()
    finally:
        led_off()
        GPIO.cleanup()
        print("GPIO cleaned up.")
