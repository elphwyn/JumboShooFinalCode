#!/usr/bin/env python3
"""
Integrated LoRa AI Camera with Power Protection
- AI: Detects elephants and replies via LoRa.
- Protection: Monitors Pin 17 for low-power shutdown.
- Status: RGB LED feedback.
"""

import os
import sys
import argparse
import uuid
import time
import csv
import subprocess
import busio
import board
import cv2
import RPi.GPIO as GPIO
from digitalio import DigitalInOut
from picamera2 import Picamera2
from ultralytics import YOLO
import adafruit_rfm9x

# ───────────────────────── CONFIGURATION ─────────────────────────

# GPIO Pins
SHUTDOWN_PIN = 17
RED_PIN      = 16
GREEN_PIN    = 20
BLUE_PIN     = 21

# Power Settings
SHUTDOWN_DEBOUNCE_S = 3.0  # Time signal must be LOW to trigger shutdown
_shutdown_low_since = None

# AI Defaults
DEF_SAVE_DIR    = "/home/a/JumboShoo/logging/ElephantHits"
DEF_MODEL_PATH  = "/home/a/JumboShoo/scripts/models/Elephants2x.pt"
DEF_CONF        = 0.50
DEF_WIDTH       = 1280
DEF_HEIGHT      = 720
DEF_CYCLES      = 2
DEF_TRIGGER     = "0.2"
DEF_HIT_REPLY   = "2.0"
DEF_LORA_FREQ   = 915.0

# ───────────────────────── GPIO & POWER ─────────────────────────

def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    # LEDs
    GPIO.setup(RED_PIN, GPIO.OUT)
    GPIO.setup(GREEN_PIN, GPIO.OUT)
    GPIO.setup(BLUE_PIN, GPIO.OUT)
    # Power Monitor (Internal Pull-Up)
    GPIO.setup(SHUTDOWN_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def check_power():
    """Monitors Pin 17. If LOW for debounce duration, shuts down Pi."""
    global _shutdown_low_since
   
    # Logic: Pin 17 is LOW when battery is < 10.5V
    if GPIO.input(SHUTDOWN_PIN) == GPIO.HIGH:
        if _shutdown_low_since is None:
            _shutdown_low_since = time.time()
            print("Warning: Low power signal detected. Debouncing...")
        elif (time.time() - _shutdown_low_since) >= SHUTDOWN_DEBOUNCE_S:
            print("CRITICAL: LOW POWER SHUTDOWN TRIGGERED")
            led_off()
            # Flash Red quickly before dying if possible
            GPIO.output(RED_PIN, GPIO.HIGH)
            subprocess.run(["sudo", "shutdown", "-h", "now"])
            sys.exit()
    else:
        _shutdown_low_since = None

def checked_sleep(duration):
    """Sleeps while continuously monitoring the power/shutdown pin."""
    end_time = time.time() + duration
    while time.time() < end_time:
        check_power()
        time.sleep(0.1)

# ───────────────────────── LED HELPERS ──────────────────────────

def led_off():
    GPIO.output(RED_PIN, GPIO.LOW)
    GPIO.output(GREEN_PIN, GPIO.LOW)
    GPIO.output(BLUE_PIN, GPIO.LOW)

def solid_blue():
    GPIO.output(RED_PIN, GPIO.LOW)
    GPIO.output(GREEN_PIN, GPIO.LOW)
    GPIO.output(BLUE_PIN, GPIO.HIGH)

def solid_white():
    GPIO.output(RED_PIN, GPIO.HIGH)
    GPIO.output(GREEN_PIN, GPIO.HIGH)
    GPIO.output(BLUE_PIN, GPIO.HIGH)

def flash_blue(times=3, on=0.5, off=0.3):
    for _ in range(times):
        solid_blue()
        checked_sleep(on)
        led_off()
        checked_sleep(off)

def flash_white(times=3, on=0.5, off=0.3):
    for _ in range(times):
        solid_white()
        checked_sleep(on)
        led_off()
        checked_sleep(off)

# ───────────────────────── AI & LORA ───────────────────────────

def init_lora(freq_mhz):
    cs    = DigitalInOut(board.CE1)
    reset = DigitalInOut(board.D25)
    spi   = busio.SPI(board.SCK, board.MOSI, board.MISO)
    try:
        radio = adafruit_rfm9x.RFM9x(spi, cs, reset, freq_mhz)
        radio.tx_power = 23
        print(f"LoRa Initialized at {freq_mhz}MHz")
        return radio
    except Exception as e:
        print(f"LoRa Init Error: {e}")
        return None

def run_detector(model, save_dir, cycles, conf_thres, cam):
    os.makedirs(save_dir, exist_ok=True)
    csv_log = os.path.join(save_dir, "detections_log.txt")
    run_id  = str(uuid.uuid4())[:8]

    hit, total_cnt, best_conf = False, 0, 0.0
    # Try to find 'elephant' class ID
    elephant_id = [k for k, v in model.names.items() if v == "elephant"]
    if not elephant_id:
        print("Error: 'elephant' class not found in model.")
        return False, 0, 0.0
   
    target = elephant_id[0]

    for idx in range(cycles):
        check_power() # Check power between frame captures
        frame = cam.capture_array()
        res = model(frame, classes=[target], conf=conf_thres, verbose=False)[0]

        if res.boxes and len(res.boxes) > 0:
            hit = True
            cnt = len(res.boxes)
            total_cnt += cnt
            best_conf = max(best_conf, float(res.boxes.conf.max()))

            annotated = res.plot()
            fname = f"ele_{run_id}_{idx:03}.jpg"
            cv2.imwrite(os.path.join(save_dir, fname), annotated)
           
            with open(csv_log, "a") as f:
                f.write(f"{time.ctime()},{run_id},{cnt},{best_conf:.3f},{fname}\n")
            print(f"[{run_id}] DETECTED {cnt} elephant(s)")
       
    return hit, total_cnt, best_conf

def process_complex_msg(message_str):
    """Handles 'deviceID:data' format from Script 2."""
    try:
        if ":" in message_str:
            device_id, data = message_str.split(":", 1)
            return device_id, data
        return "Unknown", message_str
    except:
        return "Error", "Invalid"

# ───────────────────────── MAIN LOOP ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir",   default=DEF_SAVE_DIR)
    parser.add_argument("--model_path", default=DEF_MODEL_PATH)
    parser.add_argument("--conf",       type=float, default=DEF_CONF)
    parser.add_argument("--width",      type=int,   default=DEF_WIDTH)
    parser.add_argument("--height",     type=int,   default=DEF_HEIGHT)
    parser.add_argument("--cycles",     type=int,   default=DEF_CYCLES)
    parser.add_argument("--trigger",    default=DEF_TRIGGER)
    parser.add_argument("--hit_reply",  default=DEF_HIT_REPLY)
    parser.add_argument("--freq",       type=float, default=DEF_LORA_FREQ)
    args = parser.parse_args()

    setup_gpio()
    radio = init_lora(args.freq)
    if not radio: return

    print("Loading AI Model...")
    model = YOLO(args.model_path)

    cam = Picamera2()
    cam.configure(cam.create_still_configuration(
        main={"size": (args.width, args.height), "format": "RGB888"}))
    cam.start()
   
    print("System Active. Monitoring Power & LoRa...")

    while True:
        check_power()
       
        # Non-blocking receive (timeout allows loop to return for power check)
        pkt = radio.receive(timeout=1.0)

        if pkt is not None:
            try:
                msg = pkt.decode("utf-8").strip()
                print(f"LoRa RX: '{msg}'")
            except:
                continue

            # 1. Check for AI Trigger
            if msg == args.trigger:
                flash_white()
                hit, count, best = run_detector(model, args.save_dir, args.cycles, args.conf, cam)
                if hit:
                    flash_blue()
                    radio.send(args.hit_reply.encode())
                    print(f"Hit Sent: {args.hit_reply}")

            # 2. Check for DeviceID:Data format (Script 2 logic)
            elif ":" in msg:
                dev_id, data = process_complex_msg(msg)
                # Log to CSV as per Script 2
                with open('output.csv', mode='a', newline='') as file:
                    csv.writer(file).writerow([time.ctime(), dev_id, data])
               
                if data == "0":
                    print(f"Control message from {dev_id} received.")
                    checked_sleep(2) # Short wait before responding
                    radio.send(b"ACK")

        # Minimal CPU rest
        time.sleep(0.1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopping script...")
    finally:
        led_off()
        GPIO.cleanup()