import time
import csv
import busio
import subprocess
import digitalio
import board
import adafruit_rfm9x
import pygame
import RPi.GPIO as GPIO

# --- Configuration ---
SHUTDOWN_PIN = 17
SHUTDOWN_DEBOUNCE_S = 3.0  # Time in seconds the signal must remain triggered
_shutdown_low_since = None
counter = 0

# --- GPIO Setup ---
GPIO.setmode(GPIO.BCM)

# Shutdown Input
# (Using PUD_UP ensures the pin stays HIGH unless your circuit pulls it to Ground)
GPIO.setup(SHUTDOWN_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# RGB LED Outputs
RED_PIN   = 16   # physical pin 36
GREEN_PIN = 20   # physical pin 38
BLUE_PIN  = 21   # physical pin 40

GPIO.setup(RED_PIN, GPIO.OUT)
GPIO.setup(GREEN_PIN, GPIO.OUT)
GPIO.setup(BLUE_PIN, GPIO.OUT)

# --- Audio Setup ---
pygame.mixer.init()

# --- LoRa Radio Setup ---
RADIO_FREQ_MHZ = 915.0
CS = digitalio.DigitalInOut(board.CE1)
RESET = digitalio.DigitalInOut(board.D25)
spi = busio.SPI(board.SCK, MOSI=board.MOSI, MISO=board.MISO)

try:
    rfm9x = adafruit_rfm9x.RFM9x(spi, CS, RESET, RADIO_FREQ_MHZ)
    print("LoRa Radio Initialized Successfully")
except Exception as e:
    print(f"Failed to initialize LoRa: {e}")

# --- LED Helper Functions ---
def led_off():
    GPIO.output(RED_PIN, GPIO.LOW)
    GPIO.output(GREEN_PIN, GPIO.LOW)
    GPIO.output(BLUE_PIN, GPIO.LOW)

def solid_yellow():
    GPIO.output(RED_PIN, GPIO.HIGH)
    GPIO.output(GREEN_PIN, GPIO.HIGH)
    GPIO.output(BLUE_PIN, GPIO.LOW)

def flash_yellow(times=3, on=1.0, off=0.5):
    """Flashes yellow using checked_sleep to keep shutdown monitoring alive."""
    for _ in range(times):
        solid_yellow()
        checked_sleep(on)
        led_off()
        checked_sleep(off)

# --- Safety & Monitoring Functions ---
def check_for_low_power_shutdown():
    """
    Monitors Pin 17. If it stays triggered for the debounce duration, 
    it triggers a safe system halt.
    """
    global _shutdown_low_since

    # Read the physical pin state
    pin_state = GPIO.input(SHUTDOWN_PIN)

    # Note: If your hardware pulls the pin LOW on low power, change this to GPIO.LOW
    if pin_state == GPIO.HIGH:
        if _shutdown_low_since is None:
            _shutdown_low_since = time.time()
            print("Warning: Low power signal detected. Debouncing...")
        elif (time.time() - _shutdown_low_since) >= SHUTDOWN_DEBOUNCE_S:
            print("CRITICAL: LOW POWER SHUTDOWN TRIGGERED")
            # Sync filesystem and shutdown safely
            subprocess.run(["sudo", "shutdown", "-h", "now"])
    else:
        # Reset timer if power stabilizes
        _shutdown_low_since = None

def checked_sleep(duration):
    """Sleeps while continuously monitoring the shutdown pin."""
    end_time = time.time() + duration
    while time.time() < end_time:
        check_for_low_power_shutdown()
        time.sleep(0.1)

# --- Audio & Message Processing Functions ---
def process_incoming_messages(message_str):
    """
    Parses incoming LoRa messages dynamically.
    Handles 'deviceID:data', 'deviceID,data', or single string commands.
    """
    try:
        msg = message_str.strip()
        print(f"Raw message: '{msg}'")
        
        if ":" in msg:
            device_id, data = msg.split(":", 1)
            return device_id.strip(), data.strip()
        elif "," in msg:
            device_id, data = msg.split(",", 1)
            return device_id.strip(), data.strip()
        else:
            # If it's a standalone message like just "0.34"
            return msg, msg
            
    except Exception as e:
        print(f"Error processing message: {e}")
        return "Error", "Invalid Format"

def play_audio():
    """Plays the audio file for 3 minutes without blocking the shutdown logic."""
    print("Playing audio (35Hz.mp3) for 3 minutes...")
    pygame.mixer.music.load('/home/jumboshoo/Documents/35Hz.mp3')
    pygame.mixer.music.play()
    
    # Use checked_sleep so power monitoring stays active during playback
    checked_sleep(180)  
    
    pygame.mixer.music.stop()
    print("Audio playback completed.")

# --- Main Loop ---
print("System Monitoring and Radio Receiver Active...")

try:
    while True:
        # 1. Immediate check for power status
        check_for_low_power_shutdown()

        # 2. Check for LoRa packets (Using a short 0.5s timeout to keep loop responsive)
        received_message = rfm9x.receive(timeout=0.5) 

        if received_message is not None:
            try:
                received_str = str(received_message, 'utf-8').strip()
                device_id, data = process_incoming_messages(received_str)
                print(f"Processed -> Device ID: {device_id}, Data: {data}")

                # Log to CSV
                with open('output.csv', mode='a', newline='') as file:
                    csv.writer(file).writerow([time.ctime(), device_id, data, counter])

                # --- SCRIPT 1 LOGIC: Respond back based on data ---
                if data == "0":
                    if device_id == "1":
                        checked_sleep(10)
                        rfm9x.send(bytes("0.2", "utf-8"))
                    elif device_id == "2":
                        checked_sleep(10)
                        rfm9x.send(bytes("0.34", "utf-8"))

                # --- SCRIPT 2 LOGIC: Action on "0.34" ---
                if device_id == "0.34" or data == "0.34":
                    print("Message 0.34 RECEIVED → Thumper ON for 3 min")
                    flash_yellow()   # Yellow flash feedback
                    play_audio()     # Play audio for 3 mins (safely monitored)
                
                counter += 1

            except Exception as e:
                print(f"Processing Error: {e}")

        # 3. Small idle sleep to prevent high CPU usage
        checked_sleep(0.1)

except KeyboardInterrupt:
    print("Script manually stopped.")
finally:
    # Cleanup resources safely on exit
    led_off()
    GPIO.cleanup()
    pygame.mixer.quit()
    print("GPIO and Audio cleaned up.")
