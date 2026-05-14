import time
import csv
import busio
import subprocess
from digitalio import DigitalInOut
import board
import adafruit_rfm9x
import RPi.GPIO as GPIO

# --- Configuration ---
SHUTDOWN_PIN = 17
SHUTDOWN_DEBOUNCE_S = 1.0  # Time in seconds the signal must remain LOW
_shutdown_low_since = None

# --- GPIO Setup ---
GPIO.setmode(GPIO.BCM)
# Using PUD_UP ensures the pin stays HIGH unless your circuit pulls it to Ground
GPIO.setup(SHUTDOWN_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# --- LoRa Radio Setup ---
CS = DigitalInOut(board.CE1)
RESET = DigitalInOut(board.D25)
spi = busio.SPI(board.SCK, MOSI=board.MOSI, MISO=board.MISO)

try:
    rfm9x = adafruit_rfm9x.RFM9x(spi, CS, RESET, 915.0)
    print("LoRa Radio Initialized")
except Exception as e:
    print(f"Failed to initialize LoRa: {e}")

# --- Helper Functions ---

def process_incoming_messages(message_str):
    """
    Parses incoming LoRa messages. 
    Assumes format: "deviceID:data" (e.g., "1:0")
    """
    try:
        if ":" in message_str:
            device_id, data = message_str.split(":", 1)
            return device_id, data
        return "Unknown", message_str
    except Exception:
        return "Error", "Invalid Format"

def check_for_low_power_shutdown():
    """
    Monitors Pin 17. If it stays LOW for the debounce duration, 
    it triggers a system halt.
    """
    global _shutdown_low_since
    
    # Read the physical pin state
    pin_state = GPIO.input(SHUTDOWN_PIN)
    
    # Logic: Pin 17 is LOW when battery is < 10.5V
    if pin_state == GPIO.HIGH:
        if _shutdown_low_since is None:
            _shutdown_low_since = time.time()
            print("Warning: Low power signal detected. Debouncing...")
        elif (time.time() - _shutdown_low_since) >= SHUTDOWN_DEBOUNCE_S:
            print("CRITICAL: LOW POWER SHUTDOWN TRIGGERED")
            # Sync filesystem and shutdown
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

# --- Main Loop ---
print("System Monitoring Active...")
counter = 0

try:
    while True:
        # 1. Immediate check for power status
        check_for_low_power_shutdown()

        # 2. Check for LoRa packets
        received_message = rfm9x.receive(timeout=0.5) 
        
        if received_message is not None:
            try:
                received_message_str = str(received_message, 'utf-8').strip()
                print(f"Received: {received_message_str}")
                
                device_id, data = process_incoming_messages(received_message_str)
                
                # Log to CSV
                with open('output.csv', mode='a', newline='') as file:
                    csv.writer(file).writerow([time.ctime(), device_id, data, counter])

                # Logic based on received data
                if data == "0":
                    if device_id == "1":
                        checked_sleep(10)
                        rfm9x.send(bytes("0.2", "utf-8"))
                    elif device_id == "2":
                        checked_sleep(10)
                        rfm9x.send(bytes("0.34", "utf-8"))
                
                counter += 1
                
            except Exception as e:
                print(f"Processing Error: {e}")
        
        # 3. Small idle sleep to prevent high CPU usage
        checked_sleep(0.1)

except KeyboardInterrupt:
    print("Script manually stopped.")
finally:
    GPIO.cleanup()