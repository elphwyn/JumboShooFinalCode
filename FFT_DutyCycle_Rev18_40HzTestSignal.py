#!/usr/bin/env python3
# Geophone FFT + LoRa + RGB LED Status Script
# + FFT template reference matching (synthetic or learned)

import time
import math
import numpy as np
import matplotlib.pyplot as plt
import Adafruit_ADS1x15
import csv
import board
import busio
import digitalio
import adafruit_rfm9x
import RPi.GPIO as GPIO
from collections import deque
import os
import subprocess

# -------------------------
# IDs for this node & Brain
# -------------------------
DEVICE_ID = "1.0"   # This geophone node
BRAIN_ID  = "0.1"   # The Brain node

# ========================================================
# USER CONFIG: TEMPLATE MATCHING
# ========================================================

# Which detection methods should trigger LoRa?
USE_SPIKE_DETECTOR   = False   #                                                                                                                                                                                                                                  rent mag/power spike logic
USE_TEMPLATE_MATCH   = True   # new reference matching logic
PLOT_MATCH_DEBUG = True # Set to false to disable plotting
PLOT_ONLY_ON_TRIGGER = True # Only plot when a trigger/match happens

# Template source:
#   "synthetic" = build reference FFT template from a simulated elephant-walk signal
#   "learn"     = learn template from the next N windows of REAL captured data
TEMPLATE_SOURCE = "synthetic"   # "synthetic" or "learn"

# Matching frequency band (Hz) – based on elephant footfall energy
FMIN = 5.0
FMAX = 45.0

# Sliding FFT settings for matching (independent of your 10 s capture)
# These operate on the collected data buffer each cycle.
MATCH_WINDOW_S = 1.28          # 256 samples @ 200 Hz-ish
MATCH_HOP_S    = 0.32          # 64 samples @ 200 Hz-ish

# Similarity threshold (cosine similarity of normalized band power)
SIM_THRESH = 0.70

# Persistence rule: require >= PERSIST_HITS matches within a rolling time window
PERSIST_WINDOW_S = 12.0
PERSIST_HITS     = 3

# Energy gating: require band-energy above baseline
# Baseline is estimated from last BASELINE_SECONDS of windows
BASELINE_SECONDS = 120.0
ENERGY_K_MAD     = 6.0

# ========================================================
# RGB LED SETUP (BCM pins)
# ========================================================
GPIO.setmode(GPIO.BCM)

RED_PIN = 16
GREEN_PIN = 20
BLUE_PIN = 21

GPIO.setup(RED_PIN, GPIO.OUT)
GPIO.setup(GREEN_PIN, GPIO.OUT)
GPIO.setup(BLUE_PIN, GPIO.OUT)

def led_off():
    GPIO.output(RED_PIN, GPIO.LOW)
    GPIO.output(GREEN_PIN, GPIO.LOW)
    GPIO.output(BLUE_PIN, GPIO.LOW)

def solid_green():
    GPIO.output(RED_PIN, GPIO.LOW)
    GPIO.output(GREEN_PIN, GPIO.HIGH)
    GPIO.output(BLUE_PIN, GPIO.LOW)

def solid_yellow():
    GPIO.output(RED_PIN, GPIO.HIGH)
    GPIO.output(GREEN_PIN, GPIO.HIGH)
    GPIO.output(BLUE_PIN, GPIO.LOW)

def solid_red():
    GPIO.output(RED_PIN, GPIO.HIGH)
    GPIO.output(GREEN_PIN, GPIO.LOW)
    GPIO.output(BLUE_PIN, GPIO.LOW)

def flash_green(times=3, on_time=1.0, off_time=0.5):
    for _ in range(times):
        solid_green()
        time.sleep(on_time)
        led_off()
        time.sleep(off_time)

def flash_yellow(times=3, on_time=1.0, off_time=0.5):
    for _ in range(times):
        solid_yellow()
        time.sleep(on_time)
        led_off()
        time.sleep(off_time)

def flash_red(times=3, on_time=1.0, off_time=0.5):
    for _ in range(times):
        solid_red()
        time.sleep(on_time)
        led_off()
        time.sleep(off_time)
        
        
# This is just a little cleanup to ensure that the LED always get turned off and
# doesn't stay on if you accidentally end the code too early
led_off()

# ==============================
# Low Power Shutdown Configuration
# ==============================
SHUTDOWN_PIN = 17 # GPIO17 on the Pi, physical pin 11

# Setup
GPIO.setup(SHUTDOWN_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

_shutdown_armed = False
_shutdown_low_since = None
SHUTDOWN_DEBOUNCE_S = 0.5

def check_for_low_power_shutdown():
    global _shutdown_armed, _shutdown_low_since
    
    pin_state = GPIO.input(SHUTDOWN_PIN)
    
    # Wait until normal idle state first
    if not _shutdown_armed:
        if pin_state == GPIO.HIGH:
            _shutdown_armed = True
        return
        
    # Active-low trigger
    if pin_state == GPIO.LOW:
        if _shutdown_low_since is None:
            _shutdown_low_since = time.time()
        elif (time.time() - _shutdown_low_since) >= SHUTDOWN_DEBOUNCE_S:
            print("LOW POWER SHUTDOWN TRIGGERED")
            led_off()
            subprocess.run(["sudo","shutdown","-h","now"])
    else:
        _shutdown_low_since = None
        
def checked_sleep(duration):
    end_time = time.time() + duration
    while time.time() < end_time:
        check_for_low_power_shutdown()
        time.sleep(0.1)

# print("WAITING FOR SHUTDOWN TRIGGER.")
# 
# try:
#     time.sleep(1)
#     GPIO.wait_for_edge(SHUTDOWN_PIN, GPIO.FALLING, bouncetime=300)
#     
#     print("SHUTDOWN TRIGGERED.")
#     time.sleep(5)
#     subprocess.run(["sudo","shutdown","-h","now"])
#     
# except KeyboardInterrupt:
#     pass
# 
# finally:
#     GPIO.cleanup()

# ========================================================
# ADC / LoRa INITIALIZATION
# ========================================================
adc = Adafruit_ADS1x15.ADS1115(address=0x48, busnum=1)

CS = digitalio.DigitalInOut(board.CE1)
RESET = digitalio.DigitalInOut(board.D25)
spi = busio.SPI(board.SCK, MOSI=board.MOSI, MISO=board.MISO)

rfm9x = adafruit_rfm9x.RFM9x(spi, CS, RESET, 915.0)

# Gain settings used for ADC
GAIN_ONE = 1
GAIN = GAIN_ONE

# Voltage range lookup for gains
GAIN_VOLTAGE_RANGE = {
    2/3: 6.144,
    1:   4.096,
    2:   2.048,
    4:   1.024,
    8:   0.512,
    16:  0.256
}

fsr = GAIN_VOLTAGE_RANGE[GAIN]
lsb_size = (fsr * 2) / 65536

# Sampling parameters
ADC_DATA_RATE = 250
collect_duration = 10.0
wait_duration = 5.0
thumper_avoidance_duration = 300.0
time_per_sample = 1.0 / ADC_DATA_RATE
threshold_mode = 1.0  # 1.0 for dynamic threshold, 0.0 for static threshold

# CSV logging
SPIKES_LOG_FILE = "fft_spikes.csv"
def init_spike_log():
    with open(SPIKES_LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["Frequency (Hz)", "Spike Type", "Value"])

def log_spike(freq, spike_type, value):
    with open(SPIKES_LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([freq, spike_type, value])

# ========================================================
# LoRa Receive Check
# ========================================================
def check_for_lora_rx():
    try:
        packet = rfm9x.receive(timeout=0.1)
    except Exception as e:
        print("[GEO] RX ERROR:", e)
        return

    if packet is None:
        return

    try:
        text = packet.decode("utf-8", errors="ignore").strip()
    except:
        text = repr(packet)

    print(f"[GEO] LoRa RX: '{text}'")

    if text.startswith(BRAIN_ID):
        print("[GEO] >>> RECEIVED MESSAGE FROM 0.1 (BRAIN) <<<")
        flash_yellow()

# ========================================================
# TEMPLATE MATCHING HELPERS
# ========================================================

def robust_energy_threshold(energies):
    """Return median + k*MAD threshold for positive energies list."""
    if len(energies) < 10:
        return None
    arr = np.array(energies, dtype=float)
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    sigma = 1.4826 * mad
    return med + ENERGY_K_MAD * sigma

def cosine_similarity(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))

def band_features(voltage_window, fs, fmin=FMIN, fmax=FMAX):
    """
    Compute:
      - normalized band power vector for template matching
      - band energy (unnormalized) for gating
      - band freqs (for debugging)
    """
    v = voltage_window - np.mean(voltage_window)
    wv = v * np.hanning(len(v))

    V = np.fft.rfft(wv)
    mag = np.abs(V)
    power = (mag**2) / len(wv)

    freq = np.fft.rfftfreq(len(wv), d=1.0/fs)
    band = (freq >= fmin) & (freq <= fmax)

    p_band = power[band]
    e_band = float(np.sum(p_band))

    p_norm = p_band / (np.sum(p_band) + 1e-12)
    return p_norm, e_band, freq[band]

def generate_synthetic_elephant_walk(fs, duration_s):
    """
    Synthetic "elephant walking" geophone-like voltage signal (Volts).
    Matches the spectrogram character: repeating impulsive ringdowns ~10–40 Hz.
    """
    t = np.arange(0, duration_s, 1/fs)

    T_step = 0.85
    tau_attack = 0.01
    tau_decay = 0.35

    freqs = np.array([14, 18, 26, 34])
    weights = np.array([1.0, 0.8, 0.5, 0.3])

    V0 = 0.012  # ~12 mV peaks

    v = np.zeros_like(t)
    num_steps = int(duration_s / T_step)

    for k in range(num_steps):
        tk = k*T_step + np.random.normal(0, 0.03)
        Ak = V0 * (1 + np.random.normal(0, 0.15))

        dt = t - tk
        active = dt >= 0

        env = (1 - np.exp(-dt[active]/tau_attack)) * np.exp(-dt[active]/tau_decay)

        foot = np.zeros_like(env)
        for f, w in zip(freqs, weights):
            phi = np.random.uniform(0, 2*np.pi)
            foot += w * np.cos(2*np.pi*f*dt[active] + phi)

        v[active] += Ak * env * foot

    noise = 0.0015 * np.random.randn(len(t))
    drift = 0.002 * np.sin(2*np.pi*0.4*t)

    return v + noise + drift

def generate_reference_40hz(fs, duration_s, amplitude_v = 0.01):
    t = np.arange(0, duration_s, 1/fs)
    return amplitude_v * np.sin(2*np.pi*40.0*t)

def build_reference_template(fs, N, hop):
    """
    Reference template = single spectral spike at 40 Hz within [FMIN, FMAX].
    """
    freq = np.fft.rfftfreq(N, d=1.0/fs)
    band  = (freq >= FMIN) & (freq <= FMAX)
    f_band = freq[band]
    
    # Create a delta template at 40 Hz
    ref = np.zeros_like(f_band, dtype=float)
    target_hz = 40.0
    idx = int(np.argmin(np.abs(f_band - target_hz)))
    ref[idx] = 1.0 # All the energy in that bin
    
    return ref, f_band
    
def band_power_spectrum(voltage_window, fs, fmin=FMIN, fmax=FMAX):
    """
    Return (freq_band, power_band, power_band_normalized)
    for a given window
    """
    
    v = voltage_window - np.mean(voltage_window)
    wv = v * np.hanning(len(v))
    V = np.fft.rfft(wv)
    power = (np.abs(V)**2) / len(wv)
    
    freq = np.fft.rfftfreq(len(wv), d=1.0/fs)
    band = (freq >= fmin) & (freq <= fmax)
    
    f_band = freq[band]
    p_band = power[band]
    p_norm = p_band / (np.sum(p_band) + 1e-12)
    return f_band, p_band, p_norm


def build_reference_spectrum_for_plot(fs,N,hop):
    """
    Build a "smooth reference" spectrum by averaging power over multiple
    windows of the synthetic test signal
    Returns (freq_band, ref_power_norm)
    """
    min_duration = max(5.0, (3*N)/fs)
    v_test = generate_reference_40hz(fs,duration_s=5.0, amplitude_v=0.01)
    ref_list = []
    f_band = None
    
    for start in range(0, len(v_test) - N + 1, hop):
        f_band, _, p_norm = band_power_spectrum(v_test[start:start+N], fs)
        ref_list.append(p_norm)
        
    if len(ref_list) == 0:
        if len(v_test) >= N:
            f_band, _, p_norm = band_power_spectrum(v_test[:N],fs)
            return f_band, p_norm
        else:
            raise ValueError(f"Reference build failed: v_test length {len(v_test)} < N {N}")
        
    ref_avg = np.mean(np.vstack(ref_list), axis=0)
    ref_avg = ref_avg / (np.sum(ref_avg) + 1e-12)
    return f_band, ref_avg

    # For "learn", we return None here and fill it later from real data
    return None, None

# ========================================================
# MATCH STATE (baseline + persistence)
# ========================================================

energy_history = deque()          # recent window energies for baseline threshold
energy_time_history = deque()     # timestamps for energy_history
match_times = deque()             # timestamps of successful matches

ref_template = None
ref_freqs = None

# ========================================================
# MAIN LOOP
# ========================================================
init_spike_log()

print("[GEO] STARTED")
print(f"[GEO] Node ID: {DEVICE_ID}, listening for {BRAIN_ID}")
print(f"[GEO] Template match: {USE_TEMPLATE_MATCH}, source={TEMPLATE_SOURCE}")
print(f"[GEO] Spike detector: {USE_SPIKE_DETECTOR}, mode={threshold_mode}")

try:
    while True:
        check_for_low_power_shutdown() 
        check_for_lora_rx()

        print(f"\n[GEO] Collecting {collect_duration}s of data...")
        start_time = time.perf_counter()
        data = []
        timestamps = []

        # -----------------------------
        # ADC reading loop
        # -----------------------------
        while True:
            check_for_low_power_shutdown()
            
            now = time.perf_counter()
            if now - start_time >= collect_duration:
                break

            val = adc.read_adc_difference(0, gain=GAIN, data_rate=ADC_DATA_RATE)
            data.append(val)
            timestamps.append(now)
            time.sleep(time_per_sample)

        print("[GEO] Data collection complete")

        # Actual sample rate check
        diffs = np.diff(timestamps)
        actual_rate = 1.0 / np.mean(diffs)
        print(f"[GEO] Actual Fs: {actual_rate:.2f} Hz")

        voltages = np.array(data) * lsb_size

            # ========================================================
            # Build reference template once we know actual_rate
            # ========================================================
            # Use window/hop in samples based on actual_rate
        N_match = int(round(MATCH_WINDOW_S * actual_rate))
        hop_match = max(1, int(round(MATCH_HOP_S * actual_rate)))

            # Make N a power-of-two-ish for faster FFT, without changing too much
            # (Optional: comment out if you want exact N)
            # closest_pow2 = 2**int(np.round(np.log2(N_match)))
            # N_match = int(closest_pow2)

        if USE_TEMPLATE_MATCH and ref_template is None:
                ref_template, ref_freqs = build_reference_template(actual_rate, N_match, hop_match)
                if TEMPLATE_SOURCE == "synthetic":
                    print(f"[GEO] Built synthetic reference template: N={N_match}, hop={hop_match}, band={FMIN}-{FMAX} Hz")

            # ========================================================
            # 1) EXISTING SPIKE DETECTOR (whole-capture FFT)
            # ========================================================
        spike_confirm = False

        if USE_SPIKE_DETECTOR:
                v_detr = voltages - np.mean(voltages)
                window = np.hanning(len(v_detr))
                wv = v_detr * window

                fft_vals = np.fft.fft(wv)
                magnitude = np.abs(fft_vals)
                power = (magnitude ** 2) / 2

                freq = np.fft.fftfreq(len(magnitude), d=1.0 / actual_rate)
                half = len(freq) // 2

                freq_h = freq[:half]
                magnitude_h = magnitude[:half]
                power_h = power[:half]

                # DYNAMIC THRESHOLDS
                if threshold_mode == 1.0:
                    mag_th = np.median(magnitude_h) + 2.5*np.std(magnitude_h)
                    pow_th = np.median(power_h) + 5*np.std(power_h)
                else:
                    mag_th = 10
                    pow_th = 100

                detected_mag = freq_h[magnitude_h > mag_th]
                detected_mag_vals = magnitude_h[magnitude_h > mag_th]

                detected_pow = freq_h[power_h > pow_th]
                detected_pow_vals = power_h[power_h > pow_th]

                # Magnitude spikes
                if len(detected_mag) > 0:
                    spike_confirm = True
                    print("[GEO] Magnitude spikes:")
                    for f, mv in zip(detected_mag, detected_mag_vals):
                        print(f"  {f:.2f} Hz, Mag={mv:.2f}")
                        log_spike(f, "Magnitude", mv)

                # Power spikes
                if len(detected_pow) > 0:
                    spike_confirm = True
                    print("[GEO] Power spikes:")
                    for f, pv in zip(detected_pow, detected_pow_vals):
                        print(f"  {f:.2f} Hz, Pow={pv:.3f}")
                        log_spike(f, "Power", pv)

            # ========================================================
            # 2) TEMPLATE MATCHING ON SLIDING WINDOWS
            # ========================================================
        template_confirm = False
        best_score = None
        energy_th = None

        if USE_TEMPLATE_MATCH:
                # If learning is requested, do it from this capture (first time only)
                if TEMPLATE_SOURCE == "learn" and ref_template is None:
                    # learn from first ~PERSIST_WINDOW_S seconds worth of windows
                    templates = []
                    end_learn = min(len(voltages), int(PERSIST_WINDOW_S * actual_rate))
                    for start in range(0, end_learn - N_match + 1, hop_match):
                        p_norm, e_band, _ = band_features(voltages[start:start+N_match], actual_rate)
                        templates.append(p_norm)
                    ref_template = np.mean(np.vstack(templates), axis=0)
                    ref_template = ref_template / (np.sum(ref_template) + 1e-12)
                    print(f"[GEO] Learned reference template from real data: N={N_match}, hop={hop_match}, band={FMIN}-{FMAX} Hz")

                if ref_template is not None:
                    now_t = time.time()

                    # Sliding windows over this 10s capture
                    
                    
                    scores = []
                    energies = []
                    
                    best_i = None
                    best_start = None
                    best_pnorm = None
                    best_energy = None
                    
                    i = 0
                    for start in range(0, len(voltages) - N_match + 1, hop_match):
                        win = voltages[start:start+N_match]
                        p_norm, e_band, _ = band_features(win, actual_rate)
                        s = cosine_similarity(p_norm, ref_template)

                        scores.append(s)
                        energies.append(e_band)
                        
                        if(best_score is None) or (s > best_score):
                            best_score = float(s)
                            best_i = i
                            best_start = start
                            best_pnorm = p_norm
                            best_energy = e_band

                        # Maintain baseline energy history with timestamps (for thresholding)
                        energy_history.append(e_band)
                        energy_time_history.append(now_t)
                        
                        i += 1

                    # Trim baseline history to last BASELINE_SECONDS
                    while energy_time_history and (now_t - energy_time_history[0] > BASELINE_SECONDS):
                        energy_time_history.popleft()
                        energy_history.popleft()

                    # Compute robust energy threshold
                    energy_th = robust_energy_threshold(list(energy_history))

                    # Decide match hits using both similarity and energy gate
                    hit_times_local = []
                    for i, (s, e) in enumerate(zip(scores, energies)):
                        # Approx time of window center relative to capture start
                        t_center = (i*hop_match + N_match/2) / actual_rate
                        if energy_th is None:
                            # If not enough baseline yet, only use similarity (more false positives early)
                            hit = (s >= SIM_THRESH)
                        else:
                            hit = (s >= SIM_THRESH) and (e >= energy_th)

                        if hit:
                            hit_times_local.append(time.time() + t_center)

                    if scores:
                        best_score = float(np.max(scores))

                    # Update persistence deque with hits
                    for ht in hit_times_local:
                        match_times.append(ht)

                    # Trim persistence window
                    now2 = time.time()
                    while match_times and (now2 - match_times[0] > PERSIST_WINDOW_S):
                        match_times.popleft()

                    # Confirm if enough hits recently
                    if len(match_times) >= PERSIST_HITS or best_score >= SIM_THRESH: # Testing the OR logic
                        template_confirm = True

                    print(f"[GEO] Template match: best_score={best_score if best_score is not None else float('nan'):.3f}, "
                          f"energy_th={'None' if energy_th is None else f'{energy_th:.3e}'}, "
                          f"hits_in_{PERSIST_WINDOW_S:.0f}s={len(match_times)}")
                    
                trigger = False # Initiaizes the trigger variable
                trigger_reason=[] # Creates an empty array to store the cause of the trigger
                # Useful for knowing if the trigger was due to a spike or a pattern match
                    
                if USE_SPIKE_DETECTOR and spike_confirm:
                    trigger = True
                    trigger_reason.append("SPIKE DETECTION")
                if USE_TEMPLATE_MATCH and template_confirm:
                    trigger = True
                    trigger_reason.append("TEMPLATE MATCH")
                    
                if PLOT_MATCH_DEBUG and trigger:
                    
                    try:
                        if best_start is not None:
                            win_best = voltages[best_start:best_start+N_match]
                        else:
                            win_best = voltages[:N_match]
                            
                        f_real, _, p_real_norm = band_power_spectrum(win_best, actual_rate)
                        
                        if TEMPLATE_SOURCE == "synthetic":
                            f_ref, p_ref_norm = build_reference_spectrum_for_plot(actual_rate, N_match, hop_match)
                        else:
                            f_ref = f_real
                            p_ref_norm = ref_template
                            
                        # Forec safe alignment
                        L = min(len(f_real), len(f_ref), len(p_real_norm))
                        
                        f_plot = f_real[:L]
                        p_real_plot = p_real_norm[:L]
                        p_ref_plot = p_ref_norm[:L]
                        
                        plt.figure(figsize=(10,4))
                        plt.plot(f_plot, p_real_plot, label="Real (best window) - normalized band power")
                        plt.plot(f_plot, p_ref_plot, label="Reference - normalized band power", linestyle="--")
                        #plt.axhline(y=energy_th,label="Mag Threshold",linestyle="--")
                        #plt.axhline(y=mag_th,label="Mag Threshold",linestyle="--")
                        plt.title(f"FFT Template Match (score={best_score:.3f}, energy={best_energy:.3e})")
                        plt.xlabel("Frequency (Hz)")
                        plt.ylabel("Normalized Band Power")
                        plt.grid(True)
                        
                        fname = f"fft_match_{int(time.time())}.png"
                        plt.savefig(fname, dpi=150)
                        plt.close()
                        
                        print(f"[GEO] Saved plot: {fname}")
                        
                    except Exception as e:
                        print("[GEO] Plot error", e)
                    
                                  

            # ========================================================
            # DECIDE IF WE "TRIGGER"
            # ========================================================
        trigger = False
        if USE_SPIKE_DETECTOR and spike_confirm:
                trigger = True
        if USE_TEMPLATE_MATCH and template_confirm:
                trigger = True

            # ========================================================
            # SEND TO BRAIN IF TRIGGER OCCURS
            # ========================================================
        if trigger:
                msg = DEVICE_ID.encode()

                reason_str = " + ".join(trigger_reason) # Creates a string of the trigger reason
                # Will print only one if just one triggers, will print both if both cause a trigger
                print(f"[GEO] TRIGGER=TRUE (REASON = {reason_str}) -> SENDING MESSAGE FROM {DEVICE_ID}...")
                flash_green()

                try:
                    rfm9x.send(msg)
                    print("[GEO] SEND OK")
                except Exception as e:
                    print("[GEO] SEND FAILED:", e)

                check_for_lora_rx()
        else:
                print("[GEO] TRIGGER=FALSE")
                flash_red()

            # ========================================================
            # WAIT LOGIC
            # ========================================================
        if trigger:
                print(f"[GEO] Trigger TRUE. Waiting {thumper_avoidance_duration} seconds...")
                checked_sleep(thumper_avoidance_duration)
        else:
                print(f"[GEO] No trigger. Waiting {wait_duration} seconds...")
                checked_sleep(wait_duration)
                flash_red()
finally:
    print('uhns')
    