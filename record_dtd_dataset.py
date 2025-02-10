import sys
import os
import time
import cv2
import numpy as np
import pygame
from PIL import Image
from datetime import datetime
import subprocess  # ADDED for DTD download/unpack

if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':0'

sys.path.append("../python-common")
import TIS

#############################################
# Adjustable parameters
#############################################
latency_period = 2.4   # seconds (2400 ms)
warmup_time = 3.0      # 3 seconds warm-up
gain_value = 24        # camera gain
input_dir = "input_datasets"

#############################################
# Functions
#############################################
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def download_and_unpack_dtd():  # RE-INSERTED FUNCTION
    url = "https://www.robots.ox.ac.uk/~vgg/data/dtd/releases/dtd-r1.0.1.tar.gz"
    tar_filename = "dtd-r1.0.1.tar.gz"
    if not os.path.exists(tar_filename):
        subprocess.run(["wget", url], check=True)
    subprocess.run(["tar", "xzf", tar_filename], check=True)
    if not os.path.exists("input_datasets"):
        os.makedirs("input_datasets")
    if not os.path.exists("input_datasets/images"):
        os.rename("dtd/images", "input_datasets/images")

class CustomData:
    def __init__(self):
        self.busy = False
        self.latest_image = None
        self.new_image_available = False

def on_new_image(tis, userdata):
    if userdata.busy:
        return
    userdata.busy = True
    image = tis.get_image()
    userdata.latest_image = image.copy()
    userdata.new_image_available = True
    userdata.busy = False

def capture_image_if_available(CD, timeout=2.0):
    start = time.perf_counter()
    while (time.perf_counter() - start) < timeout:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key in [pygame.K_q, pygame.K_ESCAPE]:
                return None
        if CD.new_image_available and CD.latest_image is not None:
            img = CD.latest_image
            CD.new_image_available = False
            return img
        time.sleep(0.01)
    return None

def find_images_in_input():
    images = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                images.append(os.path.join(root, f))
    return images

#############################################
# Main
#############################################
if __name__ == "__main__":
    images = find_images_in_input()
    if len(images) == 0:
        print("No images found in input_datasets. Attempting to download DTD.")
        download_and_unpack_dtd()  # Attempt to download/unpack DTD
        images = find_images_in_input()  # Re-check after downloading DTD
        if len(images) == 0:
            print("Still no images found after attempting to download DTD. Please add images and rerun.")
            sys.exit(0)

    # Ensure saved_datasets directory exists
    saved_datasets_dir = "saved_datasets"
    ensure_dir(saved_datasets_dir)

    # Create run directory inside saved_datasets
    run_dir = os.path.join(saved_datasets_dir, datetime.now().strftime("%Y%m%d_%H%M%S_run"))
    ensure_dir(run_dir)
    emissions_dir = os.path.join(run_dir, "Emissions")
    recordings_dir = os.path.join(run_dir, "Recordings")
    ensure_dir(emissions_dir)
    ensure_dir(recordings_dir)

    # Camera setup
    Tis = TIS.TIS()
    Tis.open_device("25420561", 5320, 4600, "15/1", TIS.SinkFormats.BGRA, False)
    CD = CustomData()
    Tis.set_image_callback(on_new_image, CD)
    Tis.set_property("TriggerMode", "On")
    Tis.set_property("GainAuto", "Off")
    Tis.set_property("Gain", gain_value)   # gain set to 30
    Tis.set_property("ExposureAuto", "Off")
    Tis.set_property("ExposureTime", 16667)
    try:
        Tis.set_property("BalanceWhiteAuto", "Off")
        Tis.set_property("BalanceWhiteRed", 1.2)
        Tis.set_property("BalanceWhiteGreen", 1.0)
        Tis.set_property("BalanceWhiteBlue", 1.4)
    except:
        pass

    Tis.start_pipeline()

    pygame.init()
    pygame.font.init()
    width, height = 1920, 1080
    flags = pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF
    screen = pygame.display.set_mode((width, height), flags)
    pygame.mouse.set_visible(False)

    print(f"Warming up for {warmup_time} seconds...")
    start_time = time.perf_counter()
    running = True
    while running and (time.perf_counter() - start_time < warmup_time):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key in [pygame.K_q, pygame.K_ESCAPE]:
                running = False

    if not running:
        Tis.stop_pipeline()
        pygame.quit()
        print("Program end (quit during warmup).")
        sys.exit(0)

    images.sort()
    print(f"Processing images with latency_period = {latency_period}s")

    for img_path in images:
        # Check quit
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key in [pygame.K_q, pygame.K_ESCAPE]:
                running = False
                break
        if not running:
            break

        # Load and upscale
        pil_img = Image.open(img_path).convert("RGB")
        pil_upscaled = pil_img.resize((1920,1080), Image.Resampling.LANCZOS)

        # Display emission
        upscaled_np = np.array(pil_upscaled)
        surf = pygame.surfarray.make_surface(upscaled_np.swapaxes(0,1))
        screen.blit(surf, (0,0))
        pygame.display.flip()

        # Save emission with original filename
        base_name = os.path.basename(img_path)
        emission_filename = os.path.join(emissions_dir, base_name)
        pil_upscaled.save(emission_filename)

        # Wait latency_period
        end_wait = time.perf_counter() + latency_period
        while time.perf_counter() < end_wait:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key in [pygame.K_q, pygame.K_ESCAPE]:
                    running = False
                    break
            if not running:
                break
        if not running:
            break

        # Capture image from camera
        captured_img = capture_image_if_available(CD, timeout=2.0)
        if captured_img is None:
            print(f"No image captured for {base_name}, skipping recording.")
            continue

        # Save recording with original filename
        recording_filename = os.path.join(recordings_dir, base_name)
        bgr = cv2.cvtColor(captured_img, cv2.COLOR_BGRA2BGR)
        cv2.imwrite(recording_filename, bgr)

        print(f"Processed {img_path}: Emission={emission_filename}, Recording={recording_filename}")

    Tis.stop_pipeline()
    pygame.quit()

    print("All images processed.")
    print("Program end")
