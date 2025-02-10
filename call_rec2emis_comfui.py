#!/usr/bin/env python3
import sys
import os
import time
import cv2
import numpy as np
import pygame
from PIL import Image
from datetime import datetime

# TIS camera code (adjust path if needed)
sys.path.append("../python-common")
import TIS

# 1) Import the Pix2PixHD module
from pix2pixhd_infer_module import Pix2PixHDInference

# 2) Import the ControlNet module (with generate_image function)
from controlnet_module import generate_image

###############################################################################
# DEFAULTS: Easy to edit at the top
###############################################################################

# -------------- Pygame + Camera --------------
DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 1080
CENTER_CROP_RATIO = 0.75       # vertically keep center 75%
PAUSE_TIME_WHITE = 2.4         # wait after projecting white
PAUSE_TIME_COMFYUI = 2.4       # wait after projecting comfyui result
CAMERA_GAIN = 29
CAMERA_EXPOSURE = 16667

# -------------- Pix2PixHD --------------
PIX_NAME = "reality_transform_rec2emis"
PIX_CHECKPOINTS_DIR = "./checkpoints"
PIX_WHICH_EPOCH = "latest"
PIX_GPU_IDS = "0"

# Force final dimension for pix2pixHD input
FINAL_IN_WIDTH = 2048
FINAL_IN_HEIGHT = 1024

# -------------- ComfyUI --------------
COMFY_SERVER = "127.0.0.1:8188"
COMFY_PROMPT = "Leaves."
COMFY_NEG = "worst quality, low quality, blurry"
COMFY_STEPS = 50
COMFY_CFG_SCALE = 7.0

# ControlNet toggles
COMFY_RUN_CANNY = False
COMFY_RUN_SOFTEDGE = True
COMFY_RUN_OPENPOSE = False

# Optional ControlNet parameters
CANNY_THRESHOLD1 = 100
CANNY_THRESHOLD2 = 200
CANNY_STRENGTH = 0.9
SOFTEDGE_STRENGTH = .85
OPENPOSE_STRENGTH = 0.9
OPENPOSE_END_PERCENT = 0.95

# ControlNet model paths
CANNY_MODEL = "1.5/control_v11p_sd15_canny_fp16.safetensors"
SOFTEDGE_MODEL = "1.5/control_v11p_sd15_softedge_fp16.safetensors"
OPENPOSE_MODEL = "1.5/control_v11p_sd15_openpose_fp16.safetensors"

# Output
OUTPUT_DIR = "combined_outputs"


###############################################################################
# Utility
###############################################################################
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':0'

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class CustomData:
    """Shared data for TIS camera callback."""
    def __init__(self):
        self.busy = False
        self.latest_image = None
        self.new_image_available = False

def on_new_image(tis, userdata):
    """Callback storing the newest image from TIS camera."""
    if userdata.busy:
        return
    userdata.busy = True
    image = tis.get_image()
    userdata.latest_image = image.copy()
    userdata.new_image_available = True
    userdata.busy = False

def capture_image_if_available(CD, timeout=2.0):
    """
    Poll for up to `timeout` seconds to grab a frame from the camera.
    If user presses Q or ESC, return None to signal exit.
    """
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

def vertical_center_crop(pil_img, ratio=1.0):
    """
    Crop the middle portion of the image vertically by `ratio`.
    e.g. ratio=0.75 => keep the central 75% of height.
    """
    if ratio >= 1.0:
        return pil_img
    w, h = pil_img.size
    new_h = int(h * ratio)
    top = (h - new_h)//2
    return pil_img.crop((0, top, w, top + new_h))

###############################################################################
# Combined Caller
###############################################################################
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Project white -> capture -> pix2pixHD -> ComfyUI -> project -> capture. Single or multi iteration."
    )
    parser.add_argument("iterations", nargs="?", default=1, type=int,
                        help="Number of times to run the pipeline. Default=1.")

    # ---------------- Pix2PixHD ----------------
    parser.add_argument("--pix_name", type=str, default=PIX_NAME)
    parser.add_argument("--pix_checkpoints_dir", type=str, default=PIX_CHECKPOINTS_DIR)
    parser.add_argument("--pix_which_epoch", type=str, default=PIX_WHICH_EPOCH)
    parser.add_argument("--pix_gpu_ids", type=str, default=PIX_GPU_IDS)

    # ---------------- ComfyUI Basic ----------------
    parser.add_argument("--comfy_server", type=str, default=COMFY_SERVER,
                        help="Host:port for ComfyUI server")
    parser.add_argument("--comfy_prompt", type=str, default=COMFY_PROMPT,
                        help="Positive prompt for ComfyUI")
    parser.add_argument("--comfy_neg", type=str, default=COMFY_NEG,
                        help="Negative prompt for ComfyUI")
    parser.add_argument("--comfy_steps", type=int, default=COMFY_STEPS)
    parser.add_argument("--comfy_cfg_scale", type=float, default=COMFY_CFG_SCALE)

    # ---------------- ComfyUI ControlNet toggles ----------------
    parser.add_argument("--comfy_run_canny", action="store_true", default=COMFY_RUN_CANNY)
    parser.add_argument("--comfy_run_softedge", action="store_true", default=COMFY_RUN_SOFTEDGE)
    parser.add_argument("--comfy_run_openpose", action="store_true", default=COMFY_RUN_OPENPOSE)

    # ---------------- ControlNet parameters ----------------
    parser.add_argument("--canny_threshold1", type=int, default=CANNY_THRESHOLD1)
    parser.add_argument("--canny_threshold2", type=int, default=CANNY_THRESHOLD2)
    parser.add_argument("--canny_strength", type=float, default=CANNY_STRENGTH)
    parser.add_argument("--softedge_strength", type=float, default=SOFTEDGE_STRENGTH)
    parser.add_argument("--openpose_strength", type=float, default=OPENPOSE_STRENGTH)
    parser.add_argument("--openpose_end_percent", type=float, default=OPENPOSE_END_PERCENT)

    # ---------------- ControlNet model paths ----------------
    parser.add_argument("--canny_model", type=str, default=CANNY_MODEL)
    parser.add_argument("--softedge_model", type=str, default=SOFTEDGE_MODEL)
    parser.add_argument("--openpose_model", type=str, default=OPENPOSE_MODEL)

    # ---------------- Misc Output ----------------
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)

    args = parser.parse_args()
    N = args.iterations

    # 1) Setup pix2pixHD inference
    pix2pix_model = Pix2PixHDInference(
        name=args.pix_name,
        checkpoints_dir=args.pix_checkpoints_dir,
        which_epoch=args.pix_which_epoch,
        gpu_ids=args.pix_gpu_ids,
    )

    # 2) Setup TIS camera
    Tis = TIS.TIS()
    Tis.open_device("25420561", 5320, 4600, "15/1", TIS.SinkFormats.BGRA, False)
    CD = CustomData()
    Tis.set_image_callback(on_new_image, CD)
    # Camera settings
    Tis.set_property("TriggerMode", "On")
    Tis.set_property("GainAuto", "Off")
    Tis.set_property("Gain", CAMERA_GAIN)
    Tis.set_property("ExposureAuto", "Off")
    Tis.set_property("ExposureTime", CAMERA_EXPOSURE)
    try:
        Tis.set_property("BalanceWhiteAuto", "Off")
        Tis.set_property("BalanceWhiteRed", 1.2)
        Tis.set_property("BalanceWhiteGreen", 1.0)
        Tis.set_property("BalanceWhiteBlue", 1.4)
    except:
        pass
    Tis.start_pipeline()

    # 3) Fullscreen pygame
    pygame.init()
    pygame.font.init()
    flags = pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF
    screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT), flags)
    pygame.mouse.set_visible(False)

    # White surface
    white_surf = pygame.Surface((DISPLAY_WIDTH, DISPLAY_HEIGHT))
    white_surf.fill((255,255,255))

    # 4) Single time-stamped output directory
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_combined")
    run_dir = os.path.join(args.output_dir, f"{timestamp_str}_N{N}")
    ensure_dir(run_dir)

    print(f">>> Running the combined pipeline for N={N} iteration(s).")
    print(f">>> Storing outputs in: {run_dir}\n")

    iteration = 0
    running = True

    while running and iteration < N:
        iteration += 1

        # A) PROJECT WHITE
        screen.blit(white_surf, (0,0))
        pygame.display.flip()
        
        # Wait
        t_end = time.perf_counter() + PAUSE_TIME_WHITE
        while time.perf_counter() < t_end:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key in [pygame.K_q, pygame.K_ESCAPE]:
                    running = False
                    break
            if not running:
                break
        
        if not running:
            break

        # B) CAPTURE => "step_response"
        step_img_np = capture_image_if_available(CD, timeout=2.0)
        if step_img_np is None:
            print(f"[Iter {iteration}] No capture. Exiting.")
            break

        bgr_img = cv2.cvtColor(step_img_np, cv2.COLOR_BGRA2BGR)
        pil_in = Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))

        # Center-crop + final resize
        if CENTER_CROP_RATIO < 1.0:
            pil_in = vertical_center_crop(pil_in, CENTER_CROP_RATIO)
        pil_in = pil_in.resize((FINAL_IN_WIDTH, FINAL_IN_HEIGHT), Image.LANCZOS)

        step_file = os.path.join(run_dir, f"iter_{iteration:03d}_step_response.png")
        pil_in.save(step_file)
        print(f"[Iter {iteration}] Saved step_response => {step_file}")

        # C) RUN PIX2PIXHD
        pix_out = pix2pix_model.run_inference(pil_in)
        if pix_out is None:
            print(f"[Iter {iteration}] Pix2PixHD inference returned None. Exiting.")
            break

        pix_file = os.path.join(run_dir, f"iter_{iteration:03d}_pix2pixhd.png")
        pix_out.save(pix_file)
        print(f"[Iter {iteration}] Saved pix2pixhd => {pix_file}")

        # D) RUN COMFYUI with the pix2pixhd output as input
        import io
        buf = io.BytesIO()
        pix_out.save(buf, format="PNG")
        input_image_bytes = buf.getvalue()

        comfy_result_bytes = generate_image(
            server_address=args.comfy_server,
            pos_prompt=args.comfy_prompt,
            neg_prompt=args.comfy_neg,
            steps=args.comfy_steps,
            cfg_scale=args.comfy_cfg_scale,
            input_image_bytes=input_image_bytes,
            output_dir=run_dir,  # store ComfyUI's final in same directory

            # ControlNet toggles
            run_canny=args.comfy_run_canny,
            run_softedge=args.comfy_run_softedge,
            run_openpose=args.comfy_run_openpose,

            # ControlNet parameters
            canny_threshold1=args.canny_threshold1,
            canny_threshold2=args.canny_threshold2,
            canny_strength=args.canny_strength,
            softedge_strength=args.softedge_strength,
            openpose_strength=args.openpose_strength,
            openpose_end_percent=args.openpose_end_percent,

            # ControlNet model paths
            controlnet_canny=args.canny_model,
            controlnet_softedge=args.softedge_model,
            controlnet_openpose=args.openpose_model
        )

        # E) LOAD COMFYUI result as PIL
        comfy_iteration_png = os.path.join(run_dir, f"iter_{iteration:03d}_comfyui.png")
        with open(comfy_iteration_png, "wb") as f:
            f.write(comfy_result_bytes)
        print(f"[Iter {iteration}] ComfyUI output => {comfy_iteration_png}")

        comfy_pil = Image.open(io.BytesIO(comfy_result_bytes)).convert("RGB")

        # F) PROJECT the ComfyUI result
        if comfy_pil.size != (DISPLAY_WIDTH, DISPLAY_HEIGHT):
            disp_comfy = comfy_pil.resize((DISPLAY_WIDTH, DISPLAY_HEIGHT), Image.LANCZOS)
        else:
            disp_comfy = comfy_pil

        arr_comfy = np.array(disp_comfy)
        surf_comfy = pygame.surfarray.make_surface(arr_comfy.swapaxes(0,1))
        screen.blit(surf_comfy, (0,0))
        pygame.display.flip()

        # G) Wait a bit, then capture final
        t_end2 = time.perf_counter() + PAUSE_TIME_COMFYUI
        while time.perf_counter() < t_end2:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key in [pygame.K_q, pygame.K_ESCAPE]:
                    running = False
                    break
            if not running:
                break

        if not running:
            break

        # H) Capture final -> "final_capture"
        final_np = capture_image_if_available(CD, timeout=2.0)
        if final_np is None:
            print(f"[Iter {iteration}] No final capture. Exiting.")
            break

        final_bgr = cv2.cvtColor(final_np, cv2.COLOR_BGRA2BGR)
        final_pil = Image.fromarray(cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB))

        final_path = os.path.join(run_dir, f"iter_{iteration:03d}_final_capture.png")
        final_pil.save(final_path)
        print(f"[Iter {iteration}] Saved final capture => {final_path}\n")

    # Cleanup
    Tis.stop_pipeline()
    pygame.quit()
    print("All done. Goodbye.")

if __name__ == "__main__":
    main()
