#!/usr/bin/env python3
"""
FINAL FRESH CALLER SCRIPT WITH --no-cache TO LOCALLY CALL /free

What it does:
1) Defaults to:
   - steps=150
   - neg_prompt="worst quality, low quality, blurry, nudity, nsfw"
   - openpose=0.95 (enabled)
   - softedge=0.69 (enabled)
2) If --tng is used & no --prompt => picks random TNG scenario (Worf, Riker, Borg, etc.)
   else uses your prompt or a default "An astronaut in space, photorealistic."
3) Additional expansions: --bright, --photo_realistic, --visible_face appended to final prompt.
4) If --no-cache => tries a local "requests.post('http://127.0.0.1:8188/free', ...)" 
   to free ComfyUI's cache, preventing repeated images. 
   (Adjust if your ComfyUI runs at a different address.)
5) Calls `call_rec2emis_comfui.py` with recognized arguments, logs each run to `metadata.csv`.

Usage Examples:
  python fresh_caller.py --tng --bright --num_prompts 6 --no-cache
  => 6 random TNG scenarios, bright expansions, tries to /free ComfyUI cache each run, logs each run.
"""

import argparse
import subprocess
import csv
import os
import random
from datetime import datetime

try:
    import requests
except ImportError:
    requests = None

# Defaults
DEFAULT_NEG = "worst quality, low quality, blurry, nudity, nsfw"
DEFAULT_STEPS = 150
DEFAULT_CFG = 7.0
DEFAULT_OPENPOSE = 0.95
DEFAULT_SOFTEDGE = 0.69

TNG_SCENARIOS = [
    "Commander Data operating a tricorder on the Enterprise bridge",
    "A Borg drone about to assimilate a Starfleet officer in a dark corridor",
    "Riker firing a phaser in a holodeck simulation",
    "Worf battling a Klingon warrior with a Bat'leth",
    "Captain Picard giving a diplomatic speech on the bridge",
]

def main():
    print("[DEBUG] Entered main() in fresh_caller.py")

    parser = argparse.ArgumentParser(
        description="Caller script with TNG expansions, default openpose=0.95, softedge=0.69, steps=150. If --no-cache is used, tries local /free call."
    )
    parser.add_argument("--num_prompts", type=int, default=1,
                        help="How many runs. Default=1.")

    # If user sets --tng but no --prompt => random TNG scenario
    parser.add_argument("--prompt", type=str, default=None,
                        help="Positive prompt. If --tng & no prompt => pick random TNG scenario.")
    parser.add_argument("--neg_prompt", type=str, default=DEFAULT_NEG,
                        help=f"Negative prompt (default='{DEFAULT_NEG}').")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help="Sampling steps (default=150).")
    parser.add_argument("--cfg_scale", type=float, default=DEFAULT_CFG,
                        help="CFG scale (default=7.0).")

    parser.add_argument("--openpose_strength", type=float, default=DEFAULT_OPENPOSE,
                        help="OpenPose strength (default=0.95).")
    parser.add_argument("--softedge_strength", type=float, default=DEFAULT_SOFTEDGE,
                        help="SoftEdge strength (default=0.69).")

    # expansions
    parser.add_argument("--tng", action="store_true",
                        help="If set, TNG style. If no --prompt => random TNG scenario.")
    parser.add_argument("--bright", action="store_true",
                        help="Add 'bright, luminous' expansions.")
    parser.add_argument("--photo_realistic", action="store_true",
                        help="Add 'photo-realistic, extremely detailed...' expansions.")
    parser.add_argument("--visible_face", action="store_true",
                        help="Add 'visible facial features, expressive eyes' expansions.")

    # If user wants to forcibly free ComfyUI's cache
    parser.add_argument("--no-cache", action="store_true",
                        help="Attempt local 'requests.post(http://127.0.0.1:8188/free)' to avoid repeated images")

    parser.add_argument("--base_script", type=str, default="call_rec2emis_comfui.py",
                        help="Base generation script (default=call_rec2emis_comfui.py).")

    args = parser.parse_args()
    print("[DEBUG] Parsed arguments:", args)

    # Decide final user prompt
    if args.tng and (args.prompt is None):
        # random TNG scenario
        chosen = random.choice(TNG_SCENARIOS)
        user_prompt = chosen
        print("[DEBUG] Using random TNG scenario:", chosen)
    else:
        if args.prompt:
            user_prompt = args.prompt
            print("[DEBUG] Using user-provided prompt:", user_prompt)
        else:
            user_prompt = "An astronaut in space, photorealistic"
            print("[DEBUG] Using default prompt (astronaut).")

    # expansions
    expansions = []
    if args.tng:
        expansions.append("Star Trek: The Next Generation style, futuristic starship environment, LCARS aesthetic")
    if args.bright:
        expansions.append("bright, luminous")
    if args.photo_realistic:
        expansions.append("photo-realistic, extremely detailed, cinematic lighting, highly detailed textures, sharp focus")
    if args.visible_face:
        expansions.append("visible facial features, expressive eyes, detailed face")

    expansions_str = ""
    if expansions:
        expansions_str = " " + ", ".join(expansions) + "."

    final_prompt = user_prompt + expansions_str
    print("[DEBUG] final_prompt:", final_prompt)

    print(f"\n[INFO] Running {args.num_prompts} run(s) with steps={args.steps}, openpose={args.openpose_strength}, softedge={args.softedge_strength}.\n")

    for i in range(args.num_prompts):
        print(f"[DEBUG] === Starting run {i+1} / {args.num_prompts} ===")
        print("[DEBUG] Prompt:", final_prompt)
        print("[DEBUG] Negative:", args.neg_prompt)

        # 1) If user wants no-cache => call ComfyUI /free
        if args.no_cache:
            print("[DEBUG] Attempting to call /free endpoint to clear cache.")
            if requests is None:
                print("[WARN] 'requests' module not installed. Can't call /free. Repeated images may occur.")
            else:
                try:
                    payload = {"unload_models": True, "free_memory": True}
                    resp = requests.post("http://127.0.0.1:8188/free", json=payload, timeout=3)
                    if resp.status_code == 200:
                        print("[DEBUG] Successfully called /free to clear ComfyUI cache.")
                    else:
                        print(f"[WARN] /free returned status {resp.status_code}. Possibly no effect.")
                except Exception as e:
                    print(f"[WARN] Could not call /free endpoint. Exception: {e}")

        # 2) Build command for call_rec2emis_comfui.py
        cmd = [
            "python", args.base_script,
            "1",  # single iteration
            "--comfy_prompt", final_prompt,
            "--comfy_neg", args.neg_prompt,
            "--comfy_steps", str(args.steps),
            "--comfy_cfg_scale", str(args.cfg_scale),
            "--openpose_strength", str(args.openpose_strength),
            "--softedge_strength", str(args.softedge_strength),
            "--comfy_run_openpose",
            "--comfy_run_softedge",
        ]

        print("[DEBUG] Subprocess command:", " ".join(cmd))

        # 3) Run
        try:
            completed = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            output_text = completed.stdout

            print("[DEBUG] Subprocess completed. Parsing output for 'Storing outputs in:' line...")

            # 4) Parse "Storing outputs in: <dir>"
            session_dir = None
            for line in output_text.splitlines():
                if "Storing outputs in:" in line:
                    session_dir = line.split("in:")[-1].strip()
                    break

            if session_dir:
                print(f"[DEBUG] Found session_dir: {session_dir}")
                meta_path = os.path.join(session_dir, "metadata.csv")
                file_exists = os.path.isfile(meta_path)
                with open(meta_path, "a", newline="", encoding="utf-8") as cf:
                    writer = csv.writer(cf)
                    if not file_exists:
                        writer.writerow([
                            "timestamp",
                            "prompt",
                            "neg_prompt",
                            "steps",
                            "cfg_scale",
                            "openpose_strength",
                            "softedge_strength",
                            "tng",
                            "bright",
                            "photo_realistic",
                            "visible_face",
                            "no_cache"
                        ])
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    row = [
                        now_str,
                        final_prompt,
                        args.neg_prompt,
                        args.steps,
                        args.cfg_scale,
                        args.openpose_strength,
                        args.softedge_strength,
                        args.tng,
                        args.bright,
                        args.photo_realistic,
                        args.visible_face,
                        args.no_cache
                    ]
                    writer.writerow(row)
                print(f"[DEBUG] Logged run info to {meta_path}")
            else:
                print(f"[WARN] Did not find 'Storing outputs in:' line. No metadata logged.")

        except subprocess.CalledProcessError as cpe:
            print("[ERROR] Subprocess error occurred:", cpe)
            print("[ERROR] Output was:\n", cpe.output)
            break

    print("\n[INFO] All runs complete. If you saw no output, confirm you're running THIS file.\n")

if __name__ == "__main__":
    print("[DEBUG] Invoking main() from fresh_caller.py...")
    main()
