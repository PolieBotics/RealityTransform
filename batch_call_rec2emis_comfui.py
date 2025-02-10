#!/usr/bin/env python3
"""
Batch script that calls `call_rec2emis_comfui.py` multiple times with either:
1) "normal" real-world/historical prompts (when --normal is used), OR
2) the existing random colorful pipeline.

Features:
 - Negative prompt includes "nudity, nsfw"
 - If --normal, pick from a curated 'NORMAL_PROMPTS' list (astronaut, cowboy, etc.), ignoring
   the usual random adjective/character/environment. Also ignore random SFX expansions.
 - If not --normal, we do the same random colorful flow (ADJECTIVES, CHARACTERS, etc.)
   and optional expansions like TNG, bright, random SFX.
 - ControlNets can still be toggled/used.

We log all run parameters into `metadata.csv` in the output directory
that `call_rec2emis_comfui.py` prints (line containing "Storing outputs in: ...").
"""

import subprocess
import random
import argparse
import csv
import os
from datetime import datetime

###############################################################################
# Word lists & base prompt defaults
###############################################################################

ADJECTIVES = [
    "fiery", "glowing", "psychedelic", "radiant", "vibrant", "iridescent",
    "shimmering", "electric", "mysterious", "bioluminescent", "kaleidoscopic",
    "pulsating", "luminous", "sparkling", "neon-infused", "fantastical",
    "surreal", "glittering", "hyper-colorful", "multichromatic", "chromatic",
    "astral", "galactic", "soaring", "hovering"
]

CHARACTERS = [
    "fire monster", "steampunk wizard", "cyberpunk dragon", "cosmic mermaid",
    "glowing fairy", "solar phoenix", "neon samurai", "psychedelic alien",
    "floating jellyfish king", "rainbow djinn", "electric eel warrior",
    "kaleidoscope butterfly spirit", "mystic geisha from space",
    "crystal golem", "slime shapeshifter", "prismatic centaur",
    "bio-mech angel", "vaporwave knight", "cosmic clown", "astral witch",
]

ENVIRONMENTS = [
    "lush jungle", "crystal cave", "underwater kingdom of red corals",
    "futuristic neon cityscape", "floating sky island", "holographic desert",
    "alien rainforest", "vibrant coral reef", "glowing mushroom forest",
    "cloud palace above a sea of stars", "kaleidoscope canyon",
    "pink volcanic landscape", "psychedelic candy land", "mirrored labyrinth",
    "cosmic nebula", "techno temple", "cybernetic cathedral",
    "starry void", "flaming ruins"
]

STYLES = [
    "in the style of anime", "fantasy illustration style", "psychedelic art style",
    "vibrant street art style", "glossy 3D render style", "bright neon design",
    "fantastical stained-glass style", "glitter-splashed illustration",
    "liquid paint swirling style", "kawaii pastel style", "futuristic Tron aesthetic",
    "cel-shaded style", "epic matte painting style", "concept art style"
]

# Negative prompt includes "nudity, nsfw"
DEFAULT_NEG_PROMPT = "worst quality, low quality, blurry, nudity, nsfw"

# Steps & CFG defaults
DEFAULT_STEPS = 30
DEFAULT_CFG_SCALE = 7.0

# ControlNet toggles
DEFAULT_RUN_SOFTEDGE = False
DEFAULT_RUN_OPENPOSE = False
DEFAULT_RUN_CANNY = False

# Default strengths
DEFAULT_SOFTEDGE_STRENGTH = 0.69
DEFAULT_OPENPOSE_STRENGTH = 0.95
DEFAULT_CANNY_STRENGTH = 0.9

# For canny, thresholds
DEFAULT_CANNY_THRESHOLD1 = 100
DEFAULT_CANNY_THRESHOLD2 = 200

# For openpose
DEFAULT_OPENPOSE_END_PERCENT = 0.95

###############################################################################
# "Normal" prompt list
###############################################################################
NORMAL_PROMPTS = [
    "An astronaut in space, photorealistic",
    "A cowboy in the old west, photorealistic",
    "A scuba diver underwater in a colorful coral reef, photorealistic",
    "A pirate on a pirate ship, realistic, cinematic lighting",
    "A ninja in ancient Japan, realistic style, moody lighting",
    "A medieval knight in a grand castle hall, realistic style",
    "A Victorian lady in a lavish ballroom, photorealistic",
    "A gangster in 1920s Chicago, photorealistic, film noir lighting",
    "A Roman soldier in an ancient colosseum, realistic style",
    "A Viking warrior on a stormy sea, cinematic realism",
    "A samurai in feudal Japan, realistic style, detailed armor",
    "An explorer in the Sahara desert, photorealistic, bright sunlight",
    "A dancer in a grand opera house, realistic style",
]

###############################################################################
# Additional expansions
###############################################################################

PHOTO_REALISTIC_PHRASES = [
    "photo-realistic", "extremely detailed", "cinematic lighting",
    "highly detailed textures", "sharp focus"
]

DETAILED_CLOTHES_PHRASES = [
    "intricate, detailed clothing", "visible stitch work", "realistic fabric rendering"
]

VISIBLE_FACE_PHRASES = [
    "visible facial features", "expressive eyes", "detailed face"
]

TNG_PHRASES = [
    "Star Trek: The Next Generation style", 
    "futuristic starship environment",
    "LCARS aesthetic"
]

BRIGHT_PHRASES = ["bright", "luminous"]

SFX_EXPANSIONS = [
    "fireballs shooting from their hands",
    "sparks of lightning arcing between their fingertips",
    "a glowing halo of light around their head",
    "energy swirling around their body",
    "sparkling magical dust swirling around them",
    "shooting beams of cosmic light from their palms"
]

###############################################################################
# Helpers
###############################################################################

def random_strength_around(mid, stdev=0.1, minimum=0.3, maximum=1.0):
    """Random strength near 'mid' (gaussian) clamped to [min,max]."""
    while True:
        val = random.gauss(mid, stdev)
        if minimum <= val <= maximum:
            return round(val, 2)

def generate_random_base():
    """
    Create a random base prompt from the colorful pipeline
    (adjective, character, environment, style).
    """
    adj = random.choice(ADJECTIVES)
    char = random.choice(CHARACTERS)
    env = random.choice(ENVIRONMENTS)
    style = random.choice(STYLES)
    return f"A {adj}, {char} in a {env}, {style}. Extremely colorful and cinematic!"

def pick_normal_prompt():
    """
    Pick one scenario from NORMAL_PROMPTS (pirates, ninjas, cowboys, etc.).
    """
    return random.choice(NORMAL_PROMPTS)

def add_extra_phrases(base_prompt,
                      photo_realistic=False,
                      detailed_clothes=False,
                      visible_face=False,
                      tng=False,
                      bright=False):
    """
    Append extra descriptive phrases if flags are set.
    """
    expansions = []
    if photo_realistic:
        expansions += PHOTO_REALISTIC_PHRASES
    if detailed_clothes:
        expansions += DETAILED_CLOTHES_PHRASES
    if visible_face:
        expansions += VISIBLE_FACE_PHRASES
    if tng:
        expansions += TNG_PHRASES
    if bright:
        expansions += BRIGHT_PHRASES

    if expansions:
        extra_str = ", ".join(expansions)
        base_prompt += f" {extra_str}."
    return base_prompt

def maybe_add_random_sfx(prompt, use_sfx):
    """
    If use_sfx=True, pick exactly one random special effect from SFX_EXPANSIONS
    and append it. 
    """
    if use_sfx:
        sfx = random.choice(SFX_EXPANSIONS)
        prompt += f" {sfx}."
    return prompt

def build_command(
    base_script,
    prompt,
    neg_prompt,
    steps,
    cfg_scale,
    run_softedge,
    run_openpose,
    run_canny,
    softedge_strength,
    openpose_strength,
    canny_threshold1,
    canny_threshold2,
    canny_strength,
    openpose_end_percent
):
    """
    Construct the final command for `call_rec2emis_comfui.py`.
    """
    cmd = [
        "python", base_script,
        "1",
        "--comfy_prompt", prompt,
        "--comfy_neg", neg_prompt,
        "--comfy_steps", str(steps),
        "--comfy_cfg_scale", str(cfg_scale),
        "--softedge_strength", str(softedge_strength),
        "--openpose_strength", str(openpose_strength),
        "--canny_threshold1", str(canny_threshold1),
        "--canny_threshold2", str(canny_threshold2),
        "--canny_strength", str(canny_strength),
        "--openpose_end_percent", str(openpose_end_percent),
    ]
    if run_softedge:
        cmd.append("--comfy_run_softedge")
    if run_openpose:
        cmd.append("--comfy_run_openpose")
    if run_canny:
        cmd.append("--comfy_run_canny")
    return cmd

###############################################################################
# Main
###############################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Batch caller for call_rec2emis_comfui.py with normal or colorful prompts."
    )
    parser.add_argument("--num_prompts", type=int, default=5, help="How many prompts to generate.")
    parser.add_argument("--base_script", type=str, default="call_rec2emis_comfui.py")
    parser.add_argument("--neg_prompt", type=str, default=DEFAULT_NEG_PROMPT)
    parser.add_argument("--comfy_steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--comfy_cfg_scale", type=float, default=DEFAULT_CFG_SCALE)

    # Normal vs. colorful
    parser.add_argument("--normal", action="store_true",
                        help="If set, pick from normal/historical prompts (pirates, ninjas, etc.)")

    # ControlNet toggles
    parser.add_argument("--run_softedge", action="store_true", default=DEFAULT_RUN_SOFTEDGE)
    parser.add_argument("--run_openpose", action="store_true", default=DEFAULT_RUN_OPENPOSE)
    parser.add_argument("--run_canny", action="store_true", default=DEFAULT_RUN_CANNY)

    # "Monte carlo" for SoftEdge / OpenPose
    parser.add_argument("--monte_carlo_softedge", action="store_true")
    parser.add_argument("--monte_carlo_openpose", action="store_true")

    # ControlNet param defaults
    parser.add_argument("--softedge_strength", type=float, default=DEFAULT_SOFTEDGE_STRENGTH)
    parser.add_argument("--openpose_strength", type=float, default=DEFAULT_OPENPOSE_STRENGTH)
    parser.add_argument("--canny_strength", type=float, default=DEFAULT_CANNY_STRENGTH)
    parser.add_argument("--canny_threshold1", type=int, default=DEFAULT_CANNY_THRESHOLD1)
    parser.add_argument("--canny_threshold2", type=int, default=DEFAULT_CANNY_THRESHOLD2)
    parser.add_argument("--openpose_end_percent", type=float, default=DEFAULT_OPENPOSE_END_PERCENT)

    # Photo-realistic expansions
    parser.add_argument("--photo_realistic", action="store_true")
    parser.add_argument("--detailed_clothes", action="store_true")
    parser.add_argument("--visible_face", action="store_true")

    # Additional expansions
    parser.add_argument("--tng", action="store_true",
                        help="Add a Star Trek TNG style expansion.")
    parser.add_argument("--bright", action="store_true",
                        help="Add 'bright, luminous' expansions.")
    parser.add_argument("--random_sfx", action="store_true",
                        help="Pick one random special effect each run (fireballs, lightning, etc.)")

    args = parser.parse_args()

    print(f"Generating {args.num_prompts} prompt(s)...")

    for i in range(args.num_prompts):
        # Decide if normal or colorful
        if args.normal:
            # Just pick from normal list
            base_prompt = pick_normal_prompt()
            # We skip random SFX in normal mode for more realism, 
            # but if you still want it, remove this if-check.
            use_sfx = False
        else:
            # The original colorful pipeline
            base_prompt = generate_random_base()
            use_sfx = args.random_sfx

        # Possibly add expansions (photo_realistic, tng, bright, etc.)
        with_expansions = add_extra_phrases(
            base_prompt,
            photo_realistic=args.photo_realistic,
            detailed_clothes=args.detailed_clothes,
            visible_face=args.visible_face,
            tng=args.tng,
            bright=args.bright
        )

        # If not normal or if you want to allow SFX in normal, we do:
        # final_prompt = maybe_add_random_sfx(with_expansions, use_sfx)
        # For now, skip SFX if normal is set:
        if args.normal:
            final_prompt = with_expansions
        else:
            final_prompt = maybe_add_random_sfx(with_expansions, use_sfx)

        steps = args.comfy_steps
        cfg_scale = args.comfy_cfg_scale

        # ControlNet toggles
        run_softedge = args.run_softedge
        run_openpose = args.run_openpose
        run_canny = args.run_canny

        # Strengths
        softedge_strength = args.softedge_strength
        openpose_strength = args.openpose_strength
        canny_strength = args.canny_strength
        canny_threshold1 = args.canny_threshold1
        canny_threshold2 = args.canny_threshold2
        openpose_end_percent = args.openpose_end_percent

        # If user wants random softedge => force it on
        if args.monte_carlo_softedge:
            run_softedge = True
            softedge_strength = random_strength_around(0.69, stdev=0.1, minimum=0.3, maximum=1.0)

        # If user wants random openpose => force it on
        if args.monte_carlo_openpose:
            run_openpose = True
            openpose_strength = random_strength_around(0.95, stdev=0.02, minimum=0.8, maximum=1.0)

        # Print summary
        print(f"\n=== Prompt #{i+1} ===")
        print(f"Prompt: {final_prompt}")
        print(f"Negative: {args.neg_prompt}")
        print(f"Steps: {steps}, CFG: {cfg_scale}")
        print(f"SoftEdge={run_softedge} (strength={softedge_strength}), OpenPose={run_openpose} (strength={openpose_strength}), Canny={run_canny}")
        print(f"Canny thresholds=({canny_threshold1},{canny_threshold2}), canny_strength={canny_strength}")

        # Build command
        cmd = build_command(
            base_script=args.base_script,
            prompt=final_prompt,
            neg_prompt=args.neg_prompt,
            steps=steps,
            cfg_scale=cfg_scale,
            run_softedge=run_softedge,
            run_openpose=run_openpose,
            run_canny=run_canny,
            softedge_strength=softedge_strength,
            openpose_strength=openpose_strength,
            canny_threshold1=canny_threshold1,
            canny_threshold2=canny_threshold2,
            canny_strength=canny_strength,
            openpose_end_percent=openpose_end_percent
        )

        print("Command:", " ".join(cmd))

        # Execute script & capture output
        try:
            completed = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            output_text = completed.stdout

            # Parse line "Storing outputs in: <dir>"
            session_dir = None
            for line in output_text.splitlines():
                if "Storing outputs in:" in line:
                    session_dir = line.split("in:")[-1].strip()
                    break

            # Log a row in metadata.csv if we found the folder
            if session_dir:
                metadata_path = os.path.join(session_dir, "metadata.csv")
                file_exists = os.path.isfile(metadata_path)
                with open(metadata_path, "a", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    if not file_exists:
                        writer.writerow([
                            "timestamp",
                            "prompt",
                            "neg_prompt",
                            "steps",
                            "cfg_scale",
                            "run_softedge",
                            "softedge_strength",
                            "run_openpose",
                            "openpose_strength",
                            "run_canny",
                            "canny_threshold1",
                            "canny_threshold2",
                            "canny_strength",
                            "openpose_end_percent",
                            "normal_mode"
                        ])
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    row = [
                        now_str,
                        final_prompt,
                        args.neg_prompt,
                        steps,
                        cfg_scale,
                        run_softedge,
                        softedge_strength,
                        run_openpose,
                        openpose_strength,
                        run_canny,
                        canny_threshold1,
                        canny_threshold2,
                        canny_strength,
                        openpose_end_percent,
                        args.normal
                    ]
                    writer.writerow(row)
                print(f"[{i+1}] Logged parameters to {metadata_path}")
            else:
                print(f"[{i+1}] Warning: could not parse output dir from script. No metadata logged.")

        except subprocess.CalledProcessError as cpe:
            print("Error running command:", cpe)
            print("Output was:\n", cpe.output)
            break

    print("\nAll batch runs completed. Enjoy your results!")

if __name__ == "__main__":
    main()
