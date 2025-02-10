#!/usr/bin/env python3

import websocket
import uuid
import json
import urllib.request
import time
import os
import re
import tempfile
import csv
from datetime import datetime
import requests  # Needed for clearing the cache via /free endpoint

def debug_print(*args):
    print("[DEBUG]", *args, flush=True)

def sanitize_prompt(prompt, max_len=50):
    """
    Create a filesystem-safe string from the prompt.
    """
    sanitized = re.sub(r'[^a-zA-Z0-9\\s_-]+', '', prompt)
    sanitized = sanitized.lower()
    sanitized = re.sub(r'\\s+', '_', sanitized)
    if len(sanitized) > max_len:
        sanitized = sanitized[:max_len]
    return sanitized

def queue_prompt(prompt_graph, server_address, client_id):
    """
    Send the prompt-graph to the ComfyUI queue endpoint.
    """
    try:
        data = json.dumps({"prompt": prompt_graph, "client_id": client_id}).encode("utf-8")
        req = urllib.request.Request(f"http://{server_address}/prompt", data=data)
        req.add_header("Content-Type", "application/json")
        response = urllib.request.urlopen(req).read()
        result = json.loads(response)
        debug_print("Queue prompt response:", result)
        return result
    except Exception as e:
        raise RuntimeError(f"Error in queue_prompt: {e}")

def get_images(ws, server_address, client_id, prompt_graph, timeout=120):
    """
    Sends the graph to ComfyUI and listens via websocket for the images 
    produced by the SaveImageWebsocket node (node ID "263").
    Returns a dict: node_id -> list of raw image bytes.
    """
    response = queue_prompt(prompt_graph, server_address, client_id)
    prompt_id = response.get("prompt_id")
    if not prompt_id:
        raise RuntimeError("No prompt_id in queue_prompt response.")

    debug_print("Prompt ID:", prompt_id)
    output_images = {}
    current_node = None
    start_time = time.time()

    while True:
        try:
            msg = ws.recv()
        except Exception as e:
            debug_print("Error receiving message from websocket:", e)
            break

        if isinstance(msg, str):
            # Probably a JSON status message
            try:
                m = json.loads(msg)
            except Exception as e:
                debug_print("Error decoding JSON:", e, "Message:", msg)
                continue
            debug_print("Received text message:", m)

            if m.get("type") == "executing":
                data = m.get("data", {})
                if data.get("prompt_id") == prompt_id:
                    if data.get("node") is None:
                        debug_print("Execution finished for prompt", prompt_id)
                        break
                    else:
                        current_node = data.get("node")
                        debug_print("Current executing node set to:", current_node)
            elif m.get("type") == "error":
                raise RuntimeError(f"ComfyUI returned an error: {m}")

        else:
            # Binary data from the SaveImageWebsocket node
            debug_print("Received binary message of length:", len(msg))
            if current_node == "263":
                img_data = msg[8:]  # skip 8-byte ComfyUI header
                output_images.setdefault("263", []).append(img_data)
                debug_print(
                    f"Appended binary image data (length: {len(img_data)}) from node {current_node}"
                )

        # Timeout
        if time.time() - start_time > timeout:
            raise RuntimeError("Timeout reached while waiting for output.")

    debug_print("Output images dictionary:", {k: [len(vv) for vv in v] for k,v in output_images.items()})
    return output_images

def build_combined_graph(
    # Core model / pipeline
    checkpoint,
    vae_name,
    pos_prompt,
    neg_prompt,
    image_path,
    steps,
    cfg_scale,
    sampler_name,
    scheduler,
    denoise,
    preview_method,
    vae_decode,
    seed_val,
    latent_width,
    latent_height,
    batch_size,

    # ControlNet related
    controlnet_canny,
    controlnet_softedge,
    controlnet_openpose,

    # Toggling usage
    run_canny,
    run_softedge,
    run_openpose,

    # Additional adjustable parameters
    canny_threshold1,
    canny_threshold2,
    canny_strength,
    softedge_strength,
    softedge_safe,
    softedge_resolution,
    openpose_strength,
    openpose_end_percent,
    openpose_detect_hand,
    openpose_detect_body,
    openpose_detect_face
):
    """
    Dynamically build a single graph that can apply any combination
    of Canny, SoftEdge, or OpenPose ControlNets (in that order).
    """
    base_graph = {
        "212": {
            "inputs": {"ckpt_name": checkpoint},
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": "Load Checkpoint"}
        },
        "159": {
            "inputs": {"vae_name": vae_name},
            "class_type": "VAELoader",
            "_meta": {"title": "Load VAE"}
        },
        "199": {
            "inputs": {
                "text": pos_prompt,
                "clip": ["212", 1]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Positive)"}
        },
        "56": {
            "inputs": {
                "text": neg_prompt,
                "clip": ["212", 1]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Negative)"}
        },
        "214": {
            "inputs": {"image": image_path, "upload": "image"},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Image"}
        },
        "218": {
            "inputs": {
                "width": latent_width,
                "height": latent_height,
                "batch_size": batch_size
            },
            "class_type": "EmptyLatentImage",
            "_meta": {"title": "Empty Latent Image"}
        },
        "232": {
            "inputs": {"seed": seed_val},
            "class_type": "Seed Generator",
            "_meta": {"title": "Seed Generator"}
        },
        "10": {
            "inputs": {
                "seed": ["232", 0],
                "steps": steps,
                "cfg": cfg_scale,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "preview_method": preview_method,
                "vae_decode": vae_decode,
                "positive": None,
                "negative": None,
                "model": ["212", 0],
                "latent_image": ["218", 0],
                "optional_vae": ["159", 0]
            },
            "class_type": "KSampler (Efficient)",
            "_meta": {"title": "KSampler (Efficient)"}
        },
        "263": {
            "inputs": {
                "filename_prefix": "",
                "images": ["10", 5]
            },
            "class_type": "SaveImageWebsocket",
            "_meta": {"title": "Save Image"}
        },
        "210": {
            "inputs": {
                "images": ["10", 5]
            },
            "class_type": "PreviewImage",
            "_meta": {"title": "Preview Image"}
        }
    }

    used_nets = []
    current_positive = ["199", 0]
    current_negative = ["56", 0]
    next_node_id = 300

    def make_control_apply_node(
        node_id, strength, start_percent, end_percent,
        control_net_loader_id, image_preproc_id,
        positive_in, negative_in
    ):
        return {
            "inputs": {
                "strength": strength,
                "start_percent": start_percent,
                "end_percent": end_percent,
                "positive": positive_in,
                "negative": negative_in,
                "control_net": [control_net_loader_id, 0],
                "image": [image_preproc_id, 0],
                "vae": ["159", 0]
            },
            "class_type": "ControlNetApplyAdvanced",
            "_meta": {"title": f"Apply ControlNet {control_net_loader_id}"}
        }

    # 1) Canny
    if run_canny:
        used_nets.append("canny")
        loader_id = str(next_node_id)
        base_graph[loader_id] = {
            "inputs": {"control_net_name": controlnet_canny},
            "class_type": "ControlNetLoader",
            "_meta": {"title": "Load Canny ControlNet"}
        }
        next_node_id += 1

        canny_preproc_id = str(next_node_id)
        base_graph[canny_preproc_id] = {
            "inputs": {
                "threshold1": canny_threshold1,
                "threshold2": canny_threshold2,
                "image": ["214", 0]
            },
            "class_type": "CannyEdgePreprocessor",
            "_meta": {"title": "Canny Preprocessor"}
        }
        next_node_id += 1

        apply_id = str(next_node_id)
        base_graph[apply_id] = make_control_apply_node(
            node_id=apply_id,
            strength=canny_strength,
            start_percent=0.0,
            end_percent=1.0,
            control_net_loader_id=loader_id,
            image_preproc_id=canny_preproc_id,
            positive_in=current_positive,
            negative_in=current_negative
        )
        next_node_id += 1

        current_positive = [apply_id, 0]
        current_negative = [apply_id, 1]

    # 2) SoftEdge (HED)
    if run_softedge:
        used_nets.append("softedge")
        loader_id = str(next_node_id)
        base_graph[loader_id] = {
            "inputs": {"control_net_name": controlnet_softedge},
            "class_type": "ControlNetLoader",
            "_meta": {"title": "Load SoftEdge ControlNet"}
        }
        next_node_id += 1

        hed_preproc_id = str(next_node_id)
        base_graph[hed_preproc_id] = {
            "inputs": {
                "safe": softedge_safe,
                "resolution": softedge_resolution,
                "image": ["214", 0]
            },
            "class_type": "HEDPreprocessor",
            "_meta": {"title": "HED Soft-Edge Lines"}
        }
        next_node_id += 1

        apply_id = str(next_node_id)
        base_graph[apply_id] = make_control_apply_node(
            node_id=apply_id,
            strength=softedge_strength,
            start_percent=0.0,
            end_percent=1.0,
            control_net_loader_id=loader_id,
            image_preproc_id=hed_preproc_id,
            positive_in=current_positive,
            negative_in=current_negative
        )
        next_node_id += 1

        current_positive = [apply_id, 0]
        current_negative = [apply_id, 1]

    # 3) OpenPose
    if run_openpose:
        used_nets.append("openpose")
        loader_id = str(next_node_id)
        base_graph[loader_id] = {
            "inputs": {"control_net_name": controlnet_openpose},
            "class_type": "ControlNetLoader",
            "_meta": {"title": "Load OpenPose ControlNet"}
        }
        next_node_id += 1

        # Convert booleans to "enable"/"disable"
        detect_hand_str = "enable" if openpose_detect_hand else "disable"
        detect_body_str = "enable" if openpose_detect_body else "disable"
        detect_face_str = "enable" if openpose_detect_face else "disable"

        dw_preproc_id = str(next_node_id)
        base_graph[dw_preproc_id] = {
            "inputs": {
                "detect_hand": detect_hand_str,
                "detect_body": detect_body_str,
                "detect_face": detect_face_str,
                "bbox_detector": "yolox_l.onnx",
                "pose_estimator": "dw-ll_ucoco_384.onnx",
                "scale_stick_for_xinsr_cn": "disable",
                "image": ["214", 0]
            },
            "class_type": "DWPreprocessor",
            "_meta": {"title": "DWPose Estimator"}
        }
        next_node_id += 1

        apply_id = str(next_node_id)
        base_graph[apply_id] = make_control_apply_node(
            node_id=apply_id,
            strength=openpose_strength,
            start_percent=0.0,
            end_percent=openpose_end_percent,
            control_net_loader_id=loader_id,
            image_preproc_id=dw_preproc_id,
            positive_in=current_positive,
            negative_in=current_negative
        )
        next_node_id += 1

        current_positive = [apply_id, 0]
        current_negative = [apply_id, 1]

    # Attach final prompts to the KSampler
    base_graph["10"]["inputs"]["positive"] = current_positive
    base_graph["10"]["inputs"]["negative"] = current_negative

    return base_graph, used_nets

def clear_comfyui_cache(server_address):
    """
    Calls the /free endpoint to clear ComfyUI's cache and unload models.
    This ensures fresh computations each time, but may slow down generation
    since ComfyUI must reload models.
    """
    url = f"http://{server_address}/free"
    payload = {
        "unload_models": True,
        "free_memory": True
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            debug_print("Successfully cleared ComfyUI cache.")
        else:
            debug_print(f"Failed to clear cache. Status code: {response.status_code}")
    except Exception as e:
        debug_print(f"Error calling /free endpoint: {e}")

def generate_image(
    # ComfyUI server address
    server_address="127.0.0.1:8188",

    # Model / pipeline
    checkpoint="photon_v1.safetensors",
    vae_name="SD1.5/vae-ft-mse-840000-ema-pruned.safetensors",

    # Prompting
    pos_prompt="A cat/dragon hybrid.",
    neg_prompt="worst quality, low quality, blurry",

    # Generation params
    steps=30,
    cfg_scale=5.0,
    sampler_name="euler_ancestral",
    scheduler="normal",
    denoise=1.0,
    preview_method="auto",
    vae_decode="true",
    seed_val=166281641902884,
    latent_width=1536,
    latent_height=768,
    batch_size=1,

    # ControlNets
    controlnet_canny="1.5/control_v11p_sd15_canny_fp16.safetensors",
    controlnet_softedge="1.5/control_v11p_sd15_softedge_fp16.safetensors",
    controlnet_openpose="1.5/control_v11p_sd15_openpose_fp16.safetensors",

    # Toggle whether to apply each ControlNet
    run_canny=False,
    run_softedge=False,
    run_openpose=True,

    # Adjustable parameters for each ControlNet
    canny_threshold1=100,
    canny_threshold2=200,
    canny_strength=0.9,
    softedge_strength=0.69,
    softedge_safe="enable",
    softedge_resolution=1536,
    openpose_strength=0.9,
    openpose_end_percent=0.95,
    openpose_detect_hand=True,
    openpose_detect_body=True,
    openpose_detect_face=True,

    # Input image bytes
    input_image_bytes=None,

    # Where to save the output & logs
    output_dir="output",

    # Whether to clear ComfyUI's cache before generation
    clear_cache=False
):
    """
    Generate an image using ComfyUI with optional ControlNet modules,
    then save the result to disk and append to a CSV log. If clear_cache=True,
    calls the /free endpoint to unload models and free memory, ensuring
    no cached data is used.

    Returns the resulting image as raw bytes.
    """
    if input_image_bytes is None:
        raise ValueError("No input_image_bytes provided.")

    # Optionally clear cache before generation
    if clear_cache:
        clear_comfyui_cache(server_address)

    # Ensure output directory exists
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    client_id = str(uuid.uuid4())

    # Write the user-provided image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(input_image_bytes)
        tmp.flush()
        image_path = tmp.name

    # Build the graph
    graph, used_nets = build_combined_graph(
        checkpoint=checkpoint,
        vae_name=vae_name,
        pos_prompt=pos_prompt,
        neg_prompt=neg_prompt,
        image_path=image_path,
        steps=steps,
        cfg_scale=cfg_scale,
        sampler_name=sampler_name,
        scheduler=scheduler,
        denoise=denoise,
        preview_method=preview_method,
        vae_decode=vae_decode,
        seed_val=seed_val,
        latent_width=latent_width,
        latent_height=latent_height,
        batch_size=batch_size,
        controlnet_canny=controlnet_canny,
        controlnet_softedge=controlnet_softedge,
        controlnet_openpose=controlnet_openpose,
        run_canny=run_canny,
        run_softedge=run_softedge,
        run_openpose=run_openpose,
        canny_threshold1=canny_threshold1,
        canny_threshold2=canny_threshold2,
        canny_strength=canny_strength,
        softedge_strength=softedge_strength,
        softedge_safe=softedge_safe,
        softedge_resolution=softedge_resolution,
        openpose_strength=openpose_strength,
        openpose_end_percent=openpose_end_percent,
        openpose_detect_hand=openpose_detect_hand,
        openpose_detect_body=openpose_detect_body,
        openpose_detect_face=openpose_detect_face
    )

    # Add a random ID to force fresh generation (break ComfyUI's internal node-cache)
    graph["10"]["_meta"]["force_unique_run_id"] = str(uuid.uuid4())

    # Construct a sanitized prompt snippet + timestamp + steps
    timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
    prompt_snippet = sanitize_prompt(pos_prompt, max_len=50)
    final_prefix = f"{timestamp_str}_{prompt_snippet}_st{steps}"

    # For the SaveImageWebsocket node's "filename_prefix"
    graph["263"]["inputs"]["filename_prefix"] = final_prefix

    # Open the websocket
    ws = websocket.WebSocket()
    ws.connect(f"ws://{server_address}/ws?clientId={client_id}")
    debug_print("WebSocket connected:", client_id)

    # Execute + get images
    images = get_images(ws, server_address, client_id, graph, timeout=120)
    ws.close()

    # Clean up temp file
    try:
        os.remove(image_path)
    except OSError:
        pass

    # Extract final image bytes
    if "263" in images and images["263"]:
        final_img_bytes = images["263"][0]
    else:
        raise RuntimeError("No output image found (node 263).")

    # ----------------------------------------------------------------
    # Save final image + Append run info to CSV
    # ----------------------------------------------------------------
    # Construct output file name
    image_filename = f"{final_prefix}.png"
    out_path = os.path.join(output_dir, image_filename)

    # Save the PNG file
    with open(out_path, "wb") as f:
        f.write(final_img_bytes)

    # Append run info to CSV log
    log_path = os.path.join(output_dir, "run_log.csv")
    file_exists = os.path.isfile(log_path)

    with open(log_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Write header only if file didn't exist
        if not file_exists:
            writer.writerow([
                "timestamp",
                "pos_prompt",
                "neg_prompt",
                "steps",
                "cfg_scale",
                "seed_val",
                "run_canny",
                "canny_threshold1",
                "canny_threshold2",
                "canny_strength",
                "run_softedge",
                "softedge_strength",
                "softedge_safe",
                "softedge_resolution",
                "run_openpose",
                "openpose_strength",
                "openpose_end_percent",
                "openpose_detect_hand",
                "openpose_detect_body",
                "openpose_detect_face",
                "clear_cache",
                "output_image"
            ])

        # Append a single row with the relevant data
        writer.writerow([
            timestamp_str,
            pos_prompt,
            neg_prompt,
            steps,
            cfg_scale,
            seed_val,
            run_canny,
            canny_threshold1,
            canny_threshold2,
            canny_strength,
            run_softedge,
            softedge_strength,
            softedge_safe,
            softedge_resolution,
            run_openpose,
            openpose_strength,
            openpose_end_percent,
            openpose_detect_hand,
            openpose_detect_body,
            openpose_detect_face,
            clear_cache,
            image_filename
        ])

    debug_print(f"Saved image to: {out_path}")
    debug_print(f"Appended run info to CSV log: {log_path}")

    return final_img_bytes
