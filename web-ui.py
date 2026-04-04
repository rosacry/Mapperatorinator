import traceback
from dataclasses import asdict
from pathlib import Path

from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf

import excepthook  # noqa
import base64
import functools
import os
import platform
import socket
import subprocess
import sys
import threading
import uuid
from typing import Callable, Any, Tuple, Dict

import io
import multiprocessing as mp
import queue as queue_mod
import datetime
import time

import webview
import werkzeug.serving
from flask import Flask, render_template, request, Response, jsonify, send_file

import routed_pickle
from config import InferenceConfig
from osuT5.osuT5.event import ContextType
from osuT5.osuT5.inference.server import InferenceClient
from osuT5.osuT5.utils import load_model_loaders
from inference import compile_args, get_server_address, main

# Queue system imports
try:
    from mapper_scrape import lookup_username
    from audio_fingerprint import identify_song
    from filename_utils import rename_output, compose_diff_name
    QUEUE_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Queue features not available: {e}")
    QUEUE_FEATURES_AVAILABLE = False
    def lookup_username(mapper_id): return None
    def identify_song(path): return (None, None)
    def rename_output(*args): return args[0]
    def compose_diff_name(*args): return "Mapperatorinator"

script_dir = os.path.dirname(os.path.abspath(__file__))
template_folder = os.path.join(script_dir, 'template')
static_folder = os.path.join(script_dir, 'static')

if not os.path.isdir(static_folder):
    print(f"Warning: Static folder not found at {static_folder}. Ensure it exists and contains your CSS/images.")


# Set Flask environment to production before initializing Flask app to silence warning
# os.environ['FLASK_ENV'] = 'production' # Removed, using cli patch instead

# --- Werkzeug Warning Suppressor Patch ---
def _ansi_style_supressor(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Any:
        # Check if the first argument is the specific warning string
        if args:
            first_arg = args[0]
            if isinstance(first_arg, str) and first_arg.startswith('WARNING: This is a development server.'):
                return ''  # Return empty string to suppress
        # Otherwise, call the original function
        return func(*args, **kwargs)

    return wrapper


# Apply the patch before Flask initialization
# noinspection PyProtectedMember
werkzeug.serving._ansi_style = _ansi_style_supressor(werkzeug.serving._ansi_style)
# --- End Patch ---

if hasattr(webview, "FileDialog"):
    OPEN_DIALOG = webview.FileDialog.OPEN
    FOLDER_DIALOG = webview.FileDialog.FOLDER
    SAVE_DIALOG = webview.FileDialog.SAVE
else:
    OPEN_DIALOG = webview.OPEN_DIALOG
    FOLDER_DIALOG = webview.FOLDER_DIALOG
    SAVE_DIALOG = webview.SAVE_DIALOG


def parse_file_dialog_result(result):
    if not result:
        return None
    return result[0] if isinstance(result, (list, tuple)) else result

app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
app.secret_key = os.urandom(24)  # Set a secret key for Flask

# --- Queue system shared state ---
queue_cancelled: bool = False
song_detection_cache: dict = {}
beatmapset_enabled = False
beatmapset_files = []
beatmapset_audio_path = None
beatmapset_background_path = None
beatmapset_output_dir = None


# --- pywebview API Class ---
class Api:
    # No __init__ needed as we get the window dynamically
    def save_file(self, filename):
        """Opens a save file dialog and returns the selected file path."""
        # Get the window dynamically from the global list
        if not webview.windows:
            print("Error: No pywebview window found.")
            return None
        current_window = webview.windows[0]
        result = current_window.create_file_dialog(SAVE_DIALOG, save_filename=filename)
        print(f"File dialog result: {result}")  # Debugging
        return parse_file_dialog_result(result)

    def browse_file(self, file_types=None):
        """Opens a file dialog and returns the selected file path."""
        # Get the window dynamically from the global list
        if not webview.windows:
            print("Error: No pywebview window found.")
            return None

        current_window = webview.windows[0]

        # File type filter
        try:
            if file_types and isinstance(file_types, list):
                file_types = tuple(file_types)

            result = current_window.create_file_dialog(
                OPEN_DIALOG,
                file_types=file_types
            )
        except Exception:
            result = current_window.create_file_dialog(OPEN_DIALOG)

        return parse_file_dialog_result(result)

    def browse_image(self):
        """Opens a file dialog specifically for image files and returns the selected file path."""
        # Get the window dynamically from the global list
        if not webview.windows:
            print("Error: No pywebview window found.")
            return None

        current_window = webview.windows[0]

        # Image file type filter
        image_file_types = (
            'Image Files (*.jpg;*.jpeg;*.png;*.bmp;*.gif;*.webp)',
            '*.jpg;*.jpeg;*.png;*.bmp;*.gif;*.webp',
            'JPEG Files (*.jpg;*.jpeg)',
            '*.jpg;*.jpeg',
            'PNG Files (*.png)',
            '*.png',
            'All Files (*.*)',
            '*.*'
        )

        try:
            result = current_window.create_file_dialog(
                webview.OPEN_DIALOG,
                file_types=image_file_types
            )
        except Exception:
            result = current_window.create_file_dialog(OPEN_DIALOG)

        return parse_file_dialog_result(result)

    def browse_folder(self):
        """Opens a folder dialog and returns the selected folder path."""
        # Get the window dynamically from the global list
        if not webview.windows:
            print("Error: No pywebview window found.")
            return None
        current_window = webview.windows[0]
        result = current_window.create_file_dialog(FOLDER_DIALOG)
        print(f"Folder dialog result: {result}")  # Debugging
        # FOLDER_DIALOG also returns a tuple containing the path
        return parse_file_dialog_result(result)


# --- Shared State for Inference Processes ---
# Track inference workers (multiprocessing) instead of Popen
# job_id -> {"process": mp.Process, "queue": mp.Queue, "cancelled": bool}
processes = {}
cancelled_jobs = set()
process_lock = threading.Lock()


def _ensure_inference_server(args):
    model_loader, tokenizer_loader = load_model_loaders(
        ckpt_path_str=args.model_path,
        t5_args=args.train,
        device=args.device,
        precision=args.precision,
        attn_implementation=args.attn_implementation,
        eval_mode=True,
        pickle_module=routed_pickle,
        lora_path=args.lora_path,
    )
    _server_owner_client = InferenceClient(
        model_loader,
        tokenizer_loader,
        max_batch_size=args.max_batch_size,
        socket_path=get_server_address(args.model_path),
    )

    # Start the server in a dedicated thread that outlives per-job workers.
    _server_owner_client.ensure_server()


def _coerce_optional_int(v):
    if v is None or v == '':
        return None
    return int(v)


def _coerce_optional_float(v):
    if v is None or v == '':
        return None
    return float(v)


def _coerce_bool_checkbox(form, key: str) -> bool:
    return key in form


class _QueueWriter(io.TextIOBase):
    def __init__(self, q: mp.Queue):
        self._q = q
        self._buf = ""

    def write(self, s):
        if not s:
            return 0
        self._buf += s

        # tqdm progress bars often update the same line using carriage returns.
        # Forward those updates as individual messages so the UI can parse percentage.
        while "\r" in self._buf:
            seg, self._buf = self._buf.split("\r", 1)
            if seg:
                self._q.put(seg)
            else:
                # Even an empty segment can represent a progress refresh; keep UI alive.
                self._q.put("")

        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._q.put(line)
        return len(s)

    def flush(self):
        if self._buf:
            self._q.put(self._buf)
            self._buf = ""


def _inference_worker(cfg: InferenceConfig, out_q: mp.Queue):
    """Worker entrypoint executed in a separate process (spawn-safe)."""
    import sys as _sys
    import traceback as _traceback

    try:
        # Redirect stdout/stderr to queue.
        qw = _QueueWriter(out_q)
        _sys.stdout = qw
        _sys.stderr = qw

        main(cfg)
        qw.flush()
        out_q.put({"_event": "exit", "code": 0})
    except Exception as e:
        try:
            out_q.put(str(e))
            out_q.put(_traceback.format_exc())
        except Exception:
            pass
        out_q.put({"_event": "exit", "code": 1})


# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main HTML page."""
    # Jinja rendering is now handled by Flask's render_template
    return render_template('index.html')


@app.route('/check_bf16_support', methods=['GET'])
def check_bf16_support():
    """Check if the GPU supports bf16 precision for faster inference."""
    try:
        import torch

        if not torch.cuda.is_available():
            return jsonify({"supported": False, "reason": "CUDA not available"})

        # Get GPU compute capability
        device_props = torch.cuda.get_device_properties(0)
        compute_capability = (device_props.major, device_props.minor)
        gpu_name = device_props.name

        # bf16 requires compute capability 8.0+ (Ampere and newer: RTX 30xx, 40xx, A100, etc.)
        supported = compute_capability[0] >= 8

        return jsonify({
            "supported": supported,
            "gpu_name": gpu_name,
            "compute_capability": f"{compute_capability[0]}.{compute_capability[1]}",
            "reason": "GPU supports bf16" if supported else f"GPU compute capability {compute_capability[0]}.{compute_capability[1]} < 8.0 required"
        })
    except Exception as e:
        return jsonify({"supported": False, "reason": str(e)})


@app.route('/start_inference', methods=['POST'])
def start_inference():
    """Starts the inference process based on form data."""
    job_id = uuid.uuid4().hex

    # Create config
    config_name = request.form.get('model')
    with initialize_config_dir(version_base="1.1", config_dir=str(Path(__file__).parent / "configs/inference")):
        cfg = compose(config_name=config_name)
    cfg = OmegaConf.to_object(cfg)
    cfg.use_server = True

    # Required/paths
    cfg.audio_path = request.form.get('audio_path') or None
    cfg.output_path = request.form.get('output_path') or None
    cfg.beatmap_path = request.form.get('beatmap_path') or None
    cfg.lora_path = request.form.get('lora_path') or None

    # Basic settings
    cfg.gamemode = _coerce_optional_int(request.form.get('gamemode')) or 0
    cfg.difficulty = _coerce_optional_float(request.form.get('difficulty'))
    cfg.year = _coerce_optional_int(request.form.get('year'))

    # Numeric settings
    cfg.hp_drain_rate = _coerce_optional_float(request.form.get('hp_drain_rate'))
    cfg.circle_size = _coerce_optional_float(request.form.get('circle_size'))
    cfg.overall_difficulty = _coerce_optional_float(request.form.get('overall_difficulty'))
    cfg.approach_rate = _coerce_optional_float(request.form.get('approach_rate'))
    cfg.slider_multiplier = _coerce_optional_float(request.form.get('slider_multiplier'))
    cfg.slider_tick_rate = _coerce_optional_float(request.form.get('slider_tick_rate'))
    cfg.keycount = _coerce_optional_int(request.form.get('keycount'))
    cfg.hold_note_ratio = _coerce_optional_float(request.form.get('hold_note_ratio'))
    cfg.scroll_speed_ratio = _coerce_optional_float(request.form.get('scroll_speed_ratio'))
    cfg.cfg_scale = _coerce_optional_float(request.form.get('cfg_scale')) or cfg.cfg_scale
    cfg.temperature = _coerce_optional_float(request.form.get('temperature')) or cfg.temperature
    cfg.top_p = _coerce_optional_float(request.form.get('top_p')) or cfg.top_p
    cfg.seed = _coerce_optional_int(request.form.get('seed'))
    cfg.mapper_id = _coerce_optional_int(request.form.get('mapper_id'))

    # Metadata
    cfg.title = request.form.get('title') or None
    cfg.title_unicode = request.form.get('title_unicode') or None
    cfg.artist = request.form.get('artist') or None
    cfg.artist_unicode = request.form.get('artist_unicode') or None
    cfg.creator = request.form.get('creator') or None
    cfg.version = request.form.get('version') or None
    cfg.source = request.form.get('source') or None
    cfg.tags = request.form.get('tags') or None
    cfg.preview_time = _coerce_optional_int(request.form.get('preview_time'))

    # Background image
    background_image = request.form.get('background_image')
    if background_image:
        cfg.background = background_image

    # Timing and segmentation
    cfg.start_time = _coerce_optional_int(request.form.get('start_time'))
    cfg.end_time = _coerce_optional_int(request.form.get('end_time'))

    # Checkboxes
    cfg.export_osz = _coerce_bool_checkbox(request.form, 'export_osz')
    cfg.add_to_beatmap = _coerce_bool_checkbox(request.form, 'add_to_beatmap')
    cfg.overwrite_reference_beatmap = _coerce_bool_checkbox(request.form, 'overwrite_reference_beatmap')
    cfg.hitsounded = _coerce_bool_checkbox(request.form, 'hitsounded')
    cfg.super_timing = _coerce_bool_checkbox(request.form, 'super_timing')

    # Precision
    if _coerce_bool_checkbox(request.form, 'enable_bf16'):
        cfg.precision = 'bf16'

    # Descriptor lists
    descriptors = request.form.getlist('descriptors')
    cfg.descriptors = descriptors if descriptors else None
    negative_descriptors = request.form.getlist('negative_descriptors')
    cfg.negative_descriptors = negative_descriptors if negative_descriptors else None

    # In-context options
    in_context_options = request.form.getlist('in_context_options')
    if in_context_options and cfg.beatmap_path:
        try:
            cfg.in_context = [ContextType[opt] for opt in in_context_options]
        except Exception as e:
            traceback.print_exc()
            return jsonify({"status": "error", "message": f"Invalid in-context options: {e}"}), 400

    # Validate and compile args
    try:
        compile_args(cfg, verbose=False)
    except ValueError as ve:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(ve)}), 400

    # Ensure a shared server is running, owned by web UI.
    try:
        _ensure_inference_server(cfg)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Failed to ensure inference server: {e}"}), 500

    # Spawn the worker process.
    try:
        q = mp.Queue()
        p = mp.Process(target=_inference_worker, args=(cfg, q), daemon=True)
        p.start()

        with process_lock:
            processes[job_id] = {"process": p, "queue": q}

        return jsonify({"status": "success", "message": "Inference started", "job_id": job_id}), 202
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Failed to start process: {e}"}), 500


@app.route('/stream_output')
def stream_output():
    """Streams the output of the running inference process using SSE."""

    job_id = request.args.get('job_id', '').strip()
    if not job_id:
        return Response("event: end\ndata: Missing job_id\n\n", mimetype='text/event-stream')

    def generate():
        with process_lock:
            rec = processes.get(job_id)
            if not rec:
                yield "event: end\ndata: No active process or process already finished\n\n"
                return
            proc = rec["process"]
            q = rec["queue"]

        full_output_lines = []
        error_occurred = False
        exit_code = None

        try:
            while True:
                try:
                    item = q.get(timeout=0.2)
                except queue_mod.Empty:
                    if not proc.is_alive():
                        # Process died without sending sentinel.
                        exit_code = proc.exitcode
                        break
                    continue

                if isinstance(item, dict) and item.get("_event") == "exit":
                    exit_code = item.get("code", 0)
                    break

                line = str(item)
                full_output_lines.append(line + "\n")
                yield f"data: {line.rstrip()}\n\n"
                sys.stdout.flush()

            # Determine error state.
            if exit_code and exit_code != 0:
                with process_lock:
                    was_cancelled = job_id in cancelled_jobs
                    cancelled_jobs.discard(job_id)
                if was_cancelled:
                    error_occurred = False
                else:
                    error_occurred = True
        except Exception as e:
            error_occurred = True
            full_output_lines.append(f"\n--- STREAMING ERROR ---\n{e}\n")
        finally:
            # Save logs on error (same behavior as before).
            if error_occurred:
                try:
                    log_dir = os.path.join(script_dir, 'logs')
                    os.makedirs(log_dir, exist_ok=True)
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    pid = proc.pid if proc is not None else 0
                    log_filename = f"error_{pid}_{timestamp}.log"
                    log_filepath = os.path.join(log_dir, log_filename)
                    error_content = "".join(full_output_lines)

                    with open(log_filepath, 'w', encoding='utf-8') as f:
                        f.write(error_content)
                    yield f"event: error_log\ndata: {log_filepath.replace(os.sep, '/')}\n\n"
                except Exception:
                    pass

            completion_message = "Process completed"
            if error_occurred:
                completion_message += " with errors"
            yield f"event: end\ndata: {completion_message}\n\n"

            # Cleanup.
            with process_lock:
                processes.pop(job_id, None)
                cancelled_jobs.discard(job_id)

    return Response(generate(), mimetype='text/event-stream')


@app.route('/cancel_inference', methods=['POST'])
def cancel_inference():
    """Attempts to terminate the currently running inference process."""
    global queue_cancelled

    # Support both form data and JSON
    if request.is_json:
        data = request.json or {}
        job_id = data.get('job_id', '').strip() if data.get('job_id') else ''
        clear_queue = data.get('clear_queue', False)
    else:
        job_id = request.form.get('job_id', '').strip()
        clear_queue = False

    if clear_queue:
        queue_cancelled = True

    # If no specific job_id, cancel all running jobs
    if not job_id:
        with process_lock:
            if not processes:
                msg = "Queue cancelled." if clear_queue else "No active processes"
                return jsonify({"status": "success" if clear_queue else "error", "message": msg}), 200 if clear_queue else 404
            for jid, rec in list(processes.items()):
                proc = rec["process"]
                if proc.is_alive():
                    cancelled_jobs.add(jid)
                    try:
                        if sys.platform == 'win32':
                            subprocess.run(['taskkill', '/F', '/T', '/PID', str(proc.pid)], capture_output=True, timeout=5)
                        else:
                            proc.terminate()
                    except Exception:
                        pass
        return jsonify({"status": "success", "message": "Cancel request sent"}), 200

    with process_lock:
        rec = processes.get(job_id)
        if not rec:
            return jsonify({"status": "error", "message": "No active process found"}), 404
        proc = rec["process"]

        if proc.is_alive():
            cancelled_jobs.add(job_id)
            try:
                if sys.platform == 'win32':
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(proc.pid)], capture_output=True, timeout=5)
                else:
                    proc.terminate()
                return jsonify({"status": "success", "message": "Cancel request sent"}), 200
            except Exception as e:
                return jsonify({"status": "error", "message": f"Failed to terminate process: {e}"}), 500

    return jsonify({"status": "success", "message": "Process already finished"}), 200


@app.route('/open_folder', methods=['GET'])
def open_folder():
    """Opens a folder in the file explorer."""
    folder_path = request.args.get('folder')
    print(f"Request received to open folder: {folder_path}")
    if not folder_path:
        return jsonify({"status": "error", "message": "No folder path specified"}), 400

    # Resolve to absolute path for checks
    abs_folder_path = os.path.abspath(folder_path)

    # Security check: Basic check if it's within the project directory.
    # Adjust this check based on your security needs and where output is expected.
    workspace_root = os.path.abspath(script_dir)
    # Example: Only allow opening if it's inside the workspace root
    # if not abs_folder_path.startswith(workspace_root):
    #     print(f"Security Warning: Attempt to open potentially restricted folder: {abs_folder_path}")
    #     return jsonify({"status": "error", "message": "Access denied to specified folder path."}), 403

    if not os.path.isdir(abs_folder_path):
        print(f"Invalid folder path provided or folder does not exist: {abs_folder_path}")
        return jsonify({"status": "error", "message": "Invalid or non-existent folder path specified"}), 400

    try:
        system = platform.system()
        if system == 'Windows':
            os.startfile(os.path.normpath(abs_folder_path))
        elif system == 'Darwin':
            subprocess.Popen(['open', abs_folder_path])
        else:
            subprocess.Popen(['xdg-open', abs_folder_path])
        print(f"Successfully requested to open folder: {abs_folder_path}")
        return jsonify({"status": "success", "message": "Folder open request sent."}), 200
    except Exception as e:
        print(f"Error opening folder '{abs_folder_path}': {e}")
        return jsonify({"status": "error", "message": f"Could not open folder: {e}"}), 500


@app.route('/open_log_file', methods=['GET'])
def open_log_file():
    """Opens a specific log file."""
    log_path = request.args.get('path')
    print(f"Request received to open log file: {log_path}")
    if not log_path:
        return jsonify({"status": "error", "message": "No log file path specified"}), 400

    # Security Check: Ensure the file is within the 'logs' directory
    log_dir = os.path.abspath(os.path.join(script_dir, 'logs'))
    # Normalize the input path and resolve symlinks etc.
    abs_log_path = os.path.abspath(os.path.normpath(log_path))

    # IMPORTANT SECURITY CHECK:
    if not abs_log_path.startswith(log_dir + os.sep):
        print(f"Security Alert: Attempt to open file outside of logs directory: {abs_log_path} (Log dir: {log_dir})")
        return jsonify({"status": "error", "message": "Access denied: File is outside the designated logs directory."}), 403

    if not os.path.isfile(abs_log_path):
        print(f"Log file not found at: {abs_log_path}")
        return jsonify({"status": "error", "message": "Log file not found."}), 404

    try:
        system = platform.system()
        if system == 'Windows':
            os.startfile(abs_log_path) # normpath already applied
        elif system == 'Darwin':
            subprocess.Popen(['open', abs_log_path])
        else:
            subprocess.Popen(['xdg-open', abs_log_path])
        print(f"Successfully requested to open log file: {abs_log_path}")
        return jsonify({"status": "success", "message": "Log file open request sent."}), 200
    except Exception as e:
        print(f"Error opening log file '{abs_log_path}': {e}")
        return jsonify({"status": "error", "message": f"Could not open log file: {e}"}), 500


@app.route('/save_config', methods=['POST'])
def save_config():
    try:
        file_path = request.form.get('file_path')
        config_data = request.form.get('config_data')

        if not file_path or not config_data:
            return jsonify({'success': False, 'error': 'Missing required parameters'})

        # Write the configuration file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(config_data)

        return jsonify({
            'success': True,
            'file_path': file_path,
            'message': 'Configuration saved successfully'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to save configuration: {str(e)}'
        })


@app.route('/validate_paths', methods=['POST'])
def validate_paths():
    """Validates and autofills missing paths."""
    try:
        # Get paths
        audio_path = request.form.get('audio_path', '').strip()
        beatmap_path = request.form.get('beatmap_path', '').strip()
        output_path = request.form.get('output_path', '').strip()

        inference_args = InferenceConfig()
        inference_args.audio_path = audio_path
        inference_args.beatmap_path = beatmap_path
        inference_args.output_path = output_path

        try:
            compile_args(inference_args, verbose=False)
        except ValueError as v:
            return jsonify({
                'success': False,
                'autofilled_args': None,
                'errors': [str(v)]
            }), 200

        autofilled_args = asdict(inference_args)
        del autofilled_args['in_context']
        del autofilled_args['output_type']
        del autofilled_args['train']

        # Attempt song detection if requested
        detected_artist = None
        detected_title = None
        detect_song = request.form.get('detect_song', 'false').lower() == 'true'

        if detect_song and QUEUE_FEATURES_AVAILABLE and inference_args.audio_path:
            actual_audio = inference_args.audio_path
            if os.path.isfile(actual_audio):
                if actual_audio in song_detection_cache:
                    detected_artist, detected_title = song_detection_cache[actual_audio]
                else:
                    try:
                        detected_artist, detected_title = identify_song(actual_audio)
                        song_detection_cache[actual_audio] = (detected_artist, detected_title)
                    except Exception as e:
                        print(f"Song detection error: {e}")
                        song_detection_cache[actual_audio] = (None, None)

        # Return the results
        response_data = {
            'success': True,
            'autofilled_args': autofilled_args,
            'errors': []
        }
        if detected_artist:
            response_data['detected_artist'] = detected_artist
        if detected_title:
            response_data['detected_title'] = detected_title

        return jsonify(response_data), 200

    except Exception as e:
        error_msg = f"Error during path validation: {str(e)}"
        print(error_msg)
        return jsonify({
            'success': False,
            'autofilled_args': None,
            'errors': [error_msg]
        }), 500


# ═══════════════════════════════════════════════════════════════════════
#  QUEUE SYSTEM ROUTES
# ═══════════════════════════════════════════════════════════════════════

@app.route("/lookup_mapper_name", methods=["POST"])
def api_lookup_mapper():
    """Look up mapper username by scraping public osu! profile page."""
    data = request.get_json() or {}
    mapper_id = data.get("mapper_id")
    if not mapper_id:
        return jsonify({"error": "mapper_id required"}), 400
    if not QUEUE_FEATURES_AVAILABLE:
        return jsonify({"error": "Queue features not available"}), 503
    try:
        name = lookup_username(mapper_id)
        if not name:
            return jsonify({"error": "User not found"}), 404
        return jsonify({"username": name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/queue_status', methods=['GET', 'POST'])
def queue_status():
    """Get or set the queue cancellation status."""
    global queue_cancelled
    if request.method == 'POST':
        data = request.json or {}
        if 'cancelled' in data:
            queue_cancelled = data['cancelled']
            return jsonify({"status": "success", "cancelled": queue_cancelled})
        return jsonify({"status": "error", "message": "Missing 'cancelled' field"}), 400
    return jsonify({"cancelled": queue_cancelled})


@app.route('/reset_queue', methods=['POST'])
def reset_queue():
    """Reset the queue cancellation flag."""
    global queue_cancelled
    queue_cancelled = False
    return jsonify({"status": "success", "message": "Queue reset"})


@app.route('/get_audio_info', methods=['POST'])
def get_audio_info():
    """Get audio file info and return a URL for playback."""
    data = request.get_json() or {}
    audio_path = data.get('path', '')
    if not audio_path or not os.path.exists(audio_path):
        return jsonify({"success": False, "message": "Audio file not found"})
    try:
        return jsonify({"success": True, "url": f"/serve_audio?path={audio_path}", "duration": None})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route('/serve_audio', methods=['GET'])
def serve_audio():
    """Serve an audio file for preview playback with Range request support."""
    audio_path = request.args.get('path', '')
    if not audio_path or not os.path.exists(audio_path):
        return "Audio file not found", 404

    ext = os.path.splitext(audio_path)[1].lower()
    content_types = {'.mp3': 'audio/mpeg', '.ogg': 'audio/ogg', '.wav': 'audio/wav', '.flac': 'audio/flac', '.m4a': 'audio/mp4'}
    content_type = content_types.get(ext, 'audio/mpeg')
    file_size = os.path.getsize(audio_path)
    range_header = request.headers.get('Range', None)

    if range_header:
        byte_start, byte_end = 0, file_size - 1
        range_match = range_header.replace('bytes=', '').split('-')
        if range_match[0]:
            byte_start = int(range_match[0])
        if len(range_match) > 1 and range_match[1]:
            byte_end = int(range_match[1])
        byte_end = min(byte_end, file_size - 1)
        content_length = byte_end - byte_start + 1

        def generate_range():
            with open(audio_path, 'rb') as f:
                f.seek(byte_start)
                remaining = content_length
                while remaining > 0:
                    chunk = f.read(min(8192, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        response = Response(generate_range(), status=206, mimetype=content_type, direct_passthrough=True)
        response.headers['Content-Range'] = f'bytes {byte_start}-{byte_end}/{file_size}'
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Content-Length'] = content_length
        return response
    else:
        def generate():
            with open(audio_path, 'rb') as f:
                while chunk := f.read(8192):
                    yield chunk
        response = Response(generate(), mimetype=content_type)
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Content-Length'] = file_size
        return response


@app.route('/get_image_preview', methods=['POST'])
def get_image_preview():
    """Get base64 encoded image preview for background."""
    data = request.get_json() or {}
    image_path = data.get('path', '')
    if not image_path or not os.path.exists(image_path):
        return jsonify({"success": False, "message": "Image file not found"})
    try:
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            return jsonify({"success": False, "message": "Unsupported image format"})
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        type_map = {'.jpg': 'jpeg', '.jpeg': 'jpeg', '.png': 'png', '.gif': 'gif', '.bmp': 'bmp'}
        return jsonify({"success": True, "data": image_data, "type": type_map.get(ext, 'jpeg')})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route('/init_beatmapset', methods=['POST'])
def init_beatmapset():
    """Initialize beatmap set compilation for a queue run."""
    global beatmapset_enabled, beatmapset_files, beatmapset_audio_path, beatmapset_background_path, beatmapset_output_dir
    data = request.get_json() or {}
    beatmapset_enabled = data.get('enabled', False)
    beatmapset_files = []
    beatmapset_audio_path = None
    beatmapset_background_path = None
    beatmapset_output_dir = None
    return jsonify({"status": "success", "enabled": beatmapset_enabled})


@app.route('/finalize_beatmapset', methods=['POST'])
def finalize_beatmapset():
    """Compile all collected .osu files into a single .osz beatmap set."""
    global beatmapset_enabled, beatmapset_files, beatmapset_audio_path, beatmapset_background_path, beatmapset_output_dir
    import zipfile

    if not beatmapset_enabled or len(beatmapset_files) == 0:
        return jsonify({"success": False, "message": "No files to compile"})
    try:
        output_dir = beatmapset_output_dir or os.path.dirname(beatmapset_files[0])
        first_osu = beatmapset_files[0]
        artist, title = "Unknown", "Unknown"
        try:
            with open(first_osu, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('Artist:'):
                        artist = line.split(':', 1)[1].strip()
                    elif line.startswith('Title:'):
                        title = line.split(':', 1)[1].strip()
                    if artist != "Unknown" and title != "Unknown":
                        break
        except Exception:
            pass
        safe_artist = "".join(c for c in artist if c.isalnum() or c in " -_").strip()
        safe_title = "".join(c for c in title if c.isalnum() or c in " -_").strip()
        osz_path = os.path.join(output_dir, f"{safe_artist} - {safe_title}.osz")
        with zipfile.ZipFile(osz_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for osu_file in beatmapset_files:
                zf.write(osu_file, os.path.basename(osu_file))
            if beatmapset_audio_path and os.path.exists(beatmapset_audio_path):
                zf.write(beatmapset_audio_path, os.path.basename(beatmapset_audio_path))
            if beatmapset_background_path and os.path.exists(beatmapset_background_path):
                zf.write(beatmapset_background_path, os.path.basename(beatmapset_background_path))
        beatmapset_enabled = False
        beatmapset_files = []
        beatmapset_audio_path = None
        beatmapset_background_path = None
        beatmapset_output_dir = None
        return jsonify({"success": True, "filename": os.path.basename(osz_path), "path": osz_path})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


# --- Function to Run Flask in a Thread ---
def run_flask(port):
    """Runs the Flask app."""

    # Use threaded=True for better concurrency within Flask
    # Avoid debug=True as it interferes with threading and pywebview
    print(f"Starting Flask server on http://127.0.0.1:{port}")
    try:
        # Explicitly set debug=False, in addition to FLASK_ENV=production
        app.run(host='127.0.0.1', port=port, threaded=True, debug=False)
    except OSError as e:
        print(f"Flask server could not start on port {port}: {e}")
        # Optionally: try another port or exit


# --- Function to Find Available Port ---
def find_available_port(start_port=5000, max_tries=100):
    """Finds an available TCP port."""
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                print(f"Found available port: {port}")
                return port
            except OSError:
                continue  # Port already in use
    raise IOError("Could not find an available port.")


# --- Main Execution ---
if __name__ == '__main__':
    # Find an available port for Flask
    flask_port = find_available_port()

    # Start Flask server in a daemon thread
    flask_thread = threading.Thread(target=run_flask, args=(flask_port,), daemon=True)
    flask_thread.start()

    # Give Flask a moment to start up
    time.sleep(1)

    # --- Calculate Responsive Window Size ---
    try:
        primary_screen = webview.screens[0]
        screen_width = primary_screen.width
        screen_height = primary_screen.height
        # Calculate window size (e.g., 45% width, 95% height of primary screen)
        window_width = int(screen_width * 0.45)
        window_height = int(screen_height * 0.9)
        print(f"Screen: {screen_width}x{screen_height}, Window: {window_width}x{window_height}")
    except Exception as e:
        print(f"Could not get screen dimensions, using default: {e}")
        # Fallback to default size if screen info is unavailable
        window_width = 900
        window_height = 1000
    # --- End Calculate Responsive Window Size ---

    # Create the pywebview window pointing to the Flask server
    window_title = 'Mapperatorinator'
    flask_url = f'http://127.0.0.1:{flask_port}/'

    print(f"Creating pywebview window loading URL: {flask_url}")

    # Instantiate the API class (doesn't need window object anymore)
    api = Api()

    # Pass api instance directly to create_window via js_api
    window = webview.create_window(
        window_title,
        url=flask_url,
        width=window_width,  # Use calculated width
        height=window_height,  # Use calculated height
        resizable=True,
        js_api=api  # Expose Python API class here
    )

    # Start the pywebview event loop (no args needed here now)
    webview.start()

    print("Pywebview window closed. Exiting application.")
    # Flask thread will exit automatically as it's a daemon
