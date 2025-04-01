import webview
import subprocess
import threading
import platform
import os
import json
import sys # For flushing output in SSE
import time # For SSE keep-alive
import socket # To find available port

# --- Flask Imports ---
from flask import Flask, render_template, request, Response, url_for, jsonify

# --- Flask App Setup ---

# Use absolute paths for templates and static files relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
template_folder = os.path.join(script_dir, 'template')
static_folder = template_folder

# Ensure 'static' folder exists if you moved style.css back there
# If style.css is in 'template', change static_folder=template_folder
# For this example, assuming style.css is back in 'static/'
if not os.path.isdir(static_folder):
     print(f"Warning: Static folder not found at {static_folder}. CSS might not load.")
     # If style.css is in template folder use:
     # static_folder = template_folder

app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
app.secret_key = os.urandom(24) # Set a secret key for Flask

# --- pywebview API Class ---
class Api:
    # No __init__ needed as we get the window dynamically

    def browse_file(self):
        """Opens a file dialog and returns the selected file path."""
        # Get the window dynamically from the global list
        if not webview.windows:
            print("Error: No pywebview window found.")
            return None
        current_window = webview.windows[0]
        result = current_window.create_file_dialog(webview.OPEN_DIALOG)
        print(f"File dialog result: {result}") # Debugging
        # pywebview returns a tuple, even for single file selection
        return result[0] if result else None

    def browse_folder(self):
        """Opens a folder dialog and returns the selected folder path."""
        # Get the window dynamically from the global list
        if not webview.windows:
            print("Error: No pywebview window found.")
            return None
        current_window = webview.windows[0]
        result = current_window.create_file_dialog(webview.FOLDER_DIALOG)
        print(f"Folder dialog result: {result}") # Debugging
        # FOLDER_DIALOG also returns a tuple containing the path
        return result[0] if result else None

# --- Shared State for Inference Process ---
current_process = None
process_lock = threading.Lock() # Lock for accessing current_process safely

# --- Helper Function (same as original Flask) ---
def dq_quote(s):
    """Wrap the string in double quotes and escape inner double quotes."""
    # Basic check if it looks quoted
    if isinstance(s, str) and s.startswith('"') and s.endswith('"'):
        return s
    return '"' + str(s).replace('"', '\\"') + '"'

# NEW helper function for double-single quotes
def dsq_quote(s):
    """
    Prepares a path string for Hydra command-line override.
    Wraps the path in single quotes, escaping internal single quotes (' -> \\').
    Then wraps the result in double quotes for shell safety.
    Example: "C:/My's Folder" becomes "\"'C:/My\\'s Folder'\""
    """
    path_str = str(s)

    # 1. Escape internal single quotes within the path string itself
    escaped_path = path_str.replace("'", "\\'") # Replace ' with \'

    # 2. Wrap the escaped path string in single quotes
    inner_quoted = "'" + escaped_path + "'"

    # 3. Wrap the single-quoted string in double quotes for the shell command line
    return '"' + inner_quoted + '"'

def format_list_arg(items):
    """Formats a list of strings for the command line argument."""
    return "[" + ",".join("'" + str(d) + "'" for d in items) + "]"


# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main HTML page."""
    # Jinja rendering is now handled by Flask's render_template
    return render_template('index.html')

@app.route('/start_inference', methods=['POST'])
def start_inference():
    """Starts the inference process based on form data."""
    global current_process
    with process_lock:
        if current_process and current_process.poll() is None:
            return jsonify({"status": "error", "message": "Process already running"}), 409 # Conflict

        # --- Construct Command --- (Adapted from previous pywebview version)
        cmd = ["python", "inference.py"] # Ensure inference.py is accessible

        # Required Paths
        if request.form.get('audio_path'):
            cmd.append("audio_path=" + dsq_quote(request.form.get('audio_path')))
        if request.form.get('output_path'):
            cmd.append("output_path=" + dsq_quote(request.form.get('output_path')))
        # Optional beatmap path
        if request.form.get('beatmap_path'):
            cmd.append("beatmap_path=" + dsq_quote(request.form.get('beatmap_path')))

        # Basic settings
        if request.form.get('gamemode'):
             cmd.append("gamemode=" + dq_quote(request.form.get('gamemode')))
        if request.form.get('difficulty'):
            cmd.append("difficulty=" + dq_quote(request.form.get('difficulty')))
        if request.form.get('year'):
            cmd.append("year=" + dq_quote(request.form.get('year')))

        # Optional numeric settings
        for param in ['slider_multiplier', 'circle_size', 'keycount', 'hold_note_ratio', 'scroll_speed_ratio', 'cfg_scale', 'seed']:
            if request.form.get(param):
                 cmd.append(f"{param}=" + dq_quote(request.form.get(param)))
        # Optional mapper_id
        if request.form.get('mapper_id'):
            cmd.append("mapper_id=" + dq_quote(request.form.get('mapper_id')))

        # Timing and segmentation
        for param in ['start_time', 'end_time']:
            if request.form.get(param):
                cmd.append(f"{param}=" + dq_quote(request.form.get(param)))

        # Checkboxes (Flask sends 'on' or nothing, check existence)
        if 'hitsounded' in request.form:
            cmd.append("hitsounded=true")
        if 'add_to_beatmap' in request.form:
            cmd.append("add_to_beatmap=true")
        if 'super_timing' in request.form:
            cmd.append("super_timing=true")

        # Descriptors (getlist handles multiple values for the same name)
        descriptors = request.form.getlist('descriptors')
        if descriptors:
            desc_str = format_list_arg(descriptors)
            cmd.append("descriptors=" + dq_quote(desc_str))

        command_str = " ".join(cmd)
        print("Executing Command via Flask:", command_str)

        try:
            # Start the inference process
            current_process = subprocess.Popen(
                command_str,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, # Combine stdout and stderr
                bufsize=1,
                universal_newlines=True,
                encoding='utf-8',
                errors='replace'
            )
            print(f"Started process with PID: {current_process.pid}")
            # Return success to the AJAX call
            return jsonify({"status": "success", "message": "Inference started"}), 202 # Accepted

        except Exception as e:
            print(f"Error starting subprocess: {e}")
            current_process = None # Ensure it's reset on error
            return jsonify({"status": "error", "message": f"Failed to start process: {e}"}), 500

@app.route('/stream_output')
def stream_output():
    """Streams the output of the running inference process using SSE."""
    def generate():
        global current_process
        process_to_stream = None # Local reference

        # Short lock to safely get the process object
        with process_lock:
             if current_process and current_process.poll() is None:
                 process_to_stream = current_process
             else:
                 # Handle case where process is already finished or never started
                 print("Stream requested but no active process found.")
                 yield "event: end\ndata: No active process or process already finished\n\n"
                 return # Exit generator

        if process_to_stream:
            print(f"Streaming output for PID: {process_to_stream.pid}")
            try:
                # Stream lines from stdout
                for line in iter(process_to_stream.stdout.readline, ""):
                    yield f"data: {line.rstrip()}\n\n"
                    sys.stdout.flush() # Ensure data is sent

                # Check process completion status after stream ends
                process_to_stream.stdout.close() # Close the pipe
                return_code = process_to_stream.wait() # Wait for process to terminate fully
                print(f"Process {process_to_stream.pid} finished streaming with code: {return_code}")

            except Exception as e:
                 print(f"Error during streaming for PID {process_to_stream.pid}: {e}")
                 yield f"event: error\ndata: Streaming error: {e}\n\n" # Send error event to client
            finally:
                # Send custom 'end' event
                yield "event: end\ndata: Process completed or stream terminated\n\n"
                print(f"Finished streaming for PID: {process_to_stream.pid}")
                # Clear the global process only if it's the one we streamed
                with process_lock:
                    if current_process == process_to_stream:
                         current_process = None
                         print("Cleared global current_process reference.")

    # Return the generator function wrapped in a Flask Response
    return Response(generate(), mimetype='text/event-stream')


@app.route('/open_folder', methods=['GET'])
def open_folder():
    """Opens a folder in the file explorer."""
    folder_path = request.args.get('folder')
    print(f"Request received to open folder: {folder_path}") # Debug print
    if not folder_path or not os.path.isdir(folder_path):
        print(f"Invalid folder path provided: {folder_path}")
        return "Error: Invalid or non-existent folder path specified", 400

    try:
        system = platform.system()
        if system == 'Windows':
            os.startfile(os.path.normpath(folder_path)) # More reliable on Windows
        elif system == 'Darwin':  # macOS
            subprocess.Popen(['open', folder_path])
        else:  # Linux and others
            subprocess.Popen(['xdg-open', folder_path])
        print(f"Successfully requested to open folder: {folder_path}")
        return "Folder open request sent.", 200
    except Exception as e:
        print(f"Error opening folder '{folder_path}': {e}")
        return f"Error: Could not open folder - {e}", 500


# --- Function to Run Flask in a Thread ---
def run_flask(port):
    """Runs the Flask app."""
    # Use threaded=True for better concurrency within Flask
    # Avoid debug=True as it interferes with threading and pywebview
    print(f"Starting Flask server on http://127.0.0.1:{port}")
    try:
        app.run(host='127.0.0.1', port=port, threaded=True)
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
                continue # Port already in use
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

    # Create the pywebview window pointing to the Flask server
    window_title = 'Mapperatorinator Interface (Flask+pywebview)'
    flask_url = f'http://127.0.0.1:{flask_port}/'

    print(f"Creating pywebview window loading URL: {flask_url}")

    # Instantiate the API class (doesn't need window object anymore)
    api = Api()

    # Pass api instance directly to create_window via js_api
    window = webview.create_window(
        window_title,
        url=flask_url,
        width=900,
        height=750,
        resizable=True,
        js_api=api # Expose Python API class here
    )

    # --- No need to inject the window into the api instance anymore --- #

    # Start the pywebview event loop (no args needed here now)
    webview.start(debug=True)


    print("Pywebview window closed. Exiting application.")
    # Flask thread will exit automatically as it's a daemon