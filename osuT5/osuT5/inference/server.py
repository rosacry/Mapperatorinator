import os
import time
import threading
import torch
from multiprocessing.connection import Listener, Client

from torch.utils.data import default_collate
from transformers import EncoderDecoderCache, Cache, StaticCache

from osuT5.osuT5.inference.cache_utils import get_cache
from osuT5.osuT5.model import Mapperatorinator

# Path for the Unix domain socket used for IPC
SOCKET_PATH = '/tmp/transformer_batch.sock'
# Maximum number of inputs to batch per generation call
BATCH_SIZE = 8
# Maximum time to wait (in seconds) for more requests to form a batch
BATCH_TIMEOUT = 0.1
# Idle time (in seconds) before shutting down due to no clients
IDLE_TIMEOUT = 60


class InferenceServer:
    def __init__(self, model, socket_path=SOCKET_PATH):
        self.model: Mapperatorinator = model
        self.socket_path = socket_path
        self.requests = {}  # holds pending requests
        self.lock = threading.Lock()
        self.shutdown_flag = threading.Event()
        self.listener = None

    def start(self):
        # Remove stale socket
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass

        # Start IPC listener
        self.listener = Listener(self.socket_path, family='AF_PIPE')
        threading.Thread(target=self._listener_thread, daemon=True).start()
        # Start batcher thread
        threading.Thread(target=self._batch_thread, daemon=True).start()
        # Start idle monitor
        threading.Thread(target=self._idle_monitor, daemon=True).start()

    def _listener_thread(self):
        while not self.shutdown_flag.is_set():
            conn = self.listener.accept()
            # Handle each client in its own thread
            threading.Thread(target=self._client_handler, args=(conn,), daemon=True).start()

    def _client_handler(self, conn):
        with conn:
            while True:
                try:
                    model_kwargs, generate_kwargs = conn.recv()
                except EOFError:
                    break

                # Prepare a response event
                response_event = threading.Event()
                record = {'model_kwargs': model_kwargs, 'conn': conn, 'event': response_event, 'result': None}

                # Enqueue request
                with self.lock:
                    if generate_kwargs in self.requests:
                        self.requests[generate_kwargs].append(record)
                    else:
                        self.requests[generate_kwargs] = [record]

                # Wait until batch thread processes it
                response_event.wait()

                # Send back result
                conn.send(record['result'])

    def _batch_thread(self):
        while not self.shutdown_flag.is_set():
            time.sleep(BATCH_TIMEOUT)
            with self.lock:
                if not self.requests:
                    continue
                generate_kwargs: dict = list(self.requests.keys())[0]
                records = self.requests[generate_kwargs]
                batch = records[:BATCH_SIZE]
                self.requests[generate_kwargs] = records[BATCH_SIZE:]
                if not self.requests[generate_kwargs]:
                    del self.requests[generate_kwargs]

            # Collate inputs
            model_kwargs = default_collate([record['model_kwargs'] for record in batch])

            cache = get_cache(self.model, len(batch), generate_kwargs.get('num_beams', 1), generate_kwargs.pop('cfg_scale', 1.0))

            # Perform batched generation
            outputs = self.model.generate(
                model_kwargs,
                use_cache=True,
                past_key_values=cache,
                **generate_kwargs,
            )

            # Split and dispatch results
            for out, record in zip(outputs, batch):
                record['result'] = out
                record['event'].set()

    def _idle_monitor(self):
        last_activity = time.time()
        while not self.shutdown_flag.is_set():
            time.sleep(IDLE_TIMEOUT / 2)
            with self.lock:
                if self.requests:
                    last_activity = time.time()
            if time.time() - last_activity > IDLE_TIMEOUT:
                # No requests for a while: shutdown
                self.shutdown_flag.set()
                try:
                    self.listener.close()
                    os.unlink(self.socket_path)
                except Exception:
                    pass


class InferenceClient:
    def __init__(self, model_loader, socket_path=SOCKET_PATH):
        self.socket_path = socket_path
        try:
            self.conn = Client(self.socket_path, family='AF_PIPE')
        except FileNotFoundError:
            # No server: start one
            threading.Thread(target=self._start_server, args=(model_loader,), daemon=True).start()
            # Wait for server socket to appear
            while not os.path.exists(self.socket_path):
                time.sleep(0.1)
            self.conn = Client(self.socket_path, family='AF_PIPE')

    def _start_server(self, model_loader):
        # Load model inside server process
        model = model_loader()
        server = InferenceServer(model, socket_path=self.socket_path)
        server.start()
        # Block until shutdown
        while not server.shutdown_flag.is_set():
            time.sleep(1)

    def generate(self, model_kwargs, generate_kwargs):
        # Send request and wait for response
        self.conn.send((model_kwargs, generate_kwargs))
        return self.conn.recv()
