# A script to handle file updates in the pipeline.

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
from pathlib import Path

class FileHandler(FileSystemEventHandler):
    def __init__(self, analyser, suffix):
        self.analyser = analyser
        self.suffix = suffix
        self.last_new = time.time()

    def on_created(self, event):
        if event.src_path.endswith(self.suffix):
            path = Path(event.src_path)
            self.analyser.process_file(path)  
            self.last_new = time.time()