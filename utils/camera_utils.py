import cv2
import time
import logging
from typing import Optional, Tuple, Any
import numpy as np
from threading import Lock
import atexit

class Camera:
    def __init__(self, camera_id: int = 0, max_retries: int = 3, timeout: int = 10,
                 width: int = 640, height: int = 480):
        """
        Enhanced webcam controller with robust error handling and adjustable screen size.

        Args:
            camera_id: Index of the camera (0 for default)
            max_retries: Number of connection attempts before failing
            timeout: Seconds to wait for camera initialization
            width: Desired camera width
            height: Desired camera height
        """
        self.camera_id = camera_id
        self.cap = None
        self.width = width
        self.height = height
        self.last_frame = None
        self.frame_lock = Lock()
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.is_released = False
        self._setup_logging()
        self._initialize_camera(max_retries, timeout)
        atexit.register(self._cleanup)

    def _setup_logging(self):
        """Configure logging for camera operations"""
        self.logger = logging.getLogger(f'Camera_{self.camera_id}')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def _initialize_camera(self, max_retries: int, timeout: int):
        """Robust camera initialization with retries and timeout"""
        start_time = time.time()

        for attempt in range(max_retries):
            try:
                self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)

                if not self.cap.isOpened():
                    raise IOError(f"Camera {self.camera_id} not accessible")

                # Set user-defined resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                self.logger.info(f"✅ Camera initialized at {actual_w}x{actual_h}")
                return

            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Camera initialization timed out after {timeout} seconds")
                time.sleep(1)
                continue

        raise RuntimeError(f"Failed to initialize camera after {max_retries} attempts")

    def get_frame(self) -> Optional[np.ndarray]:
        """Safely capture a frame with FPS calculation"""
        if self.is_released:
            self.logger.warning("Camera access attempted after release")
            return None

        with self.frame_lock:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.error("❌ Frame capture failed")
                    return None

                # FPS calculation
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                if elapsed > 0:
                    self.fps = self.frame_count / elapsed

                self.last_frame = frame.copy()
                return frame

            except Exception as e:
                self.logger.error(f"Frame capture exception: {str(e)}")
                return None

    def get_frame_with_metadata(self) -> Tuple[Optional[np.ndarray], dict]:
        """Returns frame along with metadata (timestamp, FPS, etc.)"""
        frame = self.get_frame()
        metadata = {
            'timestamp': time.time(),
            'fps': self.fps,
            'frame_count': self.frame_count,
            'camera_id': self.camera_id
        }
        return frame, metadata

    def get_last_frame(self) -> Optional[np.ndarray]:
        """Returns last captured frame"""
        return self.last_frame

    def release(self):
        """Release camera resources safely"""
        if self.is_released:
            return
        self.logger.info("🔻 Releasing camera resources")
        try:
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            self.is_released = True
        except Exception as e:
            self.logger.error(f"Error during release: {str(e)}")

    def _cleanup(self):
        """Ensure resources are released on exit"""
        if not self.is_released:
            self.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def get_properties(self) -> dict:
        """Get camera properties"""
        if self.is_released:
            return {}
        return {
            'width': self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            'height': self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
            'exposure': self.cap.get(cv2.CAP_PROP_EXPOSURE)
        }

    def set_property(self, prop_id: int, value: float) -> bool:
        """Set camera property"""
        if self.is_released:
            return False
        try:
            return self.cap.set(prop_id, value)
        except Exception as e:
            self.logger.error(f"Failed to set property {prop_id}: {str(e)}")
            return False

    def get_fps(self) -> float:
        return self.fps

    def reset_counter(self):
        self.frame_count = 0
        self.start_time = time.time()
