import argparse
import json
import time
import cv2
import numpy as np
import paho.mqtt.client as mqtt
from ultralyticsplus import YOLO, render_result
from datetime import datetime


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_ID      = "lewiswatson/yolov8x-tuned-hand-gestures"
MQTT_BROKER   = "broker.hivemq.com"   # Public test broker (free)
MQTT_PORT     = 1883
MQTT_TOPIC    = "hand/gesture"        # Main topic
MQTT_TOPIC_RAW= "hand/gesture/raw"    # Full JSON payload
CONFIDENCE    = 0.45                  # Detection threshold
PUBLISH_DELAY = 0.3                   # Minimum seconds between publishes


# ─────────────────────────────────────────────
# MQTT HELPER
# ─────────────────────────────────────────────
class MQTTPublisher:
    def __init__(self, broker: str, port: int):
        self.broker = broker
        self.port   = port
        self.client = mqtt.Client(client_id=f"hand-gesture-{int(time.time())}")
        self.client.on_connect    = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.connected = False
        self._last_publish = 0.0

    def _on_connect(self, client, userdata, flags, rc):
        codes = {0: "✅ Connected", 1: "❌ Wrong protocol", 4: "❌ Bad credentials", 5: "❌ Not authorised"}
        print(f"[MQTT] {codes.get(rc, f'rc={rc}')}  →  {self.broker}:{self.port}")
        self.connected = (rc == 0)

    def _on_disconnect(self, client, userdata, rc):
        self.connected = False
        print(f"[MQTT] Disconnected (rc={rc})")

    def connect(self):
        print(f"[MQTT] Connecting to {self.broker}:{self.port} ...")
        self.client.connect(self.broker, self.port, keepalive=60)
        self.client.loop_start()
        time.sleep(1.5)          # Wait for connection

    def publish(self, gesture: str, confidence: float, count: int):
        now = time.time()
        if now - self._last_publish < PUBLISH_DELAY:
            return                # Rate-limit

        payload_simple = gesture
        payload_full   = json.dumps({
            "gesture"   : gesture,
            "confidence": round(confidence, 3),
            "count"     : count,
            "timestamp" : datetime.now().isoformat()
        })

        if self.connected:
            self.client.publish(MQTT_TOPIC,     payload_simple, qos=1)
            self.client.publish(MQTT_TOPIC_RAW, payload_full,   qos=1)
            self._last_publish = now
            print(f"[MQTT ▶] Topic: {MQTT_TOPIC}  |  {payload_full}")
        else:
            print(f"[MQTT ✗] Not connected — skipped: {gesture}")

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()


# ─────────────────────────────────────────────
# DETECTOR
# ─────────────────────────────────────────────
class HandGestureDetector:
    def __init__(self, mqtt_pub: MQTTPublisher):
        print(f"[MODEL] Loading {MODEL_ID} ...")
        self.model = YOLO(MODEL_ID)
        self.model.overrides["conf"]  = CONFIDENCE
        self.model.overrides["iou"]   = 0.45
        self.model.overrides["max_det"] = 10
        self.mqtt = mqtt_pub
        print("[MODEL] Ready ✓")

    # ── Single image / frame inference ──────────
    def predict_frame(self, frame):
        results = self.model.predict(frame, verbose=False)
        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i, cls_id in enumerate(boxes.cls.tolist()):
                name  = results[0].names[int(cls_id)]
                conf  = boxes.conf[i].item()
                detections.append({"gesture": name, "confidence": conf})
        return results[0], detections

    # ── Annotate frame with bounding boxes ──────
    @staticmethod
    def annotate(frame, detections):
        for d in detections:
            label = f"{d['gesture']} {d['confidence']:.2f}"
            # Simple banner at top (render_result handles the boxes)
            cv2.putText(frame, label, (10, 30 + detections.index(d) * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 80), 2)
        return frame

    # ── STATIC IMAGE mode ────────────────────────
    def run_image(self, path: str):
        print(f"\n[IMG] Processing: {path}")
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")

        result, detections = self.predict_frame(img)

        # Render via ultralyticsplus (PIL)
        render = render_result(model=self.model, image=path, result=result)

        print(f"\n  Found {len(detections)} gesture(s):")
        for d in detections:
            print(f"  ✋  {d['gesture']:20s}  conf={d['confidence']:.3f}")
            self.mqtt.publish(d["gesture"], d["confidence"], len(detections))

        # Convert PIL → OpenCV for display
        img_out = cv2.cvtColor(np.array(render), cv2.COLOR_RGB2BGR)
        cv2.imshow("Hand Gesture Detection", img_out)
        print("\n[INFO] Press any key to close …")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ── WEBCAM / VIDEO mode ──────────────────────
    def run_video(self, source=0):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")

        src_label = "Webcam" if isinstance(source, int) else source
        print(f"\n[VIDEO] Source: {src_label}")
        print("[INFO]  Press  Q  to quit\n")

        fps_time = time.time()
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Stream ended.")
                break

            frame_count += 1
            result, detections = self.predict_frame(frame)

            # ── Draw FPS ──
            elapsed = time.time() - fps_time
            fps     = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)

            # ── Draw detections ──
            for d in detections:
                label = f"{d['gesture']}  {d['confidence']:.2f}"
                cv2.putText(frame, label, (10, 30 + detections.index(d) * 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 80), 2)

            # ── MQTT publish (best detection per frame) ──
            if detections:
                best = max(detections, key=lambda x: x["confidence"])
                self.mqtt.publish(best["gesture"], best["confidence"], len(detections))

            # ── Bounding boxes via YOLO plot ──
            annotated = result.plot()                          # BGR numpy array
            annotated = cv2.resize(annotated,
                                   (frame.shape[1], frame.shape[0]))
            cv2.imshow("Hand Gesture Detection [Q=quit]", annotated)

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Stopped.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Hand Gesture Detection + MQTT Publisher"
    )
    parser.add_argument("--source",  default=0,
                        help="Input source: 0 (webcam), path to image/video")
    parser.add_argument("--broker",  default=MQTT_BROKER,
                        help=f"MQTT broker hostname (default: {MQTT_BROKER})")
    parser.add_argument("--port",    default=MQTT_PORT, type=int,
                        help=f"MQTT broker port (default: {MQTT_PORT})")
    parser.add_argument("--topic",   default=MQTT_TOPIC,
                        help=f"MQTT topic (default: {MQTT_TOPIC})")
    parser.add_argument("--conf",    default=CONFIDENCE, type=float,
                        help=f"Detection confidence threshold (default: {CONFIDENCE})")
    args = parser.parse_args()

    # Override globals from args
    global MQTT_TOPIC, CONFIDENCE
    MQTT_TOPIC  = args.topic
    CONFIDENCE  = args.conf

    # Try to parse source as int (webcam index)
    try:
        source = int(args.source)
    except (ValueError, TypeError):
        source = args.source          # file path string

    # ── Setup MQTT ──
    pub = MQTTPublisher(args.broker, args.port)
    pub.connect()

    # ── Setup Detector ──
    detector = HandGestureDetector(pub)

    # ── Run ──
    try:
        if isinstance(source, str) and source.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            detector.run_image(source)
        else:
            detector.run_video(source)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        pub.disconnect()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()