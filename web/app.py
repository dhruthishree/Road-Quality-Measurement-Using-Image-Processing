"""
Road Quality Measurement Dashboard Web Application
Flask Backend with Real-Time Video Streaming & Civil Engineering Analytics API
"""

import os
import cv2
import json
import glob
from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename
from src.pipeline import RoadQualityPipeline
from src.config import CONFIG

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "..", "output", "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Cache recent survey results
CACHED_RESULTS = {}

def get_available_routes():
    """Lists benchmark routes in dataset/routes."""
    routes_dir = os.path.join(os.path.dirname(__file__), "..", "dataset", "routes")
    route_paths = glob.glob(os.path.join(routes_dir, "*", "*.mp4"))
    routes = []
    for rp in sorted(route_paths):
        name = os.path.splitext(os.path.basename(rp))[0]
        routes.append({
            "name": name,
            "path": rp,
            "size_mb": round(os.path.getsize(rp) / (1024 * 1024), 2)
        })
    return routes

@app.route("/")
def index():
    routes = get_available_routes()
    return render_template("index.html", routes=routes)

@app.route("/api/routes")
def api_routes():
    return jsonify(get_available_routes())

@app.route("/api/run_survey/<route_name>")
def run_survey(route_name):
    """Processes a chosen route or returns cached result."""
    routes_dir = os.path.join(os.path.dirname(__file__), "..", "dataset", "routes")
    video_path = os.path.join(routes_dir, route_name, f"{route_name}.mp4")
    
    # Check if uploaded video
    if not os.path.exists(video_path):
        video_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{route_name}.mp4")
        if not os.path.exists(video_path):
            video_path = os.path.join(app.config["UPLOAD_FOLDER"], route_name)
            
    if not os.path.exists(video_path):
        return jsonify({"error": f"Route video not found: {route_name}"}), 404

    # Run pipeline
    pipeline = RoadQualityPipeline()
    res = pipeline.process_video(video_path)
    CACHED_RESULTS[route_name] = res

    return jsonify({
        "route_name": route_name,
        "summary": res["summary"],
        "unique_defects": res["unique_defects"],
        "processed_frames": res["processed_frames"],
        "html_report": os.path.basename(res["html_report_path"]),
        "segments_csv": os.path.basename(res["segments_csv_path"]),
        "defects_csv": os.path.basename(res["defects_csv_path"])
    })

def generate_stream_frames(route_name, mode="annotated"):
    """MJPEG stream generator for live video with overlays."""
    routes_dir = os.path.join(os.path.dirname(__file__), "..", "dataset", "routes")
    video_path = os.path.join(routes_dir, route_name, f"{route_name}.mp4")
    
    if not os.path.exists(video_path):
        video_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{route_name}.mp4")
        if not os.path.exists(video_path):
            video_path = os.path.join(app.config["UPLOAD_FOLDER"], route_name)

    if not os.path.exists(video_path):
        return

    cap = cv2.VideoCapture(video_path)
    pipeline = RoadQualityPipeline()
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Loop video for continuous viewing
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            continue

        result = pipeline.process_frame(frame, frame_idx)
        frame_idx += 1

        if mode == "ipm":
            display_img = result["ipm_bev"]
            # Resize for comfortable viewing
            display_img = cv2.resize(display_img, (480, 720))
        else:
            display_img = result["annotated_frame"]

        ret_enc, buffer = cv2.imencode('.jpg', display_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret_enc:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route("/api/video_feed/<route_name>")
def video_feed(route_name):
    mode = request.args.get("mode", "annotated")
    return Response(
        generate_stream_frames(route_name, mode=mode),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
        
    filename = secure_filename(file.filename)
    dest_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(dest_path)
    
    route_name = os.path.splitext(filename)[0]
    return jsonify({
        "success": True,
        "filename": filename,
        "route_name": route_name
    })

@app.route("/reports/<path:filename>")
def download_report(filename):
    return send_from_directory(CONFIG.reports_dir, filename)

def run_app(host="127.0.0.1", port=5000, debug=False):
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    run_app(debug=True)
