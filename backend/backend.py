from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import time
import threading
import traceback
import gc

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "allow_headers": ["Content-Type"],
        "expose_headers": ["Content-Type"]
    }
})

# Global variables for camera and background
camera = None
camera_lock = threading.Lock()
stream_active = False
current_stream = None

def get_camera():
    global camera
    with camera_lock:
        if camera is None:
            # Force garbage collection before attempting to open camera
            gc.collect()
            
            for i in range(3):  # Try 3 times
                try:
                    camera = cv2.VideoCapture(0)
                    if camera.isOpened():
                        # Set camera properties for better performance
                        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        camera.set(cv2.CAP_PROP_FPS, 30)
                        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size
                        
                        # Warm up the camera
                        for _ in range(5):
                            ret, _ = camera.read()
                            if not ret:
                                raise Exception("Failed to read frame during warmup")
                            time.sleep(0.1)  # Short delay between warmup frames
                        return camera
                    else:
                        if camera:
                            camera.release()
                            camera = None
                except Exception as e:
                    print(f"Camera initialization attempt {i+1} failed: {str(e)}")
                    if camera:
                        try:
                            camera.release()
                        except:
                            pass
                        camera = None
                    time.sleep(1)  # Wait before retrying
            raise RuntimeError("Could not access the camera after multiple attempts")
        return camera

def release_camera():
    global camera, stream_active, current_stream
    with camera_lock:
        stream_active = False
        current_stream = None
        if camera is not None:
            try:
                camera.release()
            except Exception as e:
                print(f"Error releasing camera: {str(e)}")
            finally:
                camera = None
                gc.collect()  # Force garbage collection after release

@app.route('/health')
def health_check():
    try:
        # Try to initialize the camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return jsonify({"status": "error", "message": "Camera not accessible"}), 500
        
        # Try to read a frame
        ret, _ = cap.read()
        
        try:
            cap.release()
        except:
            pass
        finally:
            cap = None
            gc.collect()
        
        if not ret:
            return jsonify({"status": "error", "message": "Cannot read from camera"}), 500
            
        return jsonify({"status": "healthy"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/invisibility_cloak')
def invisibility_cloak():
    global stream_active, current_stream
    
    try:
        # Stop any existing stream
        stream_active = False
        if current_stream:
            current_stream = None
        release_camera()
        time.sleep(0.5)  # Wait for resources to be released
        
        # Get camera with lock
        cap = get_camera()
        stream_active = True
        
        print("Camera opened successfully, capturing background...")
        
        # Capture the background for the invisibility effect
        background = None
        success_frames = 0
        max_attempts = 30
        
        for i in range(max_attempts):
            if not stream_active:
                raise RuntimeError("Stream stopped by user")
                
            ret, frame = cap.read()
            if ret and frame is not None:
                success_frames += 1
                background = frame.copy()  # Make a copy to ensure we have our own data
                print(f"Captured frame {success_frames}")
            else:
                print(f"Failed to capture frame {i+1}")
            time.sleep(0.1)  # Short delay between frames
            
        if not stream_active:
            raise RuntimeError("Stream stopped by user")
            
        if success_frames < 5:
            raise RuntimeError(f"Failed to capture stable background (got {success_frames} frames)")
            
        if background is None:
            raise RuntimeError("Background capture failed")
            
        background = np.flip(background, axis=1)
        print("Background captured successfully")

        def generate():
            nonlocal background
            frame_count = 0
            last_frame_time = time.time()
            
            try:
                while stream_active:
                    current_time = time.time()
                    # Maintain consistent frame rate
                    if current_time - last_frame_time < 0.033:  # ~30 FPS
                        time.sleep(0.001)
                        continue
                        
                    ret, img = cap.read()
                    if not ret or img is None:
                        print("Failed to read frame")
                        if not stream_active:
                            break
                        continue

                    try:
                        frame_count += 1
                        img = np.flip(img, axis=1)
                        
                        # Every 30 frames (about 1 second), update background
                        if frame_count % 30 == 0:
                            background = img.copy()
                        
                        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                        # Detect red color (adjusted for better detection)
                        lower_red1 = np.array([0, 100, 50])
                        upper_red1 = np.array([10, 255, 255])
                        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

                        lower_red2 = np.array([160, 100, 50])
                        upper_red2 = np.array([180, 255, 255])
                        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

                        # Combine and refine masks
                        mask = mask1 + mask2
                        kernel = np.ones((3, 3), np.uint8)
                        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

                        # Apply the effect
                        img[np.where(mask == 255)] = background[np.where(mask == 255)]

                        # Add some post-processing for smoother edges
                        img = cv2.GaussianBlur(img, (5, 5), 0)

                        # Encode with higher quality
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                        _, buffer = cv2.imencode('.jpg', img, encode_param)
                        if buffer is None:
                            raise Exception("Failed to encode frame")
                            
                        frame = buffer.tobytes()
                        last_frame_time = current_time
                        
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                        
                    except Exception as frame_error:
                        print(f"Error processing frame: {str(frame_error)}")
                        if not stream_active:
                            break
                        continue
                        
            except Exception as e:
                print(f"Error in generate: {str(e)}")
                traceback.print_exc()
            finally:
                release_camera()

        current_stream = generate()
        response = Response(
            current_stream,
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        response.headers['Connection'] = 'close'
        return response
        
    except Exception as e:
        print(f"Error in invisibility_cloak: {str(e)}")
        traceback.print_exc()
        release_camera()
        return jsonify({"error": str(e)}), 500

@app.route('/stop')
def stop_camera():
    global stream_active, current_stream
    try:
        stream_active = False
        if current_stream:
            current_stream = None
        release_camera()
        gc.collect()  # Force garbage collection
        return jsonify({"status": "Camera stopped"})
    except Exception as e:
        print(f"Error stopping camera: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    try:
        app.run(debug=True, threaded=True)
    finally:
        release_camera()
        gc.collect()
