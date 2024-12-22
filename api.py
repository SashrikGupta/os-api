from flask import Flask, request, jsonify
import cv2
import base64
import time
import json
import numpy as np
from ultralytics import YOLO, solutions
import io
import cloudinary
import cloudinary.uploader

cloudinary.config(
    cloud_name="dyggq3cmo",
    api_key="513111976336514",
    api_secret="fgR0Rc6kn8E9hfpPv_Qyi3y00N8"
)

app = Flask(__name__)

from datetime import datetime

# Function to merge continuous times
from datetime import datetime, timedelta

def merge_times(high_density_times):
    # Filter out entries where 'end_time' is None
    high_density_times = [entry for entry in high_density_times if entry['end_time'] is not None]
    
    # Sort the high_density_times based on start_time
    sorted_times = sorted(high_density_times, key=lambda x: datetime.strptime(x['start_time'], "%H:%M"))
    
    merged_times = []
    current_start = current_end = None

    for time_entry in sorted_times:
        start_time = time_entry['start_time']
        end_time = time_entry['end_time']
        
        # Convert start_time and end_time to datetime objects for comparison
        start_dt = datetime.strptime(start_time, "%H:%M")
        end_dt = datetime.strptime(end_time, "%H:%M")

        if current_start is None:  # First iteration
            current_start = start_time
            current_end = end_time
        else:
            # Check if the current time's start is either equal to the previous end time
            # or exactly one minute after the previous end time
            current_end_dt = datetime.strptime(current_end, "%H:%M")
            if start_dt == current_end_dt or start_dt == current_end_dt + timedelta(minutes=1):
                current_end = end_time  # Extend the current range
            else:
                # Save the previous merged time and reset for the next range
                merged_times.append({
                    'start_time': current_start,
                    'end_time': current_end
                })
                current_start = start_time
                current_end = end_time
    
    # Append the last merged time
    if current_start is not None:
        merged_times.append({
            'start_time': current_start,
            'end_time': current_end
        })
    
    return merged_times


# Prepare high_density_times by zone




# Path to the YOLO model
model_path = 'yolo11n.pt'
bb_box_model = YOLO(model_path)
heatmap_model = solutions.Heatmap(
    show=False,  # Streamlit will handle visualization
    model=model_path,  # Use the same model as bb_box_model
    colormap=cv2.COLORMAP_JET,  # Heatmap colormap
    classes=[0]  # Only detecting people (class 0)
)
full_set = set()
# Function to check if a point is inside a rectangular zone
def is_center_in_zone(center, zone):
    x, y = center
    return zone['x1'] <= x <= zone['x2'] and zone['y1'] <= y <= zone['y2']

# Generate heatmap for a frame
def generate_heatmap(frame):
    return heatmap_model.generate_heatmap(frame)


# Process the video from the URL
def process_video_from_url(video_url, zones):
    
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        return {"error": f"Error opening video stream from URL: {video_url}"}

    zones = json.loads(zones)
    zone_stats = {zone['id']: {'footfall': set(), 'current_persons': 0, 'high_density_time': [], 'above_threshold': False, 'history': []} for zone in zones}
    footfall_data = {zone['id']: [] for zone in zones}  
    my_foot_fall_data = {
         f'{zone["id"] }': set() for zone in zones
      }
    total_zones = len(zones)
    frame_count = 0
    all_heatmap_frames = []  # List to store all the heatmap frames

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Process detection with YOLO and generate heatmap
        results = bb_box_model.track(frame, persist=True, verbose=False)
        heatmap_frame = generate_heatmap(frame)
        all_heatmap_frames.append(heatmap_frame)  # Accumulate the frames

        # Process YOLO detections
        tracked_centers = []
        person_count = 0
        for result in results[0].boxes:
            class_id = int(result.cls.item())
            if class_id == 0:  # Person class
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center of bounding box
                tracked_centers.append({'track_id': int(result.id.item()), 'center': (cx, cy)})
                person_count += 1
                full_set.add(int(result.id.item()))

        zone_person_count = []
        threshold = person_count / total_zones if total_zones > 0 else 0
        print("full : " , len(full_set))
        for zone in zones:
            zone_id = zone['id']
            zone_info = zone_stats[zone_id]
            current_zone_count = 0

            for person in tracked_centers:
                if is_center_in_zone(person['center'], zone):
                    current_zone_count += 1
                    zone_info['footfall'].add(person['track_id'])
                    my_foot_fall_data[f'{zone["id"]}'].add(person['track_id'])
                    print(zone["id"] , ":" ,len(my_foot_fall_data[f'{zone["id"]}']) )

            zone_info['current_persons'] = current_zone_count
            zone_person_count.append(current_zone_count)

            # Manage high-density times and other stats
            if current_zone_count > threshold:
                if not zone_info['above_threshold']:
                    zone_info['high_density_time'].append({'start_time': frame_count})
                    zone_info['above_threshold'] = True
            else:
                if zone_info['above_threshold']:
                    # Ensure 'end_time' is set before transitioning back
                    zone_info['high_density_time'][-1]['end_time'] = frame_count
                    zone_info['above_threshold'] = False

            zone_info['history'].append((frame_count, current_zone_count, len(zone_info['footfall'])))

        frame_count += 1

    cap.release()


    my_final_foot_fall_data = {}
    for id , sets  in my_foot_fall_data.items() : 
        my_final_foot_fall_data[id] = len(sets)
        

    # Create the final heatmap image after all frames are processed
    final_heatmap = np.mean(all_heatmap_frames, axis=0).astype(np.uint8)

    # Convert heatmap to a file-like object in memory
    _, img_encoded = cv2.imencode('.jpg', final_heatmap)
    heatmap_file = io.BytesIO(img_encoded.tobytes())  # Convert to bytes for file upload

    # Upload the heatmap image to Cloudinary
    upload_result = cloudinary.uploader.upload(heatmap_file)
    heatmap_url = upload_result['secure_url']

    # Prepare the footfall summary

    # Convert ndarray to list

    # Prepare high_density_times by zone
    zone_high_density_times = {}

    for zone in zones:
      zone_id = zone['id']
         
         # Gather high_density_times for each zone
      high_density_times_for_zone = [
            {
                  "start_time": time.strftime("%H:%M", time.gmtime(high_density_time['start_time'])),
                  "end_time": time.strftime("%H:%M", time.gmtime(high_density_time['end_time'])) if 'end_time' in high_density_time else None,
                  "zone_id": zone_id
            }
            for high_density_time in zone_stats[zone_id]['high_density_time']
         ]
         
         # Merge high_density_times for this zone
      zone_high_density_times[zone_id] = merge_times(high_density_times_for_zone)

      # Create the footfall summary
    footfall_summary = {
         "total_footfall": len(full_set),
         "zone_footfall": my_final_foot_fall_data,
         "high_density_times": [
            {
                  "start_time": entry['start_time'],
                  "end_time": entry['end_time'],
                  "zone_id": zone_id
            }
            for zone_id, times in zone_high_density_times.items()
            for entry in times
         ]
      }

    return {
         "footfall_summary": footfall_summary,
         "heatmap_url": heatmap_url  # Return the Cloudinary URL for the heatmap
      }


@app.route('/process_video', methods=['POST'])
def process_video():
    data = request.json
    video_url = data.get('video_stream_url')
    zones = data.get('zones')

    if not video_url or not zones:
        return jsonify({"error": "Missing video_url or zones"}), 400
    
    result = process_video_from_url(video_url, json.dumps(zones))
    if 'error' in result:
        return jsonify(result), 400
    
    return jsonify(result)

if __name__ == "__main__":
    port =5000
    app.run(host='0.0.0.0', port=port, debug=False)