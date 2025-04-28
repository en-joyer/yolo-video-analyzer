# YOLO Frame Extraction and Analysis with Person Detection Folder
# This script extracts all frames from videos using ffmpeg, analyzes each frame with YOLO,
# and saves frames with people detected to a separate folder organized by date

#https://github.com/en-joyer/yolo-video-analyzer
#This script written by @en-joyer


# Install required libraries
#!pip install opencv-python ultralytics tqdm

# Import libraries
import cv2
import os
import subprocess
import shutil
import numpy as np
from glob import glob
import time
import re
import datetime
from ultralytics import YOLO
from tqdm import tqdm

# Directory settings
VIDEO_DIR = "./videos"  # Directory containing .mp4 files (in /videos and its subdirectories)
FRAMES_DIR = "./extracted_frames"  # Directory for extracted frames
PERSON_FRAMES_DIR = "./detected_person"  # Directory for frames with people detected
OUTPUT_DIR = "./yolo_clips"  # Base output directory for extracted clips
LOG_FILE = "./processed_videos.txt"  # Log file to track processed videos

# Processing settings
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for YOLO detections
EXTRACT_FRAME_RATE = 1  # Extract 5 frames per second (adjust for efficiency)

# Create necessary directories
os.makedirs(VIDEO_DIR, exist_ok=True)  # Create videos directory if it doesn't exist
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(PERSON_FRAMES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read previously processed videos
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r") as f:
        processed_videos = set(f.read().splitlines())
else:
    processed_videos = set()

# Load YOLO model
print("Loading YOLO model...")
model = YOLO("yolov8n.pt")  # Using YOLOv8 nano model
print("YOLO model loaded")

# Find all .mp4 videos in the directory AND subdirectories
def find_all_videos(base_dir):
    all_videos = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(".mp4"):
                all_videos.append(os.path.join(root, file))
    return all_videos

# Get all videos
video_files = find_all_videos(VIDEO_DIR)
print(f"Found {len(video_files)} videos in {VIDEO_DIR} and its subdirectories")

# Function to extract date and time from filename (e.g., Camera10_20250425190000_20250425193000.mp4)
def extract_date_from_filename(filename):
    # Extract the base filename without path or extension
    base_name = os.path.basename(filename)
    
    # Try to match the pattern like Camera10_20250425190000_20250425193000.mp4
    # We'll use the first date/time found (start time)
    match = re.search(r'_(\d{14})_', base_name)
    
    if match:
        date_str = match.group(1)
        try:
            # Parse the date string (format: YYYYMMDDHHMMSS)
            year = int(date_str[0:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            hour = int(date_str[8:10])
            minute = int(date_str[10:12])
            second = int(date_str[12:14])
            
            # Format the date as requested: day-month-year---hour-time-second
            formatted_date = f"{day:02d}-{month:02d}-{year}---{hour:02d}-{minute:02d}-{second:02d}"
            date_only = f"{day:02d}-{month:02d}-{year}"
            return formatted_date, date_only
        except (ValueError, IndexError):
            pass
    
    # If no match is found or parsing fails, use current time
    now = datetime.datetime.now()
    formatted_date = f"{now.day:02d}-{now.month:02d}-{now.year}---{now.hour:02d}-{now.minute:02d}-{now.second:02d}"
    date_only = f"{now.day:02d}-{now.month:02d}-{now.year}"
    return formatted_date, date_only

# Function to extract frames using ffmpeg
def extract_frames(video_path, output_dir, fps=EXTRACT_FRAME_RATE):
    print(f"   🔄 Extracting frames at {fps} fps...")
    
    # Create a dedicated directory for this video's frames
    # Use relative path to avoid overly long directory names
    rel_path = os.path.relpath(video_path, VIDEO_DIR)
    video_name = os.path.splitext(rel_path)[0].replace(os.sep, "_")
    frames_dir = os.path.join(output_dir, video_name)
    os.makedirs(frames_dir, exist_ok=True)
    
    # Use ffmpeg to extract frames
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f'fps={fps}',
        '-q:v', '2',  # High quality
        os.path.join(frames_dir, 'frame_%05d.jpg')
    ]
    
    try:
        # Run ffmpeg command
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"   ✅ Frames extracted to {frames_dir}")
        return frames_dir, video_name
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error extracting frames: {e}")
        return None, None

# Function to detect humans in an image and save to person folder if detected
def detect_humans(image_path, video_name, formatted_date, date_only):
    image = cv2.imread(image_path)
    if image is None:
        print(f"   ⚠️ Warning: Could not read image: {image_path}")
        return False
    
    # Run YOLO detection
    results = model(image, verbose=False)
    
    # Check if any persons are detected
    person_detected = False
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = box.conf.item()
            
            # Class 0 is person in COCO dataset
            if class_id == 0 and confidence > CONFIDENCE_THRESHOLD:
                person_detected = True
                break
        
        if person_detected:
            break
    
    # If a person is detected, copy the frame to the person detection folder
    if person_detected:
        # Create a dedicated directory for this date's person frames
        date_person_dir = os.path.join(PERSON_FRAMES_DIR, date_only)
        os.makedirs(date_person_dir, exist_ok=True)
        
        # Copy the frame to the person directory
        frame_filename = os.path.basename(image_path)
        # Create a filename with date format
        timestamp = datetime.datetime.now().strftime("%H-%M-%S-%f")[:10]
        person_frame_name = f"{formatted_date}_{video_name}_{frame_filename}_{timestamp}.jpg"
        person_frame_path = os.path.join(date_person_dir, person_frame_name)
        
        # Add bounding boxes to the image
        annotated_image = results[0].plot()  # This draws bounding boxes on the image
        
        # Save the annotated image
        cv2.imwrite(person_frame_path, annotated_image)
    
    return person_detected

# Function to create clips based on frame analysis
def create_clips_from_frames(video_path, frames_dir, video_name):
    print("   🔍 Analyzing frames for people...")
    frames = sorted(glob(os.path.join(frames_dir, '*.jpg')))
    
    if not frames:
        print("   ⚠️ No frames found to analyze")
        return 0
    
    # Variables to track clip segments
    clip_segments = []
    in_segment = False
    start_frame = None
    
    # Extract date from filename for output organization
    formatted_date, date_only = extract_date_from_filename(video_path)
    
    # Create date-based output directory
    date_output_dir = os.path.join(OUTPUT_DIR, date_only)
    os.makedirs(date_output_dir, exist_ok=True)
    
    # Frame rate used during extraction (for calculating timestamps)
    extraction_fps = EXTRACT_FRAME_RATE
    
    # Process each frame to find segments with people
    for i, frame_path in enumerate(tqdm(frames, desc="Analyzing frames")):
        has_person = detect_humans(frame_path, video_name, formatted_date, date_only)
        
        # Start of a new segment with people
        if has_person and not in_segment:
            start_frame = i
            in_segment = True
        
        # End of a segment with people
        elif not has_person and in_segment:
            # Calculate timestamp in seconds
            start_time = start_frame / extraction_fps
            end_time = i / extraction_fps
            
            clip_segments.append((start_time, end_time))
            in_segment = False
    
    # Handle case where video ends while still in a segment
    if in_segment:
        start_time = start_frame / extraction_fps
        end_time = len(frames) / extraction_fps
        clip_segments.append((start_time, end_time))
    
    # Create clips from identified segments
    print(f"   🎬 Creating {len(clip_segments)} clips...")
    created_clips = 0
    
    for idx, (start_time, end_time) in enumerate(clip_segments):
        # Generate unique timestamp for this clip
        clip_time = datetime.datetime.now().strftime("%H-%M-%S-%f")[:10]
        output_filename = f"{formatted_date}_clip{idx}_{clip_time}.mp4"
        output_path = os.path.join(date_output_dir, output_filename)
        
        # Use ffmpeg to extract the clip
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-ss', str(start_time),
            '-to', str(end_time),
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-y',  # Overwrite output files
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"   ✅ Created clip {idx+1}/{len(clip_segments)}: {start_time:.1f}s to {end_time:.1f}s")
            created_clips += 1
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Error creating clip {idx+1}: {e}")
    
    return created_clips, date_only

# Process each video
for video_path in video_files:
    # Check if we've already processed this video
    if video_path in processed_videos:
        print(f"⏭️ ALREADY PROCESSED: {video_path}")
        continue

    print(f"🔍 PROCESSING: {video_path}")
    start_time = time.time()
    
    try:
        # Step 1: Extract frames using ffmpeg
        video_frames_dir, video_name = extract_frames(video_path, FRAMES_DIR)
        
        if video_frames_dir and video_name:
            # Step 2: Analyze frames and create clips
            clips_count, date_only = create_clips_from_frames(video_path, video_frames_dir, video_name)
            
            # Count how many person frames were detected
            person_frames_dir = os.path.join(PERSON_FRAMES_DIR, date_only)
            if os.path.exists(person_frames_dir):
                # Count only frames related to this video
                person_frames_count = len([f for f in os.listdir(person_frames_dir) if video_name in f])
                print(f"   👤 Saved {person_frames_count} frames with people to {person_frames_dir}")
            
            # Step 3: Clean up extracted frames to save space
            print("   🧹 Cleaning up extracted frames...")
            shutil.rmtree(video_frames_dir)
            
            # Log processing time
            process_time = time.time() - start_time
            print(f"✅ {video_path} processed in {process_time:.1f} seconds. Created {clips_count} clips.")
            
            # Add video to processed log
            with open(LOG_FILE, "a") as f:
                f.write(video_path + "\n")
                
        else:
            print(f"❌ Skipping {video_path} due to frame extraction failure")
            
    except Exception as e:
        print(f"❌ Error processing video {video_path}: {str(e)}")

print("🎉 All videos processed!")