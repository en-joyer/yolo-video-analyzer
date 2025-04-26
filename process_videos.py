# YOLO Frame Extraction and Analysis with Person Detection Folder
# This script extracts all frames from videos using ffmpeg, analyzes each frame with YOLO,
# and saves frames with people detected to a separate folder

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
from ultralytics import YOLO
from tqdm import tqdm

# Directory settings
VIDEO_DIR = "."  # Directory containing .mp4 files (current directory)
FRAMES_DIR = "./extracted_frames"  # Directory for extracted frames
PERSON_FRAMES_DIR = "./detected_person"  # Directory for frames with people detected
OUTPUT_DIR = "./yolo_clips"  # Output directory for extracted clips
LOG_FILE = "./processed_videos.txt"  # Log file to track processed videos

# Processing settings
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for YOLO detections
EXTRACT_FRAME_RATE = 2  # Extract 5 frames per second (adjust for efficiency)

# Create necessary directories
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

# Find all .mp4 videos in the directory
video_files = glob(os.path.join(VIDEO_DIR, "*.mp4"))

# Function to extract frames using ffmpeg
def extract_frames(video_path, output_dir, fps=EXTRACT_FRAME_RATE):
    print(f"   🔄 Extracting frames at {fps} fps...")
    
    # Create a dedicated directory for this video's frames
    video_name = os.path.basename(video_path).split('.')[0]
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
        return frames_dir
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error extracting frames: {e}")
        return None

# Function to detect humans in an image and save to person folder if detected
def detect_humans(image_path, video_name):
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
        # Create a dedicated directory for this video's person frames
        video_person_dir = os.path.join(PERSON_FRAMES_DIR, video_name)
        os.makedirs(video_person_dir, exist_ok=True)
        
        # Copy the frame to the person directory
        frame_filename = os.path.basename(image_path)
        person_frame_path = os.path.join(video_person_dir, frame_filename)
        
        # Add bounding boxes to the image
        annotated_image = results[0].plot()  # This draws bounding boxes on the image
        
        # Save the annotated image
        cv2.imwrite(person_frame_path, annotated_image)
    
    return person_detected

# Function to create clips based on frame analysis
def create_clips_from_frames(video_path, frames_dir):
    print("   🔍 Analyzing frames for people...")
    frames = sorted(glob(os.path.join(frames_dir, '*.jpg')))
    
    if not frames:
        print("   ⚠️ No frames found to analyze")
        return 0
    
    # Variables to track clip segments
    clip_segments = []
    in_segment = False
    start_frame = None
    video_name = os.path.basename(video_path).split('.')[0]
    
    # Frame rate used during extraction (for calculating timestamps)
    extraction_fps = EXTRACT_FRAME_RATE
    
    # Process each frame to find segments with people
    for i, frame_path in enumerate(tqdm(frames, desc="Analyzing frames")):
        has_person = detect_humans(frame_path, video_name)
        
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
    for idx, (start_time, end_time) in enumerate(clip_segments):
        output_path = os.path.join(OUTPUT_DIR, f"{video_name}_clip_{idx}.mp4")
        
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
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Error creating clip {idx+1}: {e}")
    
    return len(clip_segments)

# Process each video
for video_path in video_files:
    video_name = os.path.basename(video_path).split('.')[0]
    
    if video_path in processed_videos:
        print(f"⏭️ ALREADY PROCESSED: {video_name}")
        continue

    print(f"🔍 PROCESSING: {video_name}")
    start_time = time.time()
    
    try:
        # Step 1: Extract frames using ffmpeg
        video_frames_dir = extract_frames(video_path, FRAMES_DIR)
        
        if video_frames_dir:
            # Step 2: Analyze frames and create clips
            clips_count = create_clips_from_frames(video_path, video_frames_dir)
            
            # Count how many person frames were detected
            person_frames_dir = os.path.join(PERSON_FRAMES_DIR, video_name)
            if os.path.exists(person_frames_dir):
                person_frames_count = len(glob(os.path.join(person_frames_dir, '*.jpg')))
                print(f"   👤 Saved {person_frames_count} frames with people to {person_frames_dir}")
            
            # Step 3: Clean up extracted frames to save space
            print("   🧹 Cleaning up extracted frames...")
            shutil.rmtree(video_frames_dir)
            
            # Log processing time
            process_time = time.time() - start_time
            print(f"✅ {video_name} processed in {process_time:.1f} seconds. Created {clips_count} clips.")
            
            # Add video to processed log
            with open(LOG_FILE, "a") as f:
                f.write(video_path + "\n")
                
        else:
            print(f"❌ Skipping {video_name} due to frame extraction failure")
            
    except Exception as e:
        print(f"❌ Error processing video {video_name}: {str(e)}")

print("🎉 All videos processed!")