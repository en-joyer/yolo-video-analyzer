# YOLO Frame Extraction and Analysis with Person Detection Folder
# This script extracts all frames from videos using ffmpeg, analyzes each frame with YOLO,
# and saves frames with people detected to a separate folder organized by date

#https://github.com/en-joyer/yolo-video-analyzer
#This script written by @en-joyer

# Install required libraries
#!pip install opencv-python ultralytics tqdm python-dotenv

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
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Directory settings (with .env fallbacks)
# Check if .env file exists
if not os.path.exists('.env'):
    print("Warning: .env file not found. Using default values.")

# Directory settings from .env file
VIDEO_DIR = os.getenv("VIDEO_DIR")
FRAMES_DIR = os.getenv("FRAMES_DIR")
EXTERNAL_OUTPUT_DIR = os.getenv("EXTERNAL_OUTPUT_DIR")  
PERSON_FRAMES_DIR = os.getenv("PERSON_FRAMES_DIR")
CUSTOM_OUTPUT_DIR = os.getenv("CUSTOM_OUTPUT_DIR")
PROCESSED_FRAMES_LOG = os.getenv("PROCESSED_FRAMES_LOG")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
LOG_FILE = os.getenv("LOG_FILE")
LOG_FILE_FRAMES = os.getenv("LOG_FILE_FRAMES")

# Set default values only if environment variables are not set
VIDEO_DIR = VIDEO_DIR if VIDEO_DIR else "./videos"  # Directory containing .mp4 files
FRAMES_DIR = FRAMES_DIR if FRAMES_DIR else "./extracted_frames"  # Directory for extracted frames
EXTERNAL_OUTPUT_DIR = EXTERNAL_OUTPUT_DIR if EXTERNAL_OUTPUT_DIR else "./external_output"  # External frames to analyze
PERSON_FRAMES_DIR = PERSON_FRAMES_DIR if PERSON_FRAMES_DIR else "./detected_person"  # Base directory for frames with people detected
CUSTOM_OUTPUT_DIR = CUSTOM_OUTPUT_DIR if CUSTOM_OUTPUT_DIR else "./detected_person/custom"  # Specific output directory for detected frames. You can define by yourself
PROCESSED_FRAMES_LOG = PROCESSED_FRAMES_LOG if PROCESSED_FRAMES_LOG else "./has_processed_external_frames.txt"
OUTPUT_DIR = OUTPUT_DIR if OUTPUT_DIR else "./yolo_clips"  # Base output directory for extracted clips
LOG_FILE = LOG_FILE if LOG_FILE else "./processed_videos.txt"  # Log file to track processed videos
LOG_FILE_FRAMES = LOG_FILE_FRAMES if LOG_FILE_FRAMES else "./processed_frames.txt"  # Log file to track processed external frames

# Processing settings
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))  # Minimum confidence for YOLO detections
EXTRACT_FRAME_RATE = int(os.getenv("EXTRACT_FRAME_RATE", "1"))  # Extract 1 frame per second (adjust for efficiency. You can set 5, 30, 60 or anything.)
DETECTION_OPACITY = float(os.getenv("DETECTION_OPACITY", "0.5"))  # Opacity for detection bounding boxes (0.0-1.0)
CREATE_CLIPS = os.getenv("CREATE_CLIPS", "True").lower() == "true"  # Set to False to disable clip creation
PROCESS_EXTERNAL_FRAMES = os.getenv("PROCESS_EXTERNAL_FRAMES", "True").lower() == "true"  # Enable/disable external frame processing
USE_GPU = os.getenv("USE_GPU", "False").lower() == "true"  # Enable/disable GPU for YOLO

# Create necessary directories
os.makedirs(VIDEO_DIR, exist_ok=True)  # Create videos directory if it doesn't exist
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(PERSON_FRAMES_DIR, exist_ok=True)
os.makedirs(EXTERNAL_OUTPUT_DIR, exist_ok=True)  # Create CCAM1 specific output directory
if CREATE_CLIPS:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read previously processed files
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r") as f:
        processed_videos = set(f.read().splitlines())
else:
    processed_videos = set()

if os.path.exists(LOG_FILE_FRAMES):
    with open(LOG_FILE_FRAMES, "r") as f:
        processed_frames = set(f.read().splitlines())
else:
    processed_frames = set()

# Load YOLO model
print("Loading YOLO model...")
# GPU support debug section
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

# Try to enable GPU based on settings and availability
try:
    if USE_GPU and torch.cuda.is_available():
        print("GPU acceleration should be enabled")
        # Set environment variables
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        if USE_GPU and not torch.cuda.is_available():
            print("Warning: GPU requested but CUDA is not available. Using CPU instead.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        USE_GPU = False

    # Load model
    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt")
    
    # Try to explicitly move model to GPU if requested
    if USE_GPU and torch.cuda.is_available():
        try:
            print("Attempting to move model to CUDA...")
            model.to('cuda')
            print("Model successfully moved to CUDA")
        except Exception as e:
            print(f"Error moving model to CUDA: {e}")
            print("Falling back to CPU")
            USE_GPU = False
            
    print(f"YOLO model loaded on {'GPU' if USE_GPU and torch.cuda.is_available() else 'CPU'}")
    
except Exception as e:
    print(f"Error during GPU setup: {e}")
    print("Falling back to CPU-only mode")
    model = YOLO("yolov8n.pt")
    USE_GPU = False
    print("YOLO model loaded on CPU")

# Find all .mp4 videos in the directory AND subdirectories
def find_all_videos(base_dir):
    all_videos = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(".mp4"):
                all_videos.append(os.path.join(root, file))
    return all_videos

# Find all image files in the directory
def find_all_images(base_dir, extensions=('.jpg', '.jpeg', '.png')):
    all_images = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(extensions):
                all_images.append(os.path.join(root, file))
    return all_images

# Get all videos
video_files = find_all_videos(VIDEO_DIR)
print(f"Found {len(video_files)} videos in {VIDEO_DIR} and its subdirectories")

# Get external frames if enabled
external_frames = []
if PROCESS_EXTERNAL_FRAMES:
    external_frames = find_all_images(EXTERNAL_FRAMES_DIR)
    print(f"Found {len(external_frames)} frames in {EXTERNAL_FRAMES_DIR}")
else:
    print("External frame processing is disabled")

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
    
    # Try to extract date from AgentDVR grab files (e.g., 2025-04-03_16-01-53_128.jpg)
    match = re.search(r'(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})(?:_\d+)?', base_name)
    if match:
        date_str = match.group(1)
        time_str = match.group(2)
        try:
            # Parse the date and time strings
            year, month, day = date_str.split('-')
            hour, minute, second = time_str.split('-')
            
            # Format the date as requested
            formatted_date = f"{int(day):02d}-{int(month):02d}-{int(year)}---{int(hour):02d}-{int(minute):02d}-{int(second):02d}"
            date_only = f"{int(day):02d}-{int(month):02d}-{int(year)}"
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
    print(f"   ☑️ Extracting frames at {fps} fps...")
    
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
def detect_humans(image_path, video_name, formatted_date, date_only, source_type="video"):
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
    
    # If a person is detected, copy the frame to the appropriate person detection folder
    if person_detected:
        # Determine output directory based on source type
        if source_type == "external":
            # External frames go to CCAM1 folder
            output_dir = CCAM1_OUTPUT_DIR
        else:
            # Video frames go to date-based folder within PERSON_FRAMES_DIR
            date_person_dir = os.path.join(PERSON_FRAMES_DIR, date_only)
            output_dir = date_person_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename for the detected frame
        frame_filename = os.path.basename(image_path)
        timestamp = datetime.datetime.now().strftime("%H-%M-%S-%f")[:10]
        
        if source_type == "external":
            # For external frames, include the original filename in the output
            person_frame_name = f"{formatted_date}_{os.path.splitext(frame_filename)[0]}_{timestamp}.jpg"
        else:
            # For video frames, include the video name
            person_frame_name = f"{formatted_date}_{video_name}_{frame_filename}_{timestamp}.jpg"
            
        person_frame_path = os.path.join(output_dir, person_frame_name)
        
        # Create a copy of the original image
        original_image = image.copy()
        
        # Get annotated image with bounding boxes
        annotated_image = results[0].plot()  # This draws bounding boxes on the image
        
        # Apply opacity to bounding boxes by blending original and annotated images
        final_image = cv2.addWeighted(annotated_image, DETECTION_OPACITY, original_image, 1 - DETECTION_OPACITY, 0)
        
        # Save the image with semi-transparent bounding boxes
        cv2.imwrite(person_frame_path, final_image)
    
    return person_detected

# Function to create clips based on frame analysis
def create_clips_from_frames(video_path, frames_dir, video_name):
    print("   ☑️ Analyzing frames for people...")
    frames = sorted(glob(os.path.join(frames_dir, '*.jpg')))
    
    if not frames:
        print("   ⚠️ No frames found to analyze")
        return 0, None
    
    # Variables to track clip segments
    clip_segments = []
    in_segment = False
    start_frame = None
    
    # Extract date from filename for output organization
    formatted_date, date_only = extract_date_from_filename(video_path)
    
    # Create date-based output directory for detected person frames
    date_output_dir = None
    if CREATE_CLIPS:
        date_output_dir = os.path.join(OUTPUT_DIR, date_only)
        os.makedirs(date_output_dir, exist_ok=True)
    
    # Frame rate used during extraction (for calculating timestamps)
    extraction_fps = EXTRACT_FRAME_RATE
    
    # Process each frame to find segments with people
    for i, frame_path in enumerate(tqdm(frames, desc="Analyzing frames")):
        has_person = detect_humans(frame_path, video_name, formatted_date, date_only, source_type="video")
        
        # Skip clip segment tracking if clip creation is disabled
        if not CREATE_CLIPS:
            continue
            
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
    if CREATE_CLIPS and in_segment:
        start_time = start_frame / extraction_fps
        end_time = len(frames) / extraction_fps
        clip_segments.append((start_time, end_time))
    
    # Create clips from identified segments if enabled
    created_clips = 0
    if CREATE_CLIPS and clip_segments:
        print(f"   ☑️ Creating {len(clip_segments)} clips...")
        
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
    elif not CREATE_CLIPS:
        print("   ℹ️ Clip creation is disabled")
    
    return created_clips, date_only

# Initialize processed frames set (replace existing initialization)
if os.path.exists(PROCESSED_FRAMES_LOG):
    with open(PROCESSED_FRAMES_LOG, "r") as f:
        processed_frames = {line.strip() for line in f if line.strip()}
else:
    processed_frames = set()

# Modified process_external_frames function
def process_external_frames():
    if not PROCESS_EXTERNAL_FRAMES:
        print("\nℹ️ External frame processing is disabled")
        return 0, 0
        
    print("\n🔍 PROCESSING EXTERNAL FRAMES")
    processed_count = 0
    person_detected_count = 0
    
    for frame_path in tqdm(external_frames, desc="Processing external frames"):
        frame_key = os.path.basename(frame_path)  # Or use full path if needed
        
        # Skip if already processed
        if frame_key in processed_frames:
            continue
            
        try:
            # Extract date from filename
            formatted_date, date_only = extract_date_from_filename(frame_path)
            
            # Detect humans in the frame
            has_person = detect_humans(frame_path, "CCAM1", formatted_date, date_only, source_type="external")
            
            if has_person:
                person_detected_count += 1
                
            # Mark as processed
            processed_frames.add(frame_key)
            with open(PROCESSED_FRAMES_LOG, "a") as f:
                f.write(frame_key + "\n")
                
            processed_count += 1
            
        except Exception as e:
            print(f"⚠️ Error processing frame {frame_path}: {str(e)}")
            continue
            
    print(f"✅ Processed {processed_count} new frames, found people in {person_detected_count} frames")
    return processed_count, person_detected_count

# Process each video
for video_path in video_files:
    # Check if we've already processed this video
    if video_path in processed_videos:
        print(f"👁️ ALREADY PROCESSED: {video_path}")
        continue

    print(f"🎬 PROCESSING: {video_path}")
    start_time = time.time()
    
    try:
        # Step 1: Extract frames using ffmpeg
        video_frames_dir, video_name = extract_frames(video_path, FRAMES_DIR)
        
        if video_frames_dir and video_name:
            # Step 2: Analyze frames and create clips (if enabled)
            clips_count, date_only = create_clips_from_frames(video_path, video_frames_dir, video_name)
            
            # Count how many person frames were detected
            person_frames_dir = os.path.join(PERSON_FRAMES_DIR, date_only)
            if os.path.exists(person_frames_dir):
                # Count only frames related to this video
                person_frames_count = len([f for f in os.listdir(person_frames_dir) if video_name in f])
                print(f"   📊 Saved {person_frames_count} frames with people to {person_frames_dir}")
            
            # Step 3: Clean up extracted frames to save space
            print("   🧹 Cleaning up extracted frames...")
            shutil.rmtree(video_frames_dir)
            
            # Log processing time
            process_time = time.time() - start_time
            clip_status = f"Created {clips_count} clips." if CREATE_CLIPS else "Clip creation disabled."
            print(f"✅ {video_path} processed in {process_time:.1f} seconds. {clip_status}")
            
            # Add video to processed log
            with open(LOG_FILE, "a") as f:
                f.write(video_path + "\n")
                
        else:
            print(f"❌ Skipping {video_path} due to frame extraction failure")
            
    except Exception as e:
        print(f"❌ Error processing video {video_path}: {str(e)}")

# Process external frames from the specified folder
process_external_frames()

print("🎉 All videos and external frames processed!")