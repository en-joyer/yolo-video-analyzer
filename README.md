# YOLO Powered CCTV Footage Analysis with Person Detection

Instead of sifting through hours of CCTV footage in our apartment, I developed this system and wanted to share it. This script automatically extracts frames from your video files, analyzes each frame for people using the YOLO (You Only Look Once) AI model, and saves frames where individuals are detected into a separate folder organized by date. Optionally, it can also create short video clips of the moments when people are present.

## Features

* **Automatic Frame Extraction:** Automatically extracts frames at a specified rate (default is 1 frame per second) from all `.mp4` files found in the designated video directory (and its subdirectories).
* **YOLO Based Person Detection:** Each extracted frame is analyzed for human figures using the highly accurate YOLOv8 model.
* **Date-Based Organization:** Frames containing detected people are neatly organized into separate folders within the `detected_person` directory, named according to the date extracted from the video or external frame filename.
* **External Frame Analysis:** In addition to video files, it can also analyze image files (`.jpg`, `.jpeg`, `.png`) located in a specified external frames directory and save those with detected people to a dedicated output folder.
* **Video Clip Creation (Optional):** Offers the ability to generate short video clips from segments where people are continuously detected.
* **GPU Support (Optional):** If your system has a compatible NVIDIA graphics card, you can enable GPU acceleration for YOLO analysis, significantly speeding up the process.
* **Configurable Settings:** Various parameters such as frame extraction rate, detection confidence threshold, and bounding box opacity can be easily configured via the `.env` file.
* **Logging:** Keeps track of processed videos and external frames in log files, making it easy to monitor which files have been analyzed.

## Installation

1.  **Install Required Libraries:**
    ```
    pip install opencv-python ultralytics tqdm python-dotenv
    ```

2.  **Configure the `.env` File:** Create a `.env` file in the project directory and edit the necessary directories and settings as follows:

    ```env
    VIDEO_DIR="./videos"             # Directory containing .mp4 files
    FRAMES_DIR="./extracted_frames"   # Directory to save extracted frames
    EXTERNAL_FRAMES_DIR="./external_frames" # Directory containing external frames to analyze
    PERSON_FRAMES_DIR="./detected_person" # Base directory for frames with people detected
    EXTERNAL_OUTPUT_DIR="./detected_person/external" # Specific output directory for detected external frames
    PROCESSED_FRAMES_LOG="./has_processed_external_frames.txt" # Log file for processed external frames
    OUTPUT_DIR="./yolo_clips"         # Base output directory for created video clips
    LOG_FILE="./processed_videos.txt"   # Log file to track processed videos
    LOG_FILE_FRAMES="./processed_frames.txt" # Log file to track processed external frames

    CONFIDENCE_THRESHOLD=0.5        # Minimum confidence for YOLO detections (0.0-1.0)
    EXTRACT_FRAME_RATE=1            # Number of frames to extract per second
    DETECTION_OPACITY=0.5           # Opacity of the detection bounding boxes (0.0-1.0)
    CREATE_CLIPS=True               # Enable/disable the creation of video clips (True/False)
    PROCESS_EXTERNAL_FRAMES=True    # Enable/disable the processing of external frames (True/False)
    USE_GPU=False                   # Enable/disable GPU usage (True/False)
    ```

    Adjust the paths and settings in this file according to your specific needs.

3.  **Create Video and External Frame Directories (Optional):** If the `VIDEO_DIR` and `EXTERNAL_FRAMES_DIR` directories specified in your `.env` file do not already exist, you might need to create them manually.

## Usage

1.  **Place Video Files:** Copy the `.mp4` video files you want to analyze into the `VIDEO_DIR` folder (or its subfolders) specified in your `.env` file.
2.  **Place External Frames (Optional):** If you have external image files to analyze, copy them into the `EXTERNAL_FRAMES_DIR` folder (or its subfolders) specified in your `.env` file.
3.  **Run the Script:** Open a terminal in the project directory and execute the following command:

    ```
    python process_videos.py
    ```

## Results

Once the script runs successfully:

* Extracted frames from the video files will be saved in the `extracted_frames` directory, organized into subfolders named after the video files.
* Frames where people are detected will be saved in the `detected_person` directory, within subfolders named according to the date (`day-month-year` format) extracted from the video or external frame filename. External frames with detected people will be saved in the `detected_person/external` folder.
* If `CREATE_CLIPS` is set to `True`, short video clips of the time segments where people are detected will be created in the `yolo_clips` directory, organized into subfolders by date.
* Information about the processed videos and external frames will be recorded in the `processed_videos.txt` and `has_processed_external_frames.txt` files, respectively.

This system allows you to focus only on the moments with human activity in your security camera footage, saving you valuable time and making the review process much more efficient compared to manually watching hours of recordings. I hope this tool proves useful for you.
