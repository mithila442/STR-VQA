import os
import cv2
import numpy as np

def extract_frame(videos_dir, video_name, save_folder, dataset_type="saliency"):
    """
    Extract frames from a video and resize them to 224×398.

    Parameters:
    - videos_dir: str, path to the directory containing video files.
    - video_name: str, name of the video file (with extension).
    - save_folder: str, directory where extracted frames will be saved.
    - dataset_type: str, either "saliency" (60 frames) or "vqa" (8 frames).

    Returns:
    - Saves extracted frames to `save_folder`.
    """
    filename = os.path.join(videos_dir, video_name)

    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        print(f"Error: Couldn't open video file {filename}")
        return

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define number of frames based on dataset type
    target_frames = 60 if dataset_type == "saliency" else 8
    target_frames = min(target_frames, video_length)

    # Select frames using linspace
    selected_indices = np.linspace(0, video_length - 1, target_frames, dtype=int)

    os.makedirs(save_folder, exist_ok=True)

    last_valid_frame = None

    for i, frame_to_capture in enumerate(selected_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_capture)
        ret, frame = cap.read()

        # If frame cannot be read, try the next available frame
        while not ret or frame is None:
            frame_to_capture += 1  # Move to the next frame
            if frame_to_capture >= video_length:  # If at end of video, break
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_capture)
            ret, frame = cap.read()


        if not ret or frame is None:
            print(f"Warning: Couldn't capture valid frame for index {i} in {video_name}, using last valid frame.")
            frame = last_valid_frame  # Use last valid frame if available

        if frame is not None:
            # Resize the frame to **exactly** 224x398
            resized_frame = cv2.resize(frame, (398, 224), interpolation=cv2.INTER_LINEAR)

            # Save frame
            frame_filename = os.path.join(save_folder, f"{i:03d}.png")
            cv2.imwrite(frame_filename, resized_frame)

            # Store last valid frame for fallback use
            last_valid_frame = resized_frame

        # Resize the frame to **exactly** 224x398
        resized_frame = cv2.resize(frame, (398, 224), interpolation=cv2.INTER_LINEAR)

        frame_filename = os.path.join(save_folder, f"{i:03d}.png")
        cv2.imwrite(frame_filename, resized_frame)

    cap.release()

def process_all_videos(videos_dir, output_frames_dir, dataset_type="vqa"):
    """
    Process all videos in `videos_dir`, extracting frames to `output_frames_dir`.
    """
    for video_name in os.listdir(videos_dir):
        if video_name.endswith(".mp4"):
            video_id = video_name.split('.')[0]
            save_path = os.path.join(output_frames_dir, video_id)
            extract_frame(videos_dir, video_name, save_folder=save_path, dataset_type=dataset_type)

if __name__ == "__main__":
    raw_videos_path = "./KoNViD_1k/KoNViD_1k_videos"
    extracted_frames_path = "./KoNViD_1k_extracted_frames"
    process_all_videos(raw_videos_path, extracted_frames_path, dataset_type="vqa")
