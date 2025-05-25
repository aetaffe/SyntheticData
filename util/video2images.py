import cv2
from pathlib import Path
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts videos to images.")
    parser.add_argument("video_path", type=str, help="Path to the video file.")
    parser.add_argument("output_dir", type=str, help="Directory to save the extracted images.")
    args = parser.parse_args()

    video_dir = Path(args.video_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
    if not videos:
        print("No video files found in the specified directory.")
        exit(1)
    print(f'Found {len(videos)} video files in {video_dir}.')

    frame_count = 0
    for video in videos:
        print(f"Processing video: {video.name}")
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"Error opening video file {video.name}.")
            continue

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            print(f"Extracting frame {frame_count} from {video.name}")
            image_path = output_dir / f"{video.stem}_frame{frame_count:04d}.jpg"
            cv2.imwrite(str(image_path), frame)

        cap.release()
    print(f"Total frames extracted: {frame_count}")