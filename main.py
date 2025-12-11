import cv2
from ultralytics import YOLO

def main():
    # Load the YOLO model
    # 'yolov8n.pt' is the smallest model, good for testing.
    # It will download automatically if not present.
    model = YOLO('yolov8n.pt')

    # Input video path
    video_path = 'people-detection.mp4'
    output_path = 'output.mp4'

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    # mp4v is a common codec for mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing video: {video_path}")
    print(f"Saving output to: {output_path}")

    frame_count = 0
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1
            if frame_count % 20 == 0:
                print(f"Processing frame {frame_count}...")

            # Run YOLO tracking on the frame, persisting tracks between frames
            # classes=0 filters for 'person' class (index 0 in COCO)
            results = model.track(frame, persist=True, classes=0, verbose=False)

            # Visualize the results on the frame
            # results[0].plot() creates an annotated frame
            annotated_frame = results[0].plot()

            # Write the annotated frame
            out.write(annotated_frame)

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        # Release everything
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Done processing.")

if __name__ == "__main__":
    main()
