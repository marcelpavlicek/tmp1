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

    # Initialize counters and tracking history
    LINE_X = 240
    entered_count = 0
    left_count = 0
    track_history = {}

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

            # Process tracks for line crossing
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                # Keep track of IDs seen in this frame
                current_frame_ids = set(track_ids)

                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    center_x = float(x)

                    if track_id in track_history:
                        prev_x = track_history[track_id]

                        # Moved from right side (entered) to left side (entered)
                        # Right side: x > 240, Left side: x <= 240
                        if prev_x > LINE_X and center_x <= LINE_X:
                            entered_count += 1

                        # Moved from left side (left) to right side (left)
                        # Left side: x <= 240, Right side: x > 240
                        elif prev_x <= LINE_X and center_x > LINE_X:
                            left_count += 1

                    # Update history
                    track_history[track_id] = center_x

                # Optional: Clean up track_history for IDs that are gone
                # Simple approach: remove IDs not in current frame
                # Note: This might lose history if tracking flickers for 1 frame.
                # A better approach would use a 'last_seen' timestamp, but simple cleanup
                # helps prevent indefinite growth if memory is a concern.
                # However, for robustness against flickering, we might want to keep them a bit longer.
                # Given the simplicity of the request, we can just leave it or do a simple prune.
                # Let's prune keys that are not in current_frame_ids to prevent memory leak.

                keys_to_remove = [k for k in track_history if k not in current_frame_ids]
                for k in keys_to_remove:
                   del track_history[k]

            # Draw the vertical line
            cv2.line(annotated_frame, (LINE_X, 0), (LINE_X, height), (0, 0, 255), 2)

            # Draw the counters
            cv2.putText(annotated_frame, f"Entered: {entered_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Left: {left_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
