from pathlib import Path

from typing import Union
import cv2
import tyro


def main(video_path: str, video_capture_device: Union[str, int] = 0):
    """
    Capture video with the camera connected to your computer. Press `q` to end the recording.

    Args:
        video_path: The file path for the output video in .mp4 format.
        video_capture_device: the device id for your camera connected to the computer in OpenCV format.

    """
    cap = cv2.VideoCapture(video_capture_device)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    path = Path(video_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))

    while True:
        ret, frame = cap.read()
        writer.write(frame)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    print('Recording finished')
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    tyro.cli(main)
