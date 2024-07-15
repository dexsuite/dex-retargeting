import pickle
from pathlib import Path

import cv2
import tqdm
import tyro

from dex_retargeting.constants import HandType
from example.vector_retargeting.single_hand_detector import SingleHandDetector


def detect_video(video_path: str, output_path: str, is_right: bool):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
    else:
        data = []
        detector = SingleHandDetector(hand_type="Right", selfie=False)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm.tqdm(total=length) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                rgb = frame[..., ::-1]
                _, joint_pos, _, _ = detector.detect(rgb)
                data.append(joint_pos)
                pbar.update(1)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            pickle.dump(data, f)

        cap.release()
        cv2.destroyAllWindows()


def main(
    video_path: str,
    output_path: str,
    hand_type: HandType = HandType.right,
):
    """
    Detects the human hand pose from a video and translates the human pose trajectory into a robot pose trajectory.

    Args:
        video_path: The file path for the input video in .mp4 format.
        output_path: The file path for the output data in .pickle format.
        hand_type: Specifies which hand is being tracked, either left or right.
            Please note that retargeting is specific to the same type of hand: a left robot hand can only be retargeted
            to another left robot hand, and the same applies for the right hand.
    """

    detect_video(video_path, output_path, is_right=HandType.right == hand_type)


if __name__ == "__main__":
    tyro.cli(main)
