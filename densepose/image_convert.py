import cv2
from utils.helper import GetLogger, Predictor
from argparse import ArgumentParser
import os


parser = ArgumentParser()
parser.add_argument(
    "--input", type=str, help="Set the input path to the image", required=True
)
parser.add_argument(
    "--out", type=str, help="Set the output path to the image", default="output_densepose.jpg"
)
parser.add_argument(
    "--width", type=int, help="Output image width (optional)", default=None
)
parser.add_argument(
    "--height", type=int, help="Output image height (optional)", default=None
)
args = parser.parse_args()


logger = GetLogger.logger(__name__)
predictor = Predictor()

# Read input image
image_path = args.input

if not os.path.exists(image_path):
    logger.error(f"Error: Image file '{image_path}' not found!")
    exit(1)

logger.info(f"Reading image: {image_path}")
frame = cv2.imread(image_path)

if frame is None:
    logger.error(f"Error: Unable to read image '{image_path}'")
    exit(1)

# Get image dimensions
height, width = frame.shape[:2]
logger.info(f"Image dimensions: {width}x{height}")

# Process the image with DensePose
logger.info("Processing image with DensePose...")
out_frame, out_frame_seg = predictor.predict(frame)

# Resize output if width/height specified
if args.width or args.height:
    original_h, original_w = out_frame_seg.shape[:2]
    
    if args.width and args.height:
        # Both dimensions specified - resize to exact size
        new_width = args.width
        new_height = args.height
        logger.info(f"Resizing output to {new_width}x{new_height}")
    elif args.width:
        # Only width specified - maintain aspect ratio
        new_width = args.width
        new_height = int(original_h * (new_width / original_w))
        logger.info(f"Resizing output to width {new_width} (height {new_height} to maintain aspect ratio)")
    else:
        # Only height specified - maintain aspect ratio
        new_height = args.height
        new_width = int(original_w * (new_height / original_h))
        logger.info(f"Resizing output to height {new_height} (width {new_width} to maintain aspect ratio)")
    
    out_frame_seg = cv2.resize(out_frame_seg, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

# Save output image
output_path = args.out
final_h, final_w = out_frame_seg.shape[:2]
logger.info(f"Final output dimensions: {final_w}x{final_h}")
logger.info(f"Saving output to: {output_path}")

# Save the segmented output
success = cv2.imwrite(output_path, out_frame_seg)

if success:
    logger.info(f"✓ DensePose image saved successfully: {output_path}")
    
    # Get output file size
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # Convert to MB
    logger.info(f"Output file size: {file_size:.2f} MB")
else:
    logger.error(f"Error: Failed to save output image")
    exit(1)

logger.info("Done!")
