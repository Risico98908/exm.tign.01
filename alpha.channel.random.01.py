import cv2
import numpy as np

def set_alpha_randomly(image_path, output_path, alpha_value=0):
    """
    Set randomly 2 out of every 3 pixels' alpha channel to a specific value.

    :param image_path: Path to the input image.
    :param output_path: Path to save the output image with modified alpha channel.
    :param alpha_value: The alpha value to set (e.g., 0 for fully transparent).
    """
    # Load the image (with or without alpha channel)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Check if the image has an alpha channel, if not, add one
    if image.shape[2] == 3:
        # Add an alpha channel to the image
        alpha_channel = np.ones((image.shape[0], image.shape[1]), dtype=image.dtype) * 255
        image = np.dstack((image, alpha_channel))

    elif image.shape[2] != 4:
        raise ValueError("Unexpected number of channels in the input image")

    # Split the channels
    b, g, r, a = cv2.split(image)

    # Create a mask to select pixels
    mask = np.random.choice([0, 1], size=a.shape, p=[1/3, 2/3])

    # Set the alpha channel to the specified value where the mask is 1
    a[mask == 1] = alpha_value

    # Merge the channels back
    modified_image = cv2.merge([b, g, r, a])

    # Save the output image
    cv2.imwrite(output_path, modified_image)

    print(f"Image saved to: {output_path}")

# Example usage
input_image_path = '/media/backup_005_6/workspace.video.etc/zz/zz.01.boobs/zz.01.boobs.coll.01/reencoded/concatenated_transcoded_video/beach/frames/frame_001.png'
output_image_path = '/media/backup_005_6/workspace.video.etc/zz/zz.01.boobs/zz.01.boobs.coll.01/reencoded/concatenated_transcoded_video/beach/frames/frame_001_alpha_modified.png'
set_alpha_randomly(input_image_path, output_image_path, alpha_value=0)
