
from PIL import Image
import os

brain_frames = os.listdir('frames')
cochlea_frames = os.listdir('cochlea_frames')

for i in range(len(brain_frames)):
    # Open the images
    print(f'cochlea_frames/{cochlea_frames[i]}')
    image1 = Image.open(f'cochlea_frames/{cochlea_frames[i]}')  # 720x720 pixels
    image2 = Image.open(f'frames/{brain_frames[i]}')  # 350x750 pixels

    # Assuming both images are the same height now
    height = max(image1.height, image2.height)

    # Create a new image with the combined width and the original height
    new_image = Image.new('RGB', (image1.width + image2.width, height))

    # Paste image1 on the left
    new_image.paste(image1, (0, 0))

    # Paste image2 on the right, starting at the width of image1
    new_image.paste(image2, (image1.width, 0))

    # Save the new image
    new_image.save(f'combined/combined_{i:04d}.jpg')