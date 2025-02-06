import os
from PIL import Image
import SimpleITK as sitk
import time

def split_gif_to_frames(gif_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with Image.open(gif_path) as gif:
        frame_count = gif.n_frames
        print(f"Total frames: {frame_count}")

        for frame in range(frame_count):
            gif.seek(frame)
            frame_filename = os.path.join(output_folder, f"frame_{frame:03d}.png")
            gif.save(frame_filename, format="PNG")
            # print(f"Saved: {frame_filename}")
    
    n_frames = len(os.listdir(output_folder))
    print(f"Saved {n_frames} frames")

def stack_png_to_mha(input_folder, output_mha_path):
    # Get list of PNG files in the folder, sorted by filename (assuming sequential naming)
    png_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])

    if not png_files:
        raise ValueError("No PNG files found in the provided directory")

    # Read the first PNG file to determine the size and spacing of the slices
    first_png_path = os.path.join(input_folder, png_files[0])
    first_image = sitk.ReadImage(first_png_path, outputPixelType=sitk.sitkVectorUInt8)
    
    slice_size = first_image.GetSize()  # (width, height, depth)
    slice_components = first_image.GetNumberOfComponentsPerPixel() # Checking if they have 3 channels.
    print("slice size:", slice_size, ", num_components:", slice_components)

    # Create an empty list to hold the images
    image_list = []

    # Load each PNG file and append the images to the list
    for i, png_file in enumerate(png_files):
        png_path = os.path.join(input_folder, png_file)
        img = sitk.ReadImage(png_path, sitk.sitkVectorUInt8)
        if img.GetNumberOfComponentsPerPixel() == 3:
            image_list.append(img)

    # Stack the images along the third dimension (Z-axis)
    print("number of frames in stack:", len(image_list))
    stacked_image = sitk.JoinSeries(image_list)
    stack_size = stacked_image.GetSize()
    print("stack size:", stack_size, ", stack n_components:", stacked_image.GetNumberOfComponentsPerPixel())

    # Save the stacked image as a single 3D MHA file
    sitk.WriteImage(stacked_image, output_mha_path)
    print(f"Stacked MHA saved as {output_mha_path}")

if __name__ == "__main__":
    ## Input directory (including subfolder for attention GIFs).
    ATTENTION_PARENT_DIR = f"/data/bodyct/experiments/lung-malignancy-fairness-shaurya/nlst/sybil_attentions"
    ATTENTION_GIF_DIR = f"{ATTENTION_PARENT_DIR}/attention_gifs"
    subfolders = [f[:-4] for f in os.listdir(ATTENTION_GIF_DIR_DIR)]

    ## Make subdirectories for output filetypes.
    ATTENTION_PNG_DIR = f"{ATTENTION_PARENT_DIR}/attention_pngs"
    os.makedirs(ATTENTION_PNG_DIR, exist_ok=True)
    ATTENTION_MHA_DIR = f"{ATTENTION_PARENT_DIR}/attention_mhas"
    os.makedirs(ATTENTION_MHA_DIR, exist_ok=True)

    print("Starting conversion!")
    start_time_conv = time.time()
    
    for i in range(0, len(subfolders)):
        seriesid = subfolders[i]
        print(f"\n{i+1} / {len(subfolders)}: converting {seriesid} ...")
        
        gif_path = f"{ATTENTION_GIF_DIR}/{seriesid}.gif"
        png_folder = f"{ATTENTION_PNG_DIR}/{seriesid}"
        mha_path = f"{ATTENTION_MHA_DIR}/{seriesid}.mha"

        split_gif_to_frames(gif_path, png_folder)       
        stack_png_to_mha(png_folder, output_mha_path)
    
    end_time_conv = time.time()
    print(f"Total time for conversion: {end_time_conv - start_time_conv} seconds")