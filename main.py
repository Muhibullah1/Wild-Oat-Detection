def process_images(image_paths, mask_paths, line_paths, width_paths, output_base_path, inter_row_NL, inter_row_NL_OL, final_numb, debug_viz_dir,
                  threshold_ratio, min_branches, proximity_threshold):
    modified_dir, overlay_dir, final_numb_dir, debug_viz_dir = create_output_directories(
        output_base_path, inter_row_NL, inter_row_NL_OL, final_numb, debug_viz_dir)

    for img_path, mask_path, line_path, width_path in zip(image_paths, mask_paths, line_paths, width_paths):
        original_image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        total_pixels = mask.size
        crop_pixels = np.sum(mask == 1)
        crop_percentage = crop_pixels / total_pixels

        if os.path.exists(line_path) and crop_percentage < 0.30:
            print(f"Processing {os.path.basename(img_path)}...")

            wild_oat_mask, debug_image = process_lines_from_file(line_path, mask, width_path, threshold_ratio, min_branches, proximity_threshold)

            filtered_path = os.path.join(modified_dir, os.path.basename(img_path))
            cv2.imwrite(filtered_path, wild_oat_mask * 255)

            overlay_image = overlay_images(original_image, wild_oat_mask)
            overlay_path = os.path.join(overlay_dir, os.path.basename(img_path))
            cv2.imwrite(overlay_path, overlay_image)

            debug_overlay = overlay_debug_image(original_image, debug_image, alpha=0.6)
            debug_path = os.path.join(debug_viz_dir, os.path.basename(img_path))
            cv2.imwrite(debug_path, debug_overlay)
            
            final_numb_path = os.path.join(final_numb_dir, os.path.basename(img_path))
            cv2.imwrite(final_numb_path, debug_image)

            print(f"Saved outputs for {os.path.basename(img_path)}")
        else:
            print(f"Skipping {os.path.basename(img_path)} - No row data or crop too dense")



def overlay_images(original, mask, color=(0, 255, 0)):
    """Overlay the mask on the original image with specified color."""
    overlay = original.copy()
    mask_3ch = np.zeros_like(original)
    mask_3ch[mask > 0] = color
    alpha = 0.5    # Blend the images
    cv2.addWeighted(mask_3ch, alpha, overlay, 1 - alpha, 0, overlay)
    return overlay

def overlay_debug_image(original, debug_image, alpha=0.6):
    """Overlay the debug visualization image on the original image.
    Args:original: Original RGB image
        debug_image: Debug visualization RGB image with colored regions
        alpha: Transparency value for overlay (0-1)   
    Returns: Overlaid image"""
    # Ensure both images are RGB
    if len(original.shape) == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    
    overlay = original.copy()
    # Find non-black pixels in debug image (ignore black background)
    non_black_mask = np.any(debug_image > 20, axis=2)
    
    # Apply overlay only to non-black pixels
    overlay[non_black_mask] = cv2.addWeighted(debug_image[non_black_mask].reshape(-1, 1, 3), alpha,
        original[non_black_mask].reshape(-1, 1, 3), 1 - alpha, 0).reshape(-1, 3)
    
    return overlay


def create_output_directories(base_path, inter_row_NL, inter_row_NL_OL, final_numbered, debug_dir="debug_viz"):
    modified_dir = os.path.join(base_path, inter_row_NL)
    overlay_dir = os.path.join(base_path, inter_row_NL_OL)
    final_numb_dir = os.path.join(base_path, final_numbered)
    debug_viz_dir = os.path.join(base_path, debug_dir)

    os.makedirs(modified_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(final_numb_dir, exist_ok=True)
    os.makedirs(debug_viz_dir, exist_ok=True)

    return modified_dir, overlay_dir, final_numb_dir, debug_viz_dir
