# Geometry Aware Post Procsssing

import os
import cv2
import ast
import glob
import time
import numpy as np
from tqdm import tqdm
from skimage.morphology import skeletonize
from scipy.ndimage import convolve
from skimage.measure import label


def simple_wild_oat_detection(segmentation_mask, row_mask, threshold_ratio=0.45, 
                             min_branches=2, proximity_threshold=3, small_segment_proximity=10):
    """
    Complete rewrite of wild oat detection with correct morphological analysis.
    Logic:
    1. Clear inter-row segments → standard branch analysis
    2. Clear within-row segments → crops  
    3. Bridge segments (touching both regions) → morphological analysis:
       - Find narrowest width at row edge (connection)
       - Find maximum width in inter-row region (expansion)
       - If expansion > 1.3 * connection AND branches > 3 → wild oat
    """
    
    wild_oat_mask = np.zeros_like(segmentation_mask)     # Initialize output masks
    debug_image = cv2.cvtColor(segmentation_mask * 255, cv2.COLOR_GRAY2BGR)
    
    # Find all connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(segmentation_mask, 8)
    
    # Classification buffers
    clear_interrow_candidates = []
    clear_withinrow_candidates = []
    bridge_candidates = []  # Complex cases needing morphological analysis
    segment_lengths = []
    
    # First pass: Classify all segments
    for label in range(1, num_labels):
        component = (labels == label)
        total_pixels = np.sum(component)
        
        if total_pixels < 50:  # Skip tiny components
            continue
        
        # Calculate region distributions
        row_pixels = np.logical_and(component, row_mask).sum()
        interrow_pixels = np.logical_and(component, ~row_mask).sum()
        
        row_ratio = row_pixels / total_pixels
        interrow_ratio = interrow_pixels / total_pixels
        
        # Calculate segment length for small segment analysis
        skeleton = skeletonize(component)
        segment_length = np.sum(skeleton)
        segment_lengths.append(segment_length)
        
        # Classification logic
        if row_ratio <= threshold_ratio:
            # Clear inter-row segments (≤45% in row)
            clear_interrow_candidates.append(label)
            
        elif interrow_ratio < 0.2:
            # Clear within-row segments (<20% in inter-row)
            clear_withinrow_candidates.append(label)
            debug_image[component] = [0, 0, 255]  # Red = crop
            
        else:
            # Bridge candidates (significant presence in both regions)
            bridge_candidates.append(label)
            debug_image[component] = [100, 100, 100]  # Gray = needs analysis
    
    # Set length threshold for small segments
    mean_length = np.mean(segment_lengths) if segment_lengths else 0
    length_threshold = 0.09 * mean_length
    
    print(f"Classification: {len(clear_interrow_candidates)} inter-row, {len(clear_withinrow_candidates)} within-row, {len(bridge_candidates)} bridge candidates")
    
    # Process bridge candidates with morphological analysis
    cut_results = analyze_bridge_segments_morphology(
        bridge_candidates, labels, row_mask, centroids, debug_image)
    
    # Add detected wild oats from bridge analysis
    for result in cut_results:
        wild_oat_mask[result['mask']] = 1
        debug_image[result['mask']] = [128, 255, 0]  # Light green = morphological wild oat
        
        cy, cx = result['centroid']
        text = result['info']
        cv2.putText(debug_image, text, (int(cx-30), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 250), 1)
    
    # Process clear inter-row candidates with standard analysis
    wild_oat_labels = []
    small_interrow_segments = []
    
    for label in clear_interrow_candidates:
        component = (labels == label)
        skeleton = skeletonize(component)
        segment_length = np.sum(skeleton)
        cy, cx = centroids[label]
        
        if segment_length < length_threshold:
            # Small segments for potential merging
            small_interrow_segments.append(label)
            text = f"SMALL: l{segment_length:.0f}"
            #cv2.putText(debug_image, text, (int(cx-20), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 
             #          0.4, (255, 255, 0), 1)
            continue
        
        # Standard branch analysis for clear inter-row segments
        branch_count = count_branches_simple(skeleton)
        
        if branch_count >= min_branches:
            wild_oat_mask[component] = 1
            debug_image[component] = [0, 255, 0]  # Green = standard wild oat
            wild_oat_labels.append(label)
            text = f"WILD: b{branch_count}"
            #cv2.putText(debug_image, text, (int(cx-20), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 
                       #0.4, (0, 0, 0), 1)
        else:
            debug_image[component] = [0, 0, 255]  # Red = insufficient branches
            text = f"CROP: b{branch_count}<{min_branches}"
            #cv2.putText(debug_image, text, (int(cx-25), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 
                      # 0.4, (255, 255, 255), 1)
    
    # Handle small segment merging with proximity
    if wild_oat_labels and small_interrow_segments:
        merge_small_segments(small_interrow_segments, wild_oat_labels, labels, centroids, 
                           wild_oat_mask, debug_image, small_segment_proximity)
    
    return wild_oat_mask, debug_image


def analyze_bridge_segments_morphology(bridge_candidates, labels, row_mask, centroids, debug_image):
    """
    Analyze bridge segments using morphological criteria:
    1. Find connection width at row boundary
    2. Find maximum expansion width in inter-row region
    3. Compare expansion vs connection width
    4. Count branches in inter-row region
    """
    cut_results = []
    for label in bridge_candidates:
        component = (labels == label)
        cy, cx = centroids[label]
        
        total_pixels = np.sum(component)
        row_pixels = np.logical_and(component, row_mask).sum()
        interrow_pixels = np.logical_and(component, ~row_mask).sum()
        
        print(f"Analyzing bridge candidate {label}: total={total_pixels}, row={row_pixels}, interrow={interrow_pixels}")
        
        # Try to separate at row boundaries
        separated_parts = separate_at_row_boundaries(component, row_mask)
        
        if not separated_parts:
            text = f"NO_SEPARATION"
            #cv2.putText(debug_image, text, (int(cx-30), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 
             #          0.4, (128, 128, 128), 1)
            debug_image[component] = [0, 0, 255]  # Default to crop
            print(f"  Could not separate component")
            continue
        
        print(f"  Separated into {len(separated_parts)} parts")
        
        # Analyze each separated part
        for part_idx, part_info in enumerate(separated_parts):
            part = part_info['mask']
            part_type = part_info['type']  # 'interrow' or 'row'
            
            if part_type != 'interrow':
                # Row part - mark as crop
                part_coords = np.where(part)
                if len(part_coords[0]) > 0:
                    part_cy = np.mean(part_coords[0])
                    part_cx = np.mean(part_coords[1])
                    text = f"ROW_PART"
                    #cv2.putText(debug_image, text, (int(part_cx-20), int(part_cy)), 
                    #           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
                    debug_image[part] = [0, 0, 255]  # Red
                continue
            
            # This is an inter-row part - analyze morphology
            part_coords = np.where(part)
            if len(part_coords[0]) == 0:
                continue
                
            part_cy = np.mean(part_coords[0])
            part_cx = np.mean(part_coords[1])
            part_pixels = np.sum(part)
            
            if part_pixels < 100:  # Skip tiny parts
                text = f"TINY_PART"
                #cv2.putText(debug_image, text, (int(part_cx-20), int(part_cy)), 
                           #cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
                debug_image[part] = [150, 100, 50]  # Brown
                continue
            
            print(f"    Analyzing inter-row part {part_idx}: {part_pixels} pixels")
            
            # KEY MORPHOLOGICAL ANALYSIS
            morphology_result = analyze_connection_vs_expansion(part, row_mask, component)
            
            # Count branches in inter-row region only
            interrow_only = np.logical_and(part, ~row_mask)
            if np.sum(interrow_only) > 0:
                interrow_skeleton = skeletonize(interrow_only.astype(np.uint8))
                interrow_branches = count_branches_simple(interrow_skeleton)
            else:
                interrow_branches = 0
            
            print(f"      Morphology: {morphology_result}, Branches: {interrow_branches}")
            
            # Decision criteria
            has_expansion = morphology_result['has_expansion']
            expansion_ratio = morphology_result['expansion_ratio']
            
            #-----------------inter row branches = min branches for greenish segments----------
            interrow_branches_tresh = 20
            
            if has_expansion and interrow_branches > interrow_branches_tresh:
                # WILD OAT - expands from connection + sufficient branches
                cut_results.append({
                    'mask': part,
                    'centroid': (part_cy, part_cx),
                    'info': f"WILD: exp{expansion_ratio:.1f}x+b{interrow_branches}"
                })
                print(f"      WILD OAT: expansion {expansion_ratio:.1f}x, {interrow_branches} branches")
                
            elif interrow_branches > interrow_branches_tresh:
                # Very branchy - likely wild oat even without clear expansion
                cut_results.append({
                    'mask': part,
                    'centroid': (part_cy, part_cx),
                    'info': f"WILD: very_branchy_b{interrow_branches}"
                })
                print(f"      WILD OAT: very branchy ({interrow_branches} branches)")
               
            elif has_expansion and interrow_branches <= interrow_branches_tresh:
                # Expands but few branches
                text = f"CUTOUT: exp{expansion_ratio:.1f}x+b{interrow_branches}≤3"
                #cv2.putText(debug_image, text, (int(part_cx-40), int(part_cy)), 
                 #          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)
                debug_image[part] = [150, 100, 50]  # Brown
                print(f"      CUTOUT: expands but few branches")
                
            else:
                # No expansion, few branches - likely cultivated extension
                text = f"CROP_EXT: no_exp+b{interrow_branches}"
                cv2.putText(debug_image, text, (int(part_cx-35), int(part_cy)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
                debug_image[part] = [0, 0, 255]  # Red
                print(f"      CROP EXTENSION: no expansion, {interrow_branches} branches")
    
    return cut_results


def separate_at_row_boundaries(component, row_mask):
    """
    Separate a component at row boundaries into row and inter-row parts.
    Returns list of separated parts with their types.
    """
    # Find row boundary pixels
    row_edges = cv2.Canny((row_mask * 255).astype(np.uint8), 50, 150)
    boundary_region = cv2.dilate(row_edges, np.ones((3,3), np.uint8))
    
    # Find where component crosses boundaries
    component_at_boundary = np.logical_and(component, boundary_region > 0)
    
    if np.sum(component_at_boundary) == 0:
        return None
    
    # Try progressive cutting until separation achieved
    for dilation_size in [3, 5, 7, 9, 11]:
        cutting_mask = cv2.dilate(component_at_boundary.astype(np.uint8), 
                                np.ones((dilation_size, dilation_size), np.uint8)).astype(bool)
        
        # Don't cut away too much of the component
        if np.sum(cutting_mask) > 0.3 * np.sum(component):
            continue
        
        # Apply cutting
        component_cut = component.copy()
        component_cut[cutting_mask] = False
        
        # Label remaining parts
        from skimage.measure import label
        remaining_labels = label(component_cut, connectivity=2)
        num_parts = np.max(remaining_labels)
        
        if num_parts >= 2:
            # Successfully separated - classify each part
            separated_parts = []
            
            for part_label in range(1, num_parts + 1):
                part = (remaining_labels == part_label)
                part_pixels = np.sum(part)
                
                if part_pixels < 30:  # Skip tiny parts
                    continue
                
                # Classify as row or inter-row based on majority
                part_row_pixels = np.logical_and(part, row_mask).sum()
                part_row_ratio = part_row_pixels / part_pixels
                
                part_type = 'row' if part_row_ratio > 0.5 else 'interrow'
                
                separated_parts.append({
                    'mask': part,
                    'type': part_type,
                    'pixels': part_pixels,
                    'row_ratio': part_row_ratio
                })
            
            return separated_parts
    
    return None


def analyze_connection_vs_expansion(interrow_part, row_mask, original_component):
    """
    CORE MORPHOLOGICAL ANALYSIS:
    Compare narrowest connection width vs maximum expansion width.
    
    This is the key insight - wild oats have narrow connections at row edges
    but expand to much wider structures in inter-row regions.
    """
    if np.sum(interrow_part) < 100:
        return {'has_expansion': False, 'expansion_ratio': 1.0}
    
    # Get distance transform (local width/2 at each point)
    dist_transform = cv2.distanceTransform(interrow_part.astype(np.uint8), cv2.DIST_L2, 3)
    
    # Create distance map from row edges
    row_edge_dist = cv2.distanceTransform(1 - row_mask.astype(np.uint8), cv2.DIST_L2, 3)
    
    # Find all points in the inter-row part
    interrow_coords = np.where(interrow_part)
    
    if len(interrow_coords[0]) == 0:
        return {'has_expansion': False, 'expansion_ratio': 1.0}
    
    # Collect width measurements at different distances from row edge
    connection_widths = []  # Near row edge (≤10 pixels)
    expansion_widths = []   # All points in inter-row region
    
    for i in range(len(interrow_coords[0])):
        y, x = interrow_coords[0][i], interrow_coords[1][i]
        distance_from_row = row_edge_dist[y, x]
        width = dist_transform[y, x] * 2  # Local width
        
        # All points contribute to expansion measurement
        expansion_widths.append(width)
        
        # Only points near row edge contribute to connection measurement
        if distance_from_row <= 10:
            connection_widths.append(width)
    
    if len(connection_widths) < 3:
        # No clear connection region found
        return {'has_expansion': False, 'expansion_ratio': 1.0}
    
    # KEY COMPARISON: minimum connection width vs maximum expansion width
    min_connection_width = np.min(connection_widths)
    max_expansion_width = np.max(expansion_widths)
    avg_expansion_width = np.mean(expansion_widths)
    
    # Calculate expansion ratio (how much wider it gets)
    expansion_ratio = max_expansion_width / min_connection_width if min_connection_width > 0 else 1.0
    avg_expansion_ratio = avg_expansion_width / np.mean(connection_widths) if connection_widths else 1.0
    
    # Wild oat criteria: significant expansion from narrow connection
    # Use both max and average criteria to be robust
    has_expansion = (expansion_ratio > 1.5 or avg_expansion_ratio > 1.3) and min_connection_width < 10
    
    print(f"        Connection analysis:")
    print(f"          Min connection: {min_connection_width:.1f}, Max expansion: {max_expansion_width:.1f}")
    print(f"          Expansion ratio: {expansion_ratio:.1f}, Avg ratio: {avg_expansion_ratio:.1f}")
    print(f"          Has expansion: {has_expansion}")
    
    return {
        'has_expansion': has_expansion,
        'expansion_ratio': max(expansion_ratio, avg_expansion_ratio),
        'min_connection': min_connection_width,
        'max_expansion': max_expansion_width
    }


def merge_small_segments(small_segments, wild_oat_labels, labels, centroids, wild_oat_mask, debug_image, proximity_threshold):
    """Handle merging of small inter-row segments with nearby wild oats."""
    # Create template of existing wild oats
    wild_oat_template = np.zeros_like(wild_oat_mask)
    for label in wild_oat_labels:
        wild_oat_template[labels == label] = 1
    
    # Distance transform from wild oats
    wild_oat_dist = cv2.distanceTransform(1 - wild_oat_template, cv2.DIST_L2, 3)
    
    for label in small_segments:
        component = (labels == label)
        min_dist = np.min(wild_oat_dist[component]) if np.any(component) else float('inf')
        
        cy, cx = centroids[label]
        
        if min_dist <= proximity_threshold:
            wild_oat_mask[component] = 1
            debug_image[component] = [0, 255, 255]  # Cyan = merged
            text = f"MERGED: d{min_dist:.1f}"
            cv2.putText(debug_image, text, (int(cx-25), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, (0, 0, 0), 1)
        else:
            debug_image[component] = [0, 0, 255]  # Red = not merged
            text = f"NOT_MERGED: d{min_dist:.1f}"
            #cv2.putText(debug_image, text, (int(cx-30), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 
             #          0.4, (255, 0, 0), 1)


def count_branches_simple(skeleton):
    """Count branch points in skeleton (pixels with >2 neighbors)."""
    if skeleton.dtype != np.uint8:
        skeleton = skeleton.astype(np.uint8)
    
    branch_count = 0
    skeleton_points = np.where(skeleton > 0)
    
    for i in range(len(skeleton_points[0])):
        y, x = skeleton_points[0][i], skeleton_points[1][i]
        
        if (1 <= y < skeleton.shape[0]-1 and 1 <= x < skeleton.shape[1]-1):
            neighbors = np.sum(skeleton[y-1:y+2, x-1:x+2]) - 1
            if neighbors > 2:
                branch_count += 1
    
    return branch_count



def process_lines_from_file(filename, image, width_path, threshold_ratio, min_branches, proximity_threshold, default_width=100):
    """Process rows file and identify wild oats using ray-casting approach for field mask generation.
    FIXED VERSION - preserves text annotations from wild oat detection."""
    
    image_shape = image.shape
    with open(filename, 'r') as file:
        try:
            lines = ast.literal_eval(file.read())  # Safely read the coordinates list
        except Exception as e:
            raise ValueError(f"Error reading or parsing the file: {e}")

    if not lines:
        return np.zeros_like(image), np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    with open(width_path, 'r') as file:
        width = ast.literal_eval(file.read())
        if width == 0:
            width = default_width
    
    # Original segmentation mask
    segmentation_mask = image.copy()
    
    # Determine if rows are primarily vertical or horizontal by examining first line
    first_line = lines[0]
    x1, y1, x2, y2 = first_line
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    is_more_vertical = dy > dx  # If vertical change > horizontal change, lines are more vertical
    
    # Create field mask based on orientation
    field_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
    h, w = image_shape[:2] if len(image_shape) > 2 else image_shape
    
    # Get the edge lines
    if is_more_vertical:
        # For vertical orientation, find leftmost and rightmost lines
        leftmost_line = min(lines, key=lambda line: min(line[0], line[2]))
        rightmost_line = max(lines, key=lambda line: max(line[0], line[2]))
        
        # For each row in the image, find where it intersects with leftmost and rightmost lines
        for y in range(h):
            # Calculate intersection with leftmost line
            x_left = calculate_line_intersection_at_y(leftmost_line, y, 0, w-1)
            
            # Calculate intersection with rightmost line
            x_right = calculate_line_intersection_at_y(rightmost_line, y, 0, w-1)
            
            # Ensure x_left <= x_right
            x_left, x_right = min(x_left, x_right), max(x_left, x_right)
            
            # Set pixels between the two x-coordinates to 1
            if 0 <= x_left <= w-1 and 0 <= x_right <= w-1:
                field_mask[y, x_left:x_right+1] = 1
    else:
        # For horizontal orientation, find topmost and bottommost lines
        topmost_line = min(lines, key=lambda line: min(line[1], line[3]))
        bottommost_line = max(lines, key=lambda line: max(line[1], line[3]))
        
        # For each column in the image, find where it intersects with topmost and bottommost lines
        for x in range(w):
            # Calculate intersection with topmost line
            y_top = calculate_line_intersection_at_x(topmost_line, x, 0, h-1)
            
            # Calculate intersection with bottommost line
            y_bottom = calculate_line_intersection_at_x(bottommost_line, x, 0, h-1)
            
            # Ensure y_top <= y_bottom
            y_top, y_bottom = min(y_top, y_bottom), max(y_top, y_bottom)
            
            # Set pixels between the two y-coordinates to 1
            if 0 <= y_top <= h-1 and 0 <= y_bottom <= h-1:
                field_mask[y_top:y_bottom+1, x] = 1
    
    # Create a mask for the row areas
    row_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
    
    # Draw rectangle corners on each row
    for i, line in enumerate(lines):
        rectangle_corners = calculate_rectangle_corners(line, width, image_shape)        # Calculate rectangle corners
        polygon = np.array([rectangle_corners], dtype=np.int32)
        cv2.fillPoly(row_mask, [polygon], 1)         # Fill polygon in the row mask
    
    # Apply field mask to the segmentation 
    segmentation_mask[field_mask == 0] = 0
    
    # Apply wild oat detection - THIS RETURNS THE DEBUG IMAGE WITH TEXT ANNOTATIONS
    wild_oat_mask, detection_debug = simple_wild_oat_detection(
        segmentation_mask=segmentation_mask,
        row_mask=row_mask, 
        threshold_ratio=threshold_ratio,
        min_branches=min_branches, 
        proximity_threshold=proximity_threshold
    )
    
    # FIXED: Use the detection_debug image directly (it has all the text annotations)
    # Just add the row visualization on top of it
    debug_image = detection_debug.copy()  # Keep all text annotations!
    
    # Add row visualization on top
    for i, line in enumerate(lines):
        # Draw the line itself
        cv2.line(debug_image, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 1)    # Green for center line
        
        rectangle_corners = calculate_rectangle_corners(line, width, image_shape)
        polygon = np.array([rectangle_corners], dtype=np.int32)
        
        cv2.polylines(debug_image, [polygon], True, (0, 255, 255), 2)   # Yellow for row boundaries

        for corner in rectangle_corners:
            cv2.circle(debug_image, corner, 3, (255, 0, 255), -1) # Magenta circles for corners
            
        # Add row number label
        label_pos = (int((rectangle_corners[0][0] + rectangle_corners[2][0]) / 2),
                    int((rectangle_corners[0][1] + rectangle_corners[2][1]) / 2))
        cv2.putText(debug_image, f"{i+1}", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Add field boundary
    field_boundary = cv2.dilate(field_mask, np.ones((3,3), np.uint8)) - field_mask
    debug_image[field_boundary > 0] = [255, 0, 0]  # Blue for field boundary
    
    return wild_oat_mask, debug_image


def calculate_line_intersection_at_y(line, y, min_x, max_x):
    """Calculate the x-coordinate where a line intersects with a specific y-coordinate.
    Extrapolates the line if necessary and handles vertical lines."""
    x1, y1, x2, y2 = line
    
    # Handle horizontal line
    if y1 == y2:
        return min(max(min(x1, x2), min_x), max_x)
    
    # Calculate intersection
    if y2 != y1:
        x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        
        # Bound to image width
        return int(min(max(x, min_x), max_x))
    else:
        # Vertical line
        return min(max(x1, min_x), max_x)


def calculate_line_intersection_at_x(line, x, min_y, max_y):
    """Calculate the y-coordinate where a line intersects with a specific x-coordinate.
    Extrapolates the line if necessary and handles horizontal lines."""
    x1, y1, x2, y2 = line
    
    # Handle vertical line
    if x1 == x2:
        return min(max(min(y1, y2), min_y), max_y)
    
    # Calculate intersection
    if x2 != x1:
        y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
        
        # Bound to image height
        return int(min(max(y, min_y), max_y))
    else:
        # Horizontal line
        return min(max(y1, min_y), max_y)


def calculate_rectangle_corners(line, width, image_shape):

    x1, y1, x2, y2 = line
    angle = np.arctan2(y2 - y1, x2 - x1)  # Angle of the line
    
    percentage = 15
    percentage_width = image_shape[0]*percentage if angle>=45 else image_shape[1]*percentage
    
    dx = width * np.sin(angle)  # Perpendicular x-offset
    dy = width * np.cos(angle)  # Perpendicular y-offset

    # Calculate the rectangle corners
    pt1 = (int(x1 - dx), int(y1 + dy))
    pt2 = (int(x1 + dx), int(y1 - dy))
    pt3 = (int(x2 + dx), int(y2 - dy))
    pt4 = (int(x2 - dx), int(y2 + dy))

    return [pt1, pt2, pt3, pt4]
