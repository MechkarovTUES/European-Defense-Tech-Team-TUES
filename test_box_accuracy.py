def calculate_overlap(coords_correct, coords_to_check_accuracy):
    x1, y1, x2, y2 = coords_correct
    x1_check, y1_check, x2_check, y2_check = coords_to_check_accuracy
    
    x_intersect1 = max(x1, x1_check)
    y_intersect1 = max(y1, y1_check)
    x_intersect2 = min(x2, x2_check)
    y_intersect2 = min(y2, y2_check)
    
    intersect_width = max(0, x_intersect2 - x_intersect1)
    intersect_height = max(0, y_intersect2 - y_intersect1)
    
    intersect_area = intersect_width * intersect_height
    
    # Calculate the area of the correct coordinates
    correct_area = (x2 - x1) * (y2 - y1)
    
    accuracy = intersect_area / correct_area
    
    return accuracy

# Example usage
coords_correct = [0, 0, 4, 4]
coords_to_check_accuracy = [0, 0, 2, 2]

accuracy = calculate_overlap(coords_correct, coords_to_check_accuracy)
print(f"Accuracy: {accuracy}")
