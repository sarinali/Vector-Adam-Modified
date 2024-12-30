from typing import List, Tuple

# Function to count inversions using merge sort
def count_inversions(arr: List[float]) -> Tuple[List[float], int]:
    if len(arr) < 2:
        return arr, 0  # Base case: a single element has no inversions
    
    mid = len(arr) // 2
    left, left_inv = count_inversions(arr[:mid])  # Recursive count on left half
    right, right_inv = count_inversions(arr[mid:])  # Recursive count on right half
    merged, cross_inv = merge_and_count(left, right)  # Count inversions while merging
    
    # Total inversions is the sum of left, right, and cross inversions
    return merged, left_inv + right_inv + cross_inv

def merge_and_count(left: List[float], right: List[float]) -> Tuple[List[float], int]:
    merged = []
    i = j = inversions = 0
    
    # Merging and counting cross inversions
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            inversions += len(left) - i  # All remaining elements in left are inversions
            j += 1
    
    # Append any remaining elements
    merged.extend(left[i:])
    merged.extend(right[j:])
    
    return merged, inversions

# Input array as per the example given
example_array = [163.5, 153.01, 156.30, 151.49]
sorted_array, inversion_count = count_inversions(example_array)
print(inversion_count)
