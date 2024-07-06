import numpy as np
from collections import deque
import time

def max_pooling_naive(matrix, k):
    """
    Naive implementation of max pooling.
    
    Parameters:
    matrix (ndarray): Input (m x m) matrix.
    k (int): Size of the moving window (k x k).
    
    Returns:
    ndarray: ((m-k+1) x (m-k+1)) matrix with max values for each window position.
    """
    m = matrix.shape[0]
    result = np.zeros((m - k + 1, m - k + 1))
    
    for i in range(m - k + 1):
        for j in range(m - k + 1):
            window = matrix[i:i+k, j:j+k]
            result[i, j] = np.max(window)
    
    return result

def sliding_maximum(arr, k):
    """
    Helper function to find the maximum values in a sliding window for a 1D array.
    
    Parameters:
    arr (ndarray): Input array.
    k (int): Size of the moving window.
    
    Returns:
    list: List of maximum values for each window position.
    """
    deq = deque()
    max_values = []
    
    for i in range(len(arr)):
        # Remove elements not in the window
        while deq and deq[0] <= i - k:
            deq.popleft()
        
        # Remove elements smaller than the current element
        while deq and arr[deq[-1]] <= arr[i]:
            deq.pop()
        
        deq.append(i)
        
        # Start adding max values after the first k-1 elements
        if i >= k - 1:
            max_values.append(arr[deq[0]])
    
    return max_values

def max_pooling_optimized(matrix, k):
    """
    Optimized implementation of max pooling using sliding maximum technique.
    
    Parameters:
    matrix (ndarray): Input (m x m) matrix.
    k (int): Size of the moving window (k x k).
    
    Returns:
    ndarray: ((m-k+1) x (m-k+1)) matrix with max values for each window position.
    """
    m = matrix.shape[0]
    temp_result = np.zeros((m, m - k + 1))
    final_result = np.zeros((m - k + 1, m - k + 1))
    
    # Apply sliding maximum row-wise
    for i in range(m):
        row_max = sliding_maximum(matrix[i], k)
        temp_result[i, :len(row_max)] = row_max
    
    # Apply sliding maximum column-wise
    for j in range(m - k + 1):
        col_max = sliding_maximum(temp_result[:, j], k)
        final_result[:len(col_max), j] = col_max
    
    return final_result

def verify_performance(matrix, k):
    """
    Verify and compare the performance of naive and optimized implementations.
    
    Parameters:
    matrix (ndarray): Input (m x m) matrix.
    k (int): Size of the moving window (k x k).
    
    Returns:
    None
    """
    # Time the naive implementation
    start_time = time.time()
    result_naive = max_pooling_naive(matrix, k)
    end_time = time.time()
    print(f"Naive Implementation Time: {end_time - start_time:.6f} seconds")
    
    # Time the optimized implementation
    start_time = time.time()
    result_optimized = max_pooling_optimized(matrix, k)
    end_time = time.time()
    print(f"Optimized Implementation Time: {end_time - start_time:.6f} seconds")
    
    # Verify that the results are the same
    assert np.allclose(result_naive, result_optimized), "Results do not match!"
    print("Both implementations produce the same result.")

if __name__ == "__main__":
    # Example usage
    matrix = np.array([
        [1, 3, 2, 1],
        [4, 6, 5, 3],
        [7, 8, 9, 6],
        [2, 4, 5, 8]
    ])
    k = 2
    
    # Perform max pooling using naive implementation
    print("Naive Implementation Result:")
    print(max_pooling_naive(matrix, k))
    
    # Perform max pooling using optimized implementation
    print("\nOptimized Implementation Result:")
    print(max_pooling_optimized(matrix, k))
    
    # Verify performance
    print("\nPerformance Verification:")
    large_matrix = np.random.randint(0, 100, size=(100, 100))
    verify_performance(large_matrix, k)
