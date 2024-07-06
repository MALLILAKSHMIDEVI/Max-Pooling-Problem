# Max Pooling Problem

## Overview

This project implements a solution to the max pooling problem for a given \( m \times m \) matrix with a \( k \times k \) moving window. Two implementations are provided:
1. A naive implementation.
2. An optimized implementation using sliding maximum technique.

## Requirements

- Python 3.x
- NumPy

## Installation

To run the code, ensure you have Python 3.x installed along with the NumPy library. You can install NumPy using pip if you don't have it installed:
pip install numpy


## Usage

1. Save the script to a file named `max_pooling.py`.
2. Run the script using Python:

```sh
python max_pooling.py

Example Output:
Naive Implementation Result:
[[6. 6. 5.]
 [8. 9. 9.]
 [8. 9. 9.]]

Optimized Implementation Result:
[[6. 6. 5.]
 [8. 9. 9.]
 [8. 9. 9.]]

Performance Verification:
Naive Implementation Time: 0.046026 seconds
Optimized Implementation Time: 0.004520 seconds
Both implementations produce the same result.


This code and README file meet the given requirements, providing a clear and efficient solution to the max pooling problem with performance verification.


