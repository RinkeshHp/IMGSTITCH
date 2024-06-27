import numpy as np
import matplotlib.pyplot as plt

# Define the range for x
x = np.linspace(0, 10, 40)

# Define the coefficients for the lines
coefficients = [
    {'m': 879.99999998, 'c': -0.9999999159256469, 'label': 'Zoom 5x'},
    {'m': 1201.84376454, 'c': -0.7375058188803622, 'label': 'Zoom 7x'},
    {'m': 477.58368357, 'c': 0.6500177251101603, 'label': 'Zoom 2.5x'}
]

# Plot each line
plt.figure(figsize=(10, 6))

for coeff in coefficients:
    y = coeff['m'] * x + coeff['c']
    plt.plot(x, y, label=coeff['label'])

# Add labels and title
plt.ylabel('pixel overlap')
plt.xlabel('step size(mm)')
plt.title('Combined Huber Regressor models')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
