
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

def resample_curve(points, n_points):
    """Resample a 2D curve to have exactly n_points using linear interpolation."""
    # Compute cumulative arc length
    deltas = np.diff(points, axis=0)
    dists = np.sqrt((deltas**2).sum(axis=1))
    cumulative_dist = np.insert(np.cumsum(dists), 0, 0)
    total_dist = cumulative_dist[-1]
    new_distances = np.linspace(0, total_dist, n_points)
    
    # Interpolate x and y separately
    x_interp = np.interp(new_distances, cumulative_dist, points[:, 0])
    y_interp = np.interp(new_distances, cumulative_dist, points[:, 1])
    return np.stack((x_interp, y_interp), axis=1)

def generate_mid_track(left, right, upsample_factor=5, smoothing=0.5):
    n_points = min(len(left), len(right))
    
    # Resample both tracks to same length
    left_resampled = resample_curve(left, n_points)
    right_resampled = resample_curve(right, n_points)
    
    # Midpoints
    midpoints = (left_resampled + right_resampled) / 2

    # Smooth and upsample using spline
    tck, u = splprep([midpoints[:, 0], midpoints[:, 1]], s=smoothing)
    u_fine = np.linspace(0, 1, n_points * upsample_factor)
    x_smooth, y_smooth = midpoints[:, 0], midpoints[:, 1]

    return np.stack((x_smooth, y_smooth), axis=1)


data = pd.read_csv('./output_tracks/random_track0.csv', header=None)  # Assuming the CSV has no header
left = data[data[0] == 'blue'][[1, 2]].to_numpy()   # left = blue
right = data[data[0] == 'yellow'][[1, 2]].to_numpy()  # right = yellow


mid_track = generate_mid_track(left, right)

# Plot result
plt.scatter(left[:, 0], left[:, 1], c='blue', s=7, label='Left')
plt.scatter(right[:, 0], right[:, 1], c='gold', s=7, label='Right')
plt.scatter(mid_track[:, 0], mid_track[:, 1], c='red', linewidth=1, label='Center Track')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()

