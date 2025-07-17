# Deep Learning Path Planning
This is the model I created for Path Planning team during my time in ASU Racing Team.
### Procedure
1. It starts by generating the track
2. Creating Data Loader with the following function:
      - Add noise to data
      - Normalize data (make every batch data start from the origin and make its angle 90 to x_axis) to make relative to current car position
      - pad and mask if left array not equal to right array
      - interleave to make one array containing both left and right cones one cone after another (instead of out it was originally: all left cones then all right cones)
      - reverse track: this is only done randomly to generate both clock wise and anti clock wise tracks
      - collate: make tracks into batches
3. Training the model
4. Visualizing the output
> The generated output is as follows:
<img width="611" height="451" alt="image" src="https://github.com/user-attachments/assets/8511c1ab-9be4-4e9c-b6b4-ab34dd47ea1b" />


