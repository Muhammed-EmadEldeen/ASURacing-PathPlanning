from track_generator import TrackGenerator
from utils import Mode, SimType
import os


# Input parameters
n_points = 60
n_regions = 20
min_bound = 0.
max_bound = 150.
mode = Mode.EXTEND
sim_type = SimType.FSDS

# Output options
plot_track = True
visualise_voronoi = False
create_output_file = True
output_location = '/output_tracks/'

# Generate track
track_gen = TrackGenerator(n_points, n_regions, min_bound, max_bound, mode, plot_track, visualise_voronoi, create_output_file, output_location, lat_offset=51.197682, lon_offset=5.323411, sim_type=sim_type)

j=0
for i in range (1000):
    try:
        track_gen.create_track()
        src_file_path = f"./output_tracks/random_track{j}.csv"
        while os.path.exists(src_file_path): 
            j += 1
            src_file_path = f"./output_tracks/random_track{j}.csv"
        os.rename("./output_tracks/random_track.csv",src_file_path) 
        print(i)
    except:
        continue
        
    
