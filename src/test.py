import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

track_xy = pd.read_csv(Path("data/track_geom/barber_track_xy.csv"))

plt.figure(figsize=(5, 5))
plt.plot(track_xy["x_norm"], track_xy["y_norm"], "-o", markersize=2)
plt.gca().set_aspect("equal", "box")
plt.title("Barber track – digitized centerline")
plt.show()