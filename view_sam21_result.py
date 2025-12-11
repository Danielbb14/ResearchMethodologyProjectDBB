

import logging
from pathlib import Path

import napari
import numpy as np
from tqdm import tqdm
from traccuracy.loaders import load_ctc_data

from divisualisation import Divisualisation
from divisualisation.utils import load_tiff_timeseries, rescale_intensity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_dir = Path(__file__).resolve().parent
result_path = base_dir / "result"
raw_data_path = base_dir / "data/REF_raw_data101_110 2/Pos101/aphase"


logger.info("Loading SAM2.1 tracking results")
logger.info(f"Result path: {result_path}")
logger.info(f"Raw data path: {raw_data_path}")

pred = load_ctc_data(
    str(result_path),
    str(result_path / "man_track.txt"),
    run_checks=False,
    name="sam21_prediction",
)

img = load_tiff_timeseries(raw_data_path)
img = np.stack([
    rescale_intensity(_x, pmin=5, pmax=99.9, clip=False, subsample=1)
    for _x in tqdm(img, desc="Rescale intensity")
])

v = napari.current_viewer()
if v is not None:
    v.close()
v = napari.Viewer()
for layer in v.layers:
    v.layers.remove(layer)
v.theme = "dark"

divis = Divisualisation(
    z_scale=1,
    time_scale=12,
    tracks_width=2,
)

v = divis.visualize_gt(
    v,
    x=img,
    masks=pred.segmentation,
    graph=pred.graph,
)
v.dims.set_current_step(0, 10)

logger.info("Visualization ready! Navigate through time to see tracking.")
napari.run()
