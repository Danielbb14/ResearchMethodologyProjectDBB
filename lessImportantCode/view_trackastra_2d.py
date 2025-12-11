import logging
import argparse
from pathlib import Path

import napari
import numpy as np
from tqdm import tqdm
from traccuracy.loaders import load_ctc_data

from divisualisation import Divisualisation
from divisualisation.utils import load_tiff_timeseries, rescale_intensity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dataset configurations
DATASETS = {
    '1': {
        'name': 'RIF10_Pos202',
        'result': 'result1',
        'raw_path': 'RIF10_raw_data201_210 2/Pos202/aphase'
    },
    '2': {
        
        'name': 'RIF10_Pos201',
        'result': 'result2',
        'raw_path': 'RIF10_raw_data201_210 2/Pos201/aphase'
    },
    '3': {

        'name': 'REF_Pos102', 
        'result': 'result3',
        'raw_path': 'REF_raw_data101_110 2/Pos102/aphase'
    },
    '4': {
        'name': 'REF_Pos101',
        'result': 'result44',
        'raw_path': 'REF_raw_data101_110 2/Pos101/aphase'
    }
}

# Parse command line arguments
parser = argparse.ArgumentParser(description='View Trackastra 2D tracking results')
parser.add_argument('--dataset', type=str, default='1', choices=['1', '2', '3', '4'],
                    help='Dataset to visualize: 1=REF_Pos101, 2=REF_Pos102, 3=RIF10_Pos201, 4=RIF10_Pos202')
args = parser.parse_args()

# Get selected dataset configuration
dataset = DATASETS[args.dataset]
base_dir = Path("/Users/dbb14/Desktop/Uppsala University/period 6/ResearchMethodology")

result_path = base_dir / "trackastra_results" / dataset['result']
raw_data_path = base_dir / dataset['raw_path']

logger.info(f"Loading dataset: {dataset['name']}")
logger.info(f"Result path: {result_path}")
logger.info(f"Raw data path: {raw_data_path}")

pred = load_ctc_data(
    str(result_path),
    str(result_path / "res_track.txt"),
    run_checks=False,
    name="trackastra_prediction",
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

napari.run()
