# =============================================================================
# Copyright (c) 2025 Oak Ridge National Laboratory
#
# All rights reserved.
#
# This file is part of Raptor.
#
# For details, see the top-level LICENSE file at:
# https://github.com/ORNL-MDF/Raptor/LICENSE
# =============================================================================
import numpy as np

from raptor.api import (
    compute_morphology,
    compute_porosity,
    create_grid,
    create_melt_pool,
    create_path_vectors,
    write_morphology,
    write_vtk,
)
from raptor.utilities import MeltPoolFilter


# =============================================================================
# User configuration
# =============================================================================
# These are the main parameters to vary in a standard AM optimization study.
# All lengths are in meters, speed is in m/s, power is in W, and angles are in
# degrees.

# Process parameters
# Accepted by the API, but does not affect the result (see note below).
LASER_POWER = 370.0
SCAN_SPEED = 1.7
HATCH_SPACING = 140.0e-6
LAYER_HEIGHT = 30.0e-6
SCAN_ROTATION = 67.0

# Mean melt-pool geometry
MELT_POOL_WIDTH = 148.0e-6
MELT_POOL_DEPTH = 118.0e-6
MELT_POOL_HEIGHT = 59.2e-6

# Stochastic melt-pool model
MELT_POOL_WIDTH_STD_DEV = 18.0e-6
MELT_POOL_LENGTH = 300.0e-6
N_SPECTRAL_MODES = 50
RANDOM_SEED = 42

# Lamé-curve exponents: 1 = parabola, 2 = ellipse.
HEIGHT_SHAPE_FACTOR = 1.0
DEPTH_SHAPE_FACTOR = 1.0

# RVE and output controls
RVE_MIN_POINT = np.array([0.0, 0.0, 0.0])
RVE_MAX_POINT = np.array([5.0e-4, 5.0e-4, 5.0e-4])
VOXEL_RESOLUTION = 5.0e-6
VTK_OUTPUT = "rve.vti"
MORPHOLOGY_OUTPUT = "rve_morphology.csv"


def build_melt_pool():
    """Create a stochastic melt pool from the top-level configuration."""
    np.random.seed(RANDOM_SEED)

    # Internal signal-generation settings. These control the numerical
    # representation of the stochastic melt-pool history.
    sampling_frequency = 250000.0
    time_series_duration = 0.08

    melt_pool_filter = MeltPoolFilter(
        MELT_POOL_WIDTH,
        MELT_POOL_WIDTH_STD_DEV,
        SCAN_SPEED,
        [sampling_frequency, time_series_duration],
    )

    # MeltPoolFilter converts this length scale to a temporal frequency
    # using SCAN_SPEED / MELT_POOL_LENGTH.
    melt_pool_filter.add_effect("melt_pool", [MELT_POOL_LENGTH, None, 1.0])
    melt_pool_filter.initialize()
    width_data = melt_pool_filter.generate_fluctuations(noise_scale=1.0)

    # Depth and height reuse the width history, scaled to their requested means.
    melt_pool_dict = {
        "width": (width_data, N_SPECTRAL_MODES, 1.0, 1.0),
        "depth": (
            width_data,
            N_SPECTRAL_MODES,
            MELT_POOL_DEPTH / MELT_POOL_WIDTH,
            DEPTH_SHAPE_FACTOR,
        ),
        "height": (
            width_data,
            N_SPECTRAL_MODES,
            MELT_POOL_HEIGHT / MELT_POOL_WIDTH,
            HEIGHT_SHAPE_FACTOR,
        ),
    }

    return create_melt_pool(melt_pool_dict, enable_random_phases=True)


def main():
    bound_box = np.array([RVE_MIN_POINT, RVE_MAX_POINT])
    grid = create_grid(VOXEL_RESOLUTION, bound_box=bound_box)

    # LASER_POWER is accepted by the current scan-path API but is not used by
    # the porosity calculation. An optimization must supply its relationship to
    # melt-pool geometry through the parameters.
    path_vectors = create_path_vectors(
        bound_box,
        LASER_POWER,
        SCAN_SPEED,
        HATCH_SPACING,
        LAYER_HEIGHT,
        SCAN_ROTATION,
        scan_extension=max(RVE_MAX_POINT - RVE_MIN_POINT),
        extra_layers=0,
    )

    melt_pool = build_melt_pool()
    porosity = compute_porosity(grid, path_vectors, melt_pool, jit_warmup=True)

    write_vtk(grid.origin, grid.resolution, porosity, VTK_OUTPUT)

    morphology = compute_morphology(
        porosity,
        VOXEL_RESOLUTION,
        ["area", "equivalent_diameter_area"],
    )
    write_morphology(morphology, MORPHOLOGY_OUTPUT)


if __name__ == "__main__":
    main()
