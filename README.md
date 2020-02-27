# SLAM using Particle Filter

This project aims to simultaneously localize a walking humanoid robot and map an unknown indoor environment using odometry data, and a 2D laser range scanner (LIDAR). A particle filter based approach is taken to achieve the objective.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Plese download the data [here](https://drive.google.com/open?id=1dIeLDzlFw1VsbEY6s6QgxyKUs6b8uKvf)

## Code organization

    .
    ├── docs                  # Folder contains robot and data specs
    ├── report                # Folder contains my report analysis
    ├── results               # Folder contains final results images
    ├── src                   # Python scripts
    │   ├── main.py           # Main particle filter SLAM file
    │   ├── tools.py          # Helper for partical filter SLAM
    │   ├── load_data.py	  # Load lidar / joint / cam data
    │   ├── map_utils.py	  # Utility sets of the map function
    │   └── motion_utils.py	  # Utility sets of the motion function
    └── README.md

## Running the tests

### Steps

1. Modify line 12 and 13 in `main.py` if you want to try different dataset.
2. Run the command `python main.py` and the resulting images will display.

## Implementations

* See the [report](https://github.com/arthur960304/particle-filter-slam/blob/master/report/report.pdf) for detailed implementations.

## Results
![Data 4](https://github.com/arthur960304/particle-filter-slam/blob/master/results/data4.png)


## Authors

* **Arthur Hsieh** - *Initial work* - [arthur960304](https://github.com/arthur960304)

## References
[1] - [Sebastian Thrun, “Particle Filters in Robotics"](http://robots.stanford.edu/papers/thrun.pf-in-robotics-uai02.pdf)
