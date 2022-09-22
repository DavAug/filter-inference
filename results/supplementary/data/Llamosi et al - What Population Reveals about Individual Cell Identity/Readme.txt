This files contain the two data sets Di and Dv used in the paper:
"What Population Reveals about Individual Cell Identity: Single-cell Parameter Estimation of Models of Gene Expression in Yeast"

Report to the aforementioned paper for details on experimental method. For both experiments, single-cell fluorescence values are provided. The subtraction of the camera offset is the only post-processing operation done. This can yield to slightly negative values for fluorescence as the camera noise is Gaussian centered on 0.

A Matlab and an excel versions are provided. 

Matlab file version.
The file contains two structs, Di and Dv, corresponding each to one experiment.
These structs contain 4 fields:
- singleCellData: this is the fluorescence values for each cell and each time point. NaN indicates when cells are either not born yet or lost (went out of the field of view)
- SamplingTime: this gives the time (in min) at which each frame of singleCellData is taken.
- t_on: gives the time (in min) at which hyperosmotic shocks are applied.
- t_dur: gives the duration of each shock (here all are equally long)

Excel file version
Two separate .xls files are given. One for each experiment.
The first page (called "Data") of each spreadsheet gives the raw fluorescence for each cell (in line) and each time point (in column). Blank indicates cells that are either not born yet or lost (went out of the field of view).
The second page (called "Input") of each spreadsheet gives the time at which osmotic stresses are applied.

