# Reproducible Research Archive

Haoran Chen and Robert F. Murphy\
Ray & Stephanie Lane Computational Biology Department
School of Computer Science
Carnegie Mellon University\
March 8, 2024 (updated August 5, 2024)

This is the reproducible research archive that can be used to reproduce all results and figures in the manuscript "3DCellComposer - A Versatile Pipeline Utilizing 2D Cell Segmentation Methods for 3D Cell Segmentation".

## Overview

This repository can be used in three ways.  

The first is to **(re)generate all results in the manuscript** from original images.  This requires first downloading the input datasets (IMC from HuBMAP) and hiPSC from Allen Institute for Cell Science) - scripts for this purpose are available in the 'downloading' folder.  The images are dowloaded into the 'data' folder, which contains an 'AICS' folder with subfolders for different image sets.  Within those subfolders are lists of specific images to be downloaded.  Further information on regenerating all results is provided below. 

The second is to only **(re)generate the figures and tables** from the results.  This requires downloading a large dataset that contains all of the intermediate data generated.  The 'download_zenodo.py' script can be used for this purpose.

The third is to do an **example run of 3DCellComposer** using just a single input image and one 2D segmentation model.

## Regenerating all results

As described above, the input images must first be downloaded using `download_IMC.py` and `download_hiPSC.py`.  After this, the main `run_RRA.py` can be used to reproduce all results.  Note that this requires approximately __ days to run.  Note also that it requires prior installation of the packages for the various segmentation models.

The main script consists of four steps.  The channels needed for segmentation are selected from the input images using code in the *preprocessing* folder.  The individual segmentation models are run on the preprocessed images using wrappers contained in the *segmentation_2D* and *segmentation_3D* folders.  Evaluation metrics are calculated from the segmentation results using code in the *evaluation* folder.  Finally, figures are generated using the *plotting* folder.

## Regenerating figures and tables

The generated results must first be dowloaded using `download_zenodo.py`.  The `run_plotting.py` script in the *plotting* folder can then be used to generate all figures and tables (see exceptions below).  Alternatively, individual figures or tables can be generated with the appropriate script in the *plotting* folder.

The exceptions are the scripts Figures 1 and 2 generated pieces that are used outside python to generate the final figures.

## Use example

The *example* folder illustrates the basic use of 3DCellComposer.  It contains an `run_example.py` script and a small example 3D image.

Contact: murphy@cmu.edu





