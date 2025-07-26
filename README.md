# 2viewReconstruction

Implements a full two-view stereo pipeline for 3D scene reconstruction from RGB image pairs. Given calibrated camera poses, the system rectifies views, extracts matching patches, computes disparity using SSD, SAD, or ZNCC, filters results with left-right consistency, and reconstructs a depth map and point cloud in world coordinates
