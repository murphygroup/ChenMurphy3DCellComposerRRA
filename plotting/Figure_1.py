import numpy as np
import matplotlib.pyplot as plt
import pickle
import bz2
import pandas as pd
import vtk

def get_indices_pandas(data):
	d = data.ravel()
	f = lambda x: np.unravel_index(x.index, data.shape)
	return pd.Series(d).groupby(d).apply(f)

method = 'deepcell_membrane-0.12.6'
# cell = pickle.load(bz2.BZ2File(
# 	f'/data/3D/IMC_3D/florida-3d-imc/a296c763352828159f3adfa495becf3e/original/mask_{method}_matched_3D_final_0.3.pkl',
# 	'r')).astype(np.int64)
cell = pickle.load(bz2.BZ2File(
	f'/data/3D/AICS/AICS_actin/AICS-7_11536/original/mask_{method}_matched_3D_final_0.3.pkl',
	'r')).astype(np.int64)
cell = cell[20:50,200:500,200:500]
cell_coords = get_indices_pandas(cell)[1:]
for i in cell_coords.index:
	cell[cell_coords[i]] = np.random.randint(4)


# Convert numpy array to VTK array (vtkImageData)
imageData = vtk.vtkImageData()
imageData.SetDimensions(cell.shape)
imageData.AllocateScalars(vtk.VTK_INT, 1)

for z in range(cell.shape[2]):
	for y in range(cell.shape[1]):
		for x in range(cell.shape[0]):
			value = cell[x, y, z]
			imageData.SetScalarComponentFromFloat(x, y, z, 0, value)

# Create a lookup table for colors
colorLookupTable = vtk.vtkLookupTable()
colorLookupTable.SetNumberOfTableValues(4)
colorLookupTable.Build()
colorLookupTable.SetTableValue(0, 0, 0, 0, 0)  # Transparent for value 0
colorLookupTable.SetTableValue(1, 1, 0, 0, 1)  # Red for value 1
colorLookupTable.SetTableValue(2, 0, 1, 0, 1)  # Green for value 2
colorLookupTable.SetTableValue(3, 0, 0, 1, 1)  # Blue for value 3

# Map the image through the lookup table
colorMap = vtk.vtkImageMapToColors()
colorMap.SetLookupTable(colorLookupTable)
colorMap.SetInputData(imageData)
colorMap.Update()

# Create an actor
actor = vtk.vtkImageActor()
actor.GetMapper().SetInputData(colorMap.GetOutput())

# Create a renderer
renderer = vtk.vtkRenderer()

# Create a render window
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)

# Create a render window interactor
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

# Add the actor to the scene
renderer.AddActor(actor)

# Reset the camera to show the full scene
renderer.ResetCamera()

# Start the interaction
renderWindow.Render()
renderWindowInteractor.Start()
