import cv2
import numpy as np
import pyvista as pv


img = cv2.imread(r'IMG\a-caneca-branca-na-mesa-de-madeira-para-maquete-ou-plano-de-fundo_35719-3034.png')
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


output_image = img.copy()
output_image = cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)


height, width = output_image.shape[:2]

xyz_points = []

depth_factor = 30  # Aumente este valor para aumentar a profundidade

for y in range(height):
    for x in range(width):
        z = (output_image[y, x, 0] / 255.0) * depth_factor
        xyz_points.append([x, y, z])

xyz_points = np.array(xyz_points)


grid = pv.StructuredGrid()
grid.points = xyz_points
grid.dimensions = (width, height, 1)


texture = pv.numpy_to_texture(img)
grid.texture_map_to_plane(inplace=True)
texture.flip(1)


p = pv.Plotter()
p.add_mesh(grid, texture=texture)
p.show()
