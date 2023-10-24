import cv2
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os
import supervision as sv


# Carregue a imagem
img = cv2.imread('CG\Projeto\mesa_internet.png')

# Converta a imagem para tons de cinza
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplique a segmentação na imagem em tons de cinza
_, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Encontre os contornos na imagem segmentada
contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Crie uma máscara para os contornos
mask = np.zeros_like(img)

cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

# Aplique o algoritmo K-Means na imagem original para segmentar por cor
k = 3  # Número de clusters (pode ajustar isso)
img_flat = img.reshape((-1, 3)).astype(np.float32)  # Transforme a imagem em uma matriz 2D
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv2.kmeans(img_flat, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Recrie a imagem segmentada com base nos clusters
segmented_image = centers[labels.flatten()].reshape(img.shape).astype(np.uint8)

HOME = os.getcwd()
print("HOME:", HOME)

CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))

model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=CHECKPOINT_PATH)
mask_generator1 = SamAutomaticMaskGenerator(sam, points_per_batch=16)
mask_generator = SamAutomaticMaskGenerator(sam)

sam_result = mask_generator.generate(img)

"""
para executar ou pelo menos tentar, precisa baixar o modelo já pré treinado

!wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
"""

print(sam_result[0].keys())


mask_annotator = sv.MaskAnnotator()

detections = sv.Detections.from_sam(sam_result=sam_result)

annotated_image = mask_annotator.annotate(scene=img.copy(), detections=detections)

print(annotated_image)