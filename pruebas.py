from argparse import ArgumentParser as AP
import cv2
from matplotlib import pyplot as plt

import numpy as np

import util.josetoolkit as jtk
from util.josetoolkit import *


# Defininos el menú del programa.
ap = AP()
ap.add_argument('-i', '--image', default=DEF_SUDOKU_IMG, required=False,
                help='Ruta a la imagen de entrada.')
args = vars(ap.parse_args())

# #############################################################################
# Leer y mostrar imagen
# #############################################################################

img_original = jtk.show_image(args['image'])

# Redimensionar imagen para trabajar siempre con el mismo ancho
img_resized, img_height, img_width, min_dim = std_resize(img_original)

# Mostrar lienzo en color y copiar imagen original
img_color = jtk.show_window("ORIGINAL", img_resized.copy(), wait=WAIT_DELAY)


# #############################################################################
# Buscar tablero
# #############################################################################

# Convertir imagen a escala de grises
print("Convertir imagen a escala de grises")
img_gray = jtk.show_window(
    "Sudoku",
    cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY),
    wait=WAIT_DELAY)

# Aplicar filtro gausiano para eliminar ruido
print("Aplicar filtro gausiano para eliminar ruido")
img_denoise = jtk.show_window(
    "Sudoku",
    jtk.gaussian_filter(img_gray, 7),
    wait=WAIT_DELAY)

# Umbralización
print("Umbralización")
pixels = img_height*img_width
brightness = int(np.sum(img_denoise)/pixels)
thr = brightness - (brightness*0.2)
img_denoise = jtk.show_window(
    "Sudoku",
    jtk.umbralizacion(img_denoise, thr=thr, type=cv2.THRESH_BINARY_INV),
    wait=WAIT_DELAY)

# Detección de bordes
print("Detección de bordes")
img_borders = jtk.show_window(
    "Sudoku",
    jtk.canny_filter(img_denoise, min_thr=15, max_thr=30),
    wait=WAIT_DELAY)

# Dilatar bordes
print("Dilatar bordes")
img_dilate = jtk.show_window(
    "Sudoku",
    cv2.dilate(
        img_borders,
        cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    ),
    wait=WAIT_DELAY)


# Erosionar bordes
print("Erosionar bordes")
img_erode = jtk.show_window(
    "Sudoku",
    cv2.erode(
        img_dilate,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    ),
    wait=WAIT_DELAY)


# Crear matriz para mascara de bordes
print("Generar base para mascara de bordes")
mask = np.zeros((img_height, img_width), np.uint8)
show_window("Masck", mask, wait=WAIT_DELAY)


# Deteccion de lineas
print("Deteccion de lineas")
lines = cv2.HoughLinesP(
    img_erode,
    rho=1, theta=np.pi/180,
    threshold=int(min_dim*0.5),
    minLineLength=min_dim/20,
    maxLineGap=min_dim*0.001)


# Dibujar lineas
print("Dibujar lineas")
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.line(mask, (x1, y1), (x2, y2), (255, 255, 255), 2)
show_window("ORIGINAL", img_color)
show_window("Masck", mask, wait=WAIT_DELAY*0.01)


# Eliminar lineas finas
print("Eliminar lineas finas")
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (22, 22))
mask = cv2.dilate(mask, kernel, iterations=1)
show_window("Masck", mask, wait=WAIT_DELAY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (22, 22))
mask = cv2.erode(mask, kernel, iterations=1)
show_window("Masck", mask, wait=WAIT_DELAY)

# Contornos
contornos, h = cv2.findContours(
    mask,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE)
contorno = max(contornos, key=cv2.contourArea)
perimetro = cv2.arcLength(contorno, True)
poligon = cv2.approxPolyDP(contorno, 0.1 * perimetro, True)
cv2.drawContours(mask, [poligon], -1, (255), -1)
img_color = img_resized.copy()
cv2.drawContours(img_color, [poligon], -1, (0, 0, 255), 6)
jtk.show_window("ORIGINAL", img_color)
jtk.show_window("Masck", mask, wait=WAIT_DELAY)

mask = np.zeros((img_height, img_width), np.uint8)

# Detector de esquinas
# mask = jtk.show_window(
#     "Masck",
#     cv2.cornerHarris(mask, 11, 7, 0.01),
#     wait=0)


cv2.waitKey(0)
cv2.destroyWindow("Sudoku")
# #################################################
# FIN
# #################################################
cv2.waitKey(0)
cv2.destroyAllWindows()
exit(0)


# lines2 = cv2.HoughLines(img_borders, 1, np.pi/180, int(img_height*0.2))
# for line in lines:
#     rho, theta = line[0]
#     v = np.cos(theta), np.sin(theta)
#     p0 = (int(v[0] * rho), int(v[1] * rho))
#     p1 = (int(p0[0]), int(p0[1]))
#     p2 = (int(p0[0] + img_width * v[1]), int(p0[1] + img_height * v[0]))
#     cv2.line(img_borders, p1, p2, (255, 255, 255), 2)
#     cv2.line(img_orig_board, p1, p2, (0, 0, 255), 2)
#     jtk.show_window("Sudoku board PROCESSED", img_borders, force_square=True)
#     jtk.show_window("Sudoku board ORIGINAL", img_orig_board, force_square=True)
#     cv2.waitKey(jtk.WAIT_DELAY//3)


while True:
    # reading the image which is to be transformed
    imagergb = cv2.imread(".\img\opencv_sudoku_puzzle_outline.png",
                          cv2.IMREAD_UNCHANGED)
    # imagergb = cv2.imread('C:/Users/admin/Desktop/plane.jpg')
    # specifying the points in the source image which is to be transformed to the corresponding points in the destination image
    srcpts = np.float32([[0, 100], [700, 260], [0, 700], [700, 400]])
    destpts = np.float32([[0, 200], [600, 0], [0, 700], [1000, 700]])
    # applying PerspectiveTransform() function to transform the perspective of the given source image to the corresponding points in the destination image
    resmatrix = cv2.getPerspectiveTransform(srcpts, destpts)
    # applying warpPerspective() function to display the transformed image
    resultimage = cv2.warpPerspective(imagergb, resmatrix, (500, 600))
    # displaying the original image and the transformed image as the output on the screen
    cv2.imshow('frame', imagergb)
    cv2.imshow('frame1', resultimage)
    if cv2.waitKey(24) == 27:
        break


""" 
# cv2.getPerspectiveTransform(source_coordinates, destination_coordinates)

Donde las coordenadas de origen (source_coordinates) son los puntos de la 
imagen de origen cuya perspectiva debe cambiarse y las coordenadas de destino 
(destination_coordinates) son los puntos correspondientes a los puntos de la 
imagen de origen, en la imagen de destino 
"""


# #############################################################################
# FIN
# #############################################################################
cv2.waitKey(0)
cv2.destroyAllWindows()
exit(0)


# transform_matrix = cv.getPerspectiveTransform(img_coord, frame_coord)
# donde las coordenadas de origen son los puntos de la imagen de origen cuya perspectiva debe cambiarse y las coordenadas de destino son los puntos correspondientes a los puntos de la imagen de origen, en la imagen de destino
# cv.warpPerspective
