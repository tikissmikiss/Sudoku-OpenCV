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

b, g, r = cv2.split(img_color)
ceros = np.zeros(img_color.shape[:2], dtype="uint8")
unos = np.ones(img_color.shape[:2], dtype="uint8") * 255
jtk.show_window("B", cv2.merge([b, ceros, ceros]))
jtk.show_window("G", cv2.merge([ceros, g, ceros]))
jtk.show_window("R", cv2.merge([ceros, ceros, r]))
cv2.waitKey(WAIT_DELAY)

B2 = process("B", b, 215)
G2 = process("G", g, 87)
R2 = process("R", r, 87)
jtk.show_window("B", cv2.merge([B2, ceros, ceros]))
jtk.show_window("G", cv2.merge([ceros, G2, ceros]))
jtk.show_window("R", cv2.merge([ceros, ceros, R2]))
cv2.waitKey(WAIT_DELAY)

B2 = jtk.show_window("B", cv2.merge([B2, ceros, ceros]))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
B2 = cv2.erode(B2, kernel, iterations=4)
B2 = jtk.show_window("B", B2, wait=WAIT_DELAY)
B2 = cv2.dilate(B2, kernel, iterations=4)
B2 = jtk.show_window("B", B2, wait=WAIT_DELAY)
contornos, h = cv2.findContours(
    cv2.split(B2)[0],
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE)
contornos = sorted(contornos, key=cv2.contourArea, reverse=True)
contorno_mayor = contornos[0]
(x, y, w, h) = cv2.boundingRect(contorno_mayor)
cv2.rectangle(B2, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.drawContours(B2, [contorno_mayor], -1, (0, 0, 255), 2)
B2 = jtk.show_window("B", B2)
cv2.waitKey(0)
# mask = ceros.copy()
perimetro = cv2.arcLength(contorno_mayor, True)
pol = cv2.approxPolyDP(contorno_mayor, 0.001 * perimetro, True)
mask = cv2.drawContours(ceros.copy(), [pol], -1, (255, 0, 0), -1)
jtk.show_window("B", mask)
cv2.waitKey(0)
mask_inv = cv2.bitwise_not(mask)
jtk.show_window("B", mask_inv)
cv2.waitKey(0)
# B2 = cv2.bitwise_and(b, b, mask=mask)
# B2 = jtk.show_window("B", B2)
# cv2.waitKey(0)

# b, g, r = mask +  (b, g, r)
mask, mask_inv = mask // 255, mask_inv // 255
b = b * mask_inv
jtk.show_window("B", cv2.merge([b, ceros, ceros]))
jtk.show_window("G", cv2.merge([ceros, g, ceros]))
jtk.show_window("R", cv2.merge([ceros, ceros, r]))
cv2.waitKey(0)

# img_color = cv2.merge([b, g, r])
img_color = img_color - cv2.merge([b, ceros, ceros])
jtk.show_window("ORIGINAL", img_color, wait=0)

# #############################################################################
# Buscar tablero
# #############################################################################

# Convertir imagen a escala de grises
img_gray = jtk.show_window(
    "Sudoku",
    cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY),
    wait=WAIT_DELAY)

# Aplicar filtro gausiano para eliminar ruido
img_denoise = jtk.show_window(
    "Sudoku",
    jtk.gaussian_filter(img_gray, 7),
    wait=WAIT_DELAY)


# # Umbralización adaptativa
# img_denoise = jtk.show_window(
#     "Sudoku",
#     jtk.umbralizacion_adaptativa(
#         img_denoise,
#         type=cv2.THRESH_BINARY,
#         vecinos=50,
#         c_substract=0
#     ),
#     wait=WAIT_DELAY)

# jtk.show_hist(img_gray)


# Umbralización
pixels = img_height*img_width
brightness = int(np.sum(img_denoise)/pixels)
thr = brightness - (brightness*0.2)
img_denoise = jtk.show_window(
    "Sudoku",
    jtk.umbralizacion(img_denoise, thr=thr, type=cv2.THRESH_BINARY_INV),
    wait=WAIT_DELAY)


# Detección de bordes
img_borders = jtk.show_window(
    "Sudoku",
    jtk.canny_filter(img_denoise, min_thr=15, max_thr=30),
    wait=WAIT_DELAY)


# kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (3, 3))
img_dilate = jtk.show_window(
    "Sudoku",
    cv2.dilate(
        img_borders,
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    ),
    wait=WAIT_DELAY)


# Generar base para mascara de bordes
masck = np.zeros((img_height, img_width), np.float32)
show_window("Masck", masck, wait=WAIT_DELAY)


# Deteccion de lineas
lines = cv2.HoughLinesP(
    img_dilate,
    rho=1, theta=np.pi/180,
    threshold=int(min_dim*0.3),
    minLineLength=min_dim*0.3,
    maxLineGap=min_dim*0.02)
# Dibujar lineas
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.line(masck, (x1, y1), (x2, y2), (255, 255, 255), 2)
show_window("ORIGINAL", img_color)
show_window("Masck", masck, wait=WAIT_DELAY*10)


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
masck = cv2.dilate(masck, kernel, iterations=2)
show_window("Masck", masck, wait=WAIT_DELAY*10)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
masck = cv2.erode(masck, kernel, iterations=6)
show_window("Masck", masck, wait=WAIT_DELAY*10)


# Detector de esquinas
# masck = jtk.show_window(
#     "Masck",
#     cv2.cornerHarris(masck, 11, 7, 0.01),
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
