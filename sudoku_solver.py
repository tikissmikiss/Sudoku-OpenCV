# @ref https://pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/
# @ref https://nanonets.com/blog/ocr-with-tesseract/

# Importamos las librerías necesarias
import os
import tkinter
import webbrowser
from tkinter import messagebox
from argparse import ArgumentParser as AP

import cv2
import numpy as np
from imutils.perspective import four_point_transform
from matplotlib.pyplot import contour
from skimage.segmentation import clear_border

import util.josetoolkit as jtk
from sudoku import print_board, solve

try:
    import pytesseract as ocr
except ImportError:
    window = tkinter.Tk()
    window.wm_withdraw()
    sel_exit = False
    res = messagebox.askquestion(
        'Módulo no instalado', '¿Desea instalar la librería pytesseract? ' +
        '\n\nAun asi es necesario tener instalado Tesseract, ' +
        'puede descargarlo aqui: \n\n' +
        'https://tesseract-ocr.github.io/tessdoc/Installation.html')
    if res == 'yes':
        os.system('pip install pytesseract')
        import pytesseract as ocr
        webbrowser.open(
            'https://tesseract-ocr.github.io/tessdoc/Installation.html')
    else:
        res = messagebox.askquestion(
            '¿Continuar?', '¿Desea continuar sin el modulo de OCR?')
        if res == 'no':
            sel_exit = True
    window.destroy()
    if sel_exit:
        exit(0)


# Defininos el menú del programa.
ap = AP()
ap.add_argument('-i', '--image', default=jtk.DEF_SUDOKU_IMG, required=False,
                help='Ruta a la imagen de entrada.')
args = vars(ap.parse_args())

# #############################################################################
# Leer y mostrar imagen
# #############################################################################

img_original = jtk.show_image(args['image'])

img_height, img_width = img_original.shape[:2]

# Mostramos la imagen original
jtk.show_window("Sudoku", img_original)
cv2.waitKey(jtk.WAIT_DELAY*4)

# Convertir imagen a escala de grises
img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
jtk.show_window("Sudoku", img_gray)
cv2.waitKey(jtk.WAIT_DELAY*2)

# Aplicar filtro gausiano para eliminar ruido
img_denoise = jtk.gaussian_filter(img_gray, 5)
jtk.show_window("Sudoku", img_denoise)
cv2.waitKey(jtk.WAIT_DELAY*2)

# Detección de bordes
if jtk.op_borders == jtk.OP_BORDERS_CANNY:                   # Umbralización o Canny
    img_borders = jtk.canny_filter(img_denoise, min_thr=15, max_thr=25)
elif jtk.op_borders == jtk.OP_BORDERS_ADAPTATIVE_THRESHOLD:  # Umbralización o Canny
    img_borders = jtk.umbralizacion_adaptativa(
        img_denoise, cv2.THRESH_BINARY_INV, 25, 2)
elif jtk.op_borders == jtk.OP_BORDERS_THRESHOLD:             # Umbralización o Canny
    img_borders = jtk.umbralizacion(img_denoise, 160, cv2.THRESH_BINARY_INV)
# Umbralización o Sobel TODO: Pendiente de implementar
elif jtk.op_borders == jtk.OP_BORDERS_SOBEL:
    pass
jtk.show_window("Sudoku", img_borders)
cv2.waitKey(jtk.WAIT_DELAY*2)

# Seleccionar tablero
rectangle, board = jtk.get_tablero(img_borders)
img_board = jtk.draw_board(img_original, rectangle, board)
jtk.show_window("Sudoku", img_board)
cv2.waitKey(jtk.WAIT_DELAY*2)

# Focus en el tablero
img_orig_board = jtk.transform_board(img_original, board)
img_gray_board = jtk.transform_board(img_gray, board)
jtk.show_window("Sudoku board ORIGINAL", img_orig_board, force_square=True)
jtk.show_window("Sudoku board PROCESSED", img_gray_board, force_square=True)
cv2.waitKey(jtk.WAIT_DELAY*2)


# Detección de bordes
if jtk.op_borders == jtk.OP_BORDERS_CANNY:                   # Umbralización o Canny
    img_borders = jtk.canny_filter(img_gray_board, min_thr=20, max_thr=70)
elif jtk.op_borders == jtk.OP_BORDERS_ADAPTATIVE_THRESHOLD:  # Umbralización o Canny
    img_borders = jtk.umbralizacion_adaptativa(
        img_gray_board, cv2.THRESH_BINARY_INV, 25, 5)
elif jtk.op_borders == jtk.OP_BORDERS_THRESHOLD:             # Umbralización o Canny
    img_borders = jtk.umbralizacion(img_gray_board, 160, cv2.THRESH_BINARY_INV)
# Umbralización o Sobel TODO: Pendiente de implementar
elif jtk.op_borders == jtk.OP_BORDERS_SOBEL:
    pass
jtk.show_window("Sudoku board PROCESSED", img_borders, force_square=True)
cv2.waitKey(jtk.WAIT_DELAY*2)

# #############################################################################
# Deteccion de lineas
# #############################################################################
# Detectar lineas horizontales y verticales
lines = cv2.HoughLines(img_borders, 1, np.pi/2, int(img_height*0.2))
# Dibujar lineas horizontales y verticales
for line in lines:
    rho, theta = line[0]
    v = np.cos(theta), np.sin(theta)
    p0 = (int(v[0] * rho), int(v[1] * rho))
    p1 = (int(p0[0]), int(p0[1]))
    p2 = (int(p0[0] + img_width * v[1]), int(p0[1] + img_height * v[0]))
    cv2.line(img_borders, p1, p2, (255, 255, 255), 2)
    cv2.line(img_orig_board, p1, p2, (0, 0, 255), 2)
    jtk.show_window("Sudoku board PROCESSED", img_borders, force_square=True)
    jtk.show_window("Sudoku board ORIGINAL", img_orig_board, force_square=True)
    cv2.waitKey(jtk.WAIT_DELAY//3)

# #############################################################################
# Deteccion de celdas
# #############################################################################
# There are four types in retrieval mode in OpenCV.
# - cv2.RETR_LIST → Retrieve all contours
# - cv2.RETR_EXTERNAL → Retrieves external or outer contours only
# - cv2.RETR_COMP → Retrieves all in a 2-level hierarchy
# - cv2.RETR_TREE → Retrieves all in the full hierarchy
# Hierarchy is stored in the following format[next, previous, First child, parent].
contornos = cv2.findContours(
    img_borders, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

min_area = img_orig_board.shape[0] * img_orig_board.shape[1] * 0.5 / (9*9)
max_area = img_orig_board.shape[0] * img_orig_board.shape[1] / (9*9)
cells = []
for c in contornos[0]:
    if min_area < cv2.contourArea(c) and cv2.contourArea(c) < max_area:
        # Obtenemos el rectángulo que engloba al contorno
        (x, y, w, h) = cv2.boundingRect(c)
        cells.append((x, y, w, h))
        # cv2.drawContours(img_orig_board, [c], -1, (0, 255, 0), 2)
        # j.show_window("Sudoku board ORIGINAL", img_orig_board, force_square=True)
        # cv2.waitKey(int(j.tDelay/10))

# Comprobacion del numero de celdas encontradas
if len(cells) != 9*9:
    raise jtk.CellsError("No se encontraron todos los contornos de las celdas")

# Ordenar las celdas
cells = sorted(cells, key=lambda x: x[1])
for i in range(9):
    cells[i*9:i*9+9] = sorted(cells[i*9:i*9+9], key=lambda x: x[0])

# Dibujar las celdas
for c in cells:
    (x, y, w, h) = c
    cv2.rectangle(img_orig_board, (x, y), (x + w, y + h), (0, 0, 0), 3)
    jtk.show_window("Sudoku board ORIGINAL", img_orig_board, force_square=True)
    cv2.waitKey(int(jtk.WAIT_DELAY/10))


def draw_number(image, cell, number=8, color=(0, 0, 0)):
    (x, y, w, h) = cell
    wc, hc = 16, 21
    s = int(h*0.8/hc)
    wm, hm = -2*s, -1
    wc, hc = wc*s, hc*s
    font = cv2.FONT_HERSHEY_COMPLEX
    position = (x+int((wm+w/2-wc/2)), y+int((hm+h/2+hc/2)))
    line = s+1
    cv2.putText(image, str(number), position, font, s, color, line)


# Leer el contenido de las celdas
board_data = []
for c in cells:
    x, y, w, h = c
    # Obtenemos la imagen de la celda
    m = int(w*0.05)  # margen para evitar bordes
    points = np.array([[m+x, m+y], [x-m+w, m+y],
                      [x-m+w, y-m+h], [m+x, y-m+h]])
    cell = four_point_transform(img_gray_board, points)
    jtk.show_window("Sudoku cell", cell, size=200, force_square=True)
    cv2.waitKey(int(jtk.WAIT_DELAY/10))
    cell = cv2.GaussianBlur(cell, (5, 5), 1)
    jtk.show_window("Sudoku cell", cell, size=200, force_square=True)
    cv2.waitKey(int(jtk.WAIT_DELAY/10))
    cell = jtk.umbralizacion_adaptativa(cell, cv2.THRESH_BINARY_INV, 40, 5)
    cell = cv2.bitwise_not(cell)
    cell = cv2.GaussianBlur(cell, (7, 7), 1)
    jtk.show_window("Sudoku cell", cell, size=200, force_square=True)
    cv2.waitKey(int(jtk.WAIT_DELAY/10))
    try:
        custom_config = r'--oem 3 --psm 6 outputbase digits'
        text = ocr.image_to_string(cell, config=custom_config)
    except Exception as e:
        jtk.tesseract_error(e)
    jtk.draw_number(img_orig_board, c, text[0] if text != '' else '', (0, 0, 255))
    jtk.show_window("Sudoku board ORIGINAL", img_orig_board, force_square=True)
    text = text[0] if text != '' else '0'
    board_data.append(text)
    print(text, end=' ')
    jtk.show_window("Sudoku cell", cell, size=200, force_square=True)
    cv2.waitKey(int(jtk.WAIT_DELAY/10))
board_data = np.array(board_data).reshape(9, 9).astype(int)
print()
print(board_data)
cv2.waitKey(jtk.WAIT_DELAY*3)

# Resolver sudoku
print_board('Problem', board_data)
cv2.waitKey(jtk.WAIT_DELAY*3)
solution = solve(board_data)
print_board('Solution', solution)

# Imprimir solucion
for r in range(9):
    for c in range(9):
        if board_data[r, c] == 0:
            draw_number(img_orig_board,
                        cells[r*9+c], solution[r][c], (14, 168, 12))
            jtk.show_window("Sudoku board ORIGINAL",
                            img_orig_board, force_square=True)
            cv2.waitKey(jtk.WAIT_DELAY//10)


# #################################################
# FIN
# #################################################
cv2.waitKey(0)
cv2.destroyAllWindows()
exit(0)


# transform_matrix = cv.getPerspectiveTransform(img_coord, frame_coord)
# donde las coordenadas de origen son los puntos de la imagen de origen cuya perspectiva debe cambiarse y las coordenadas de destino son los puntos correspondientes a los puntos de la imagen de origen, en la imagen de destino
# cv.warpPerspective
