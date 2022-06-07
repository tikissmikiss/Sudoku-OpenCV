# @ref https://pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/
# @ref https://nanonets.com/blog/ocr-with-tesseract/

# Importamos las librerías necesarias
import os
from time import sleep

import cv2
import numpy as np
import webbrowser
from imutils.perspective import four_point_transform
from imutils import grab_contours
from matplotlib.pyplot import contour
from regex import T
from skimage.segmentation import clear_border
import tkinter
from tkinter import messagebox

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


# #############################################################################
# Constantes
# #############################################################################
OP_BORDERS_CANNY = 0                 # Bordes mediante el filtro de Canny
OP_BORDERS_SOBEL = 1                 # Bordes mediante el filtro de Sobel
OP_BORDERS_THRESHOLD = 2             # Bordes mediante umbralización básica
OP_BORDERS_ADAPTATIVE_THRESHOLD = 3  # Bordes mediante umbralización adaptativa

SOBEL_X = 0
SOBEL_Y = 1

WAIT_KEY = False

# #############################################################################
# Inicialización opciones
# #############################################################################
op_borders = OP_BORDERS_CANNY
tDelay = 100

# #############################################################################
# Excepciones
# #############################################################################


class SudokuError(Exception):
    """Clase de excepción de la que derivarán todas las excepciones. """
    pass


class BoardError(SudokuError):
    """Excepcion que se lanza cuando no se encuentra el tablero. """
    pass


class CellsError(SudokuError):
    """Excepcion que se lanza cuando no se encuentran las celdas. """
    pass


# #############################################################################
# Métodos funcionales
# #############################################################################
def show_window(name, image, size=640, force_square=False):
    """ Muestra una ventana con la imagen
    @param name: Nombre de la ventana
    @param image: Imagen a mostrar
    @param size: Tamaño de la ventana (Ancho fijo)
    @param force_square: Si es True, asegura que la imagen sea cuadrada
    """
    # Escalar imagen a tamaño fijo
    scale_percent = size / image.shape[1]
    # Calcular ancho y alto de la imagen
    width = int(image.shape[1] * scale_percent)
    height = width if force_square else int(image.shape[0] * scale_percent)
    dim = (width, height)
    # resize image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow(name, resized)
    cv2.namedWindow(name, cv2.INTER_LINEAR)


def canny_filter(img, min_thr=30, max_thr=50):
    """ Aplica el filtro de Canny en la imagen y la devuelve
    normalizada.
    @param img: Imagen a filtrar
    """
    # Filtro de Canny. Los dos números son: nivel inferior, nivel superior
    canny = cv2.Canny(img, min_thr, max_thr)
    return cv2.normalize(canny, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def filtro_sobel(image, type=SOBEL_X):
    if type == SOBEL_X:
        kernel = np.array([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]])
    elif type == SOBEL_Y:
        kernel = np.array([[-1, -2, -1],
                           [0,  0,  0],
                           [1,  2,  1]])
    # Aplicamos el filtro de Sobel
    sobel = cv2.filter2D(image, -1, kernel)
    return sobel


def gaussian_filter(image, ksize=5):
    """ Aplica el filtro de Gaussiano en la imagen.
    @param image: Imagen a filtrar
    @param ksize: Tamaño del kernel    
    \nSe asegura que el tamaño del kernel sea impar.
    """
    size = (ksize//2)*2+1
    return cv2.GaussianBlur(image, (size, size), 3)


def umbralizacion_adaptativa(image, type=cv2.THRESH_BINARY, vecinos=11, c_substract=2):
    """ Aplica el filtro de umbralización adaptativa en la imagen, de modo que solo tienen en cuenta los pixels vecinos.
    @param image: Imagen a umbralizar
    @param vecinos: Número de vecinos a considerar
    @param c_substract: Constante restada de la media o de la media ponderada 
    \nSe asegura que el tamaño del kernel sea impar.
    """
    # · cv2.ADAPTIVE_THRESH_GAUSSIAN_C:
    #   El valor del umbral es una suma ponderada del bloque de vecinos
    #   (vecinos x vecinos) menos C.
    # · cv2.ADAPTIVE_THRESH_MEAN_C:
    #   El valor del umbral es la media de los vecions del bloque
    #   (vecinos x vecinos) menos C
    size = (vecinos//2)*2+1
    umbral = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   type, size, c_substract)
    return clear_border(umbral)


def umbralizacion(image, thr=50, type=cv2.THRESH_BINARY_INV):
    # Aplicamos un umbral
    umbral = cv2.threshold(image, thr, 255, type)[1]
    return clear_border(umbral)


def dilate_image(image):
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=2)
    return dilated


# TODO: BORAR no se usa
def find_contours(image):
    # Buscamos contorno en la imagen
    # contornos = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contornos = sorted(contornos, key=cv2.contourArea, reverse=True)
    # Recorremos todos los contornos encontrados
    contornos = cv2.findContours(
        image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contornos = grab_contours(contornos)
    # Eliminamos los contornos más pequeños
    # for c in cont:
    #     # if cv2.contourArea(c) < 500:
    #     #     continue

    #     # Obtenemos el bounds del contorno, el rectángulo mayor que engloba al contorno
    #     (x, y, w, h) = cv2.boundingRect(c)
    #     # Dibujamos el rectángulo del bounds
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return contornos


def draw_board(image, rectangle, board):
    """ Dibuja en la imagen el rectangulo que contiene el tablero y el tablero
    @param image: Imagen sobre la que dibujar 
    @param rectangle: Rectangulo que contiene el tablero
    @param board: Tablero a dibujar
    \n@return: Devuelve copia de la imagen con el tablero dibujado
    """
    _img = image.copy()
    # Dibujamos el rectángulo del bounds
    (x, y, w, h) = rectangle
    cv2.rectangle(_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Dibujamos el tablero
    cv2.drawContours(_img, [board], -1, (0, 0, 255), 2)
    return _img


def get_tablero(img_borders):
    # Buscamos contorno en la imagen
    contornos, hierarchy = cv2.findContours(
        img_borders, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Ordenamos para quedarnos con el mayor
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)
    contorno_mayor = contornos[0]

    # Obtenemos el bounds del contorno, el rectángulo mayor que engloba al contorno
    bounds = cv2.boundingRect(contorno_mayor)

    # Obtenemos el poligono del contorno ajustado. Si esta inclinado nos permitirá encuadrarlo
    perimetro = cv2.arcLength(contorno_mayor, True)
    sudoku_box = cv2.approxPolyDP(contorno_mayor, 0.02 * perimetro, True)
    if len(sudoku_box) != 4:
        raise BoardError("El tablero encontrado no es un cuadrado")

    return bounds, sudoku_box


def transform_board(image, poligon_board):
    # Transformacion para ajustar a ventana
    return four_point_transform(image, poligon_board.reshape(4, 2))


def tesseract_error(e):
    print(e)
    print("Error al procesar la celda")
    window = tkinter.Tk()
    window.wm_withdraw()
    messagebox.showerror('Tesseract no encontrado', str(e) +
                         '\n\nPuede descargarlo aqui:\n\n' +
                         'https://tesseract-ocr.github.io/tessdoc/Installation.html')
    webbrowser.open(
        'https://tesseract-ocr.github.io/tessdoc/Installation.html')
    window.destroy()
    exit(0)


# #############################################################################
# Leer imagen
# #############################################################################
img_original = cv2.imread(".\img\sudoku_01.jpg", cv2.IMREAD_UNCHANGED)
img_original = cv2.imread(".\img\opencv_sudoku_puzzle_outline.png",
                          cv2.IMREAD_UNCHANGED)
img_height, img_width = img_original.shape[:2]

# Mostramos la imagen original
show_window("Sudoku", img_original)
cv2.waitKey(tDelay)

# Convertir imagen a escala de grises
img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
show_window("Sudoku", img_gray)
cv2.waitKey(tDelay)

# Aplicar filtro gausiano para eliminar ruido
img_denoise = gaussian_filter(img_gray, 5)
show_window("Sudoku", img_denoise)
cv2.waitKey(tDelay)

# Detección de bordes
if op_borders == OP_BORDERS_CANNY:                   # Umbralización o Canny
    img_borders = canny_filter(img_denoise, min_thr=15, max_thr=25)
elif op_borders == OP_BORDERS_ADAPTATIVE_THRESHOLD:  # Umbralización o Canny
    img_borders = umbralizacion_adaptativa(
        img_denoise, cv2.THRESH_BINARY_INV, 25, 5)
elif op_borders == OP_BORDERS_THRESHOLD:             # Umbralización o Canny
    img_borders = umbralizacion(img_denoise, 160, cv2.THRESH_BINARY_INV)
elif op_borders == OP_BORDERS_SOBEL:                 # Umbralización o Sobel TODO: Pendiente de implementar
    pass
show_window("Sudoku", img_borders)
cv2.waitKey(tDelay)

# Seleccionar tablero
rectangle, board = get_tablero(img_borders)
img_board = draw_board(img_original, rectangle, board)
show_window("Sudoku", img_board)
cv2.waitKey(tDelay)

# Focus en el tablero
img_orig_board = transform_board(img_original, board)
img_gray_board = transform_board(img_gray, board)
show_window("Sudoku board ORIGINAL", img_orig_board, force_square=True)
show_window("Sudoku board PROCESSED", img_gray_board, force_square=True)
cv2.waitKey(tDelay)


# Detección de bordes
if op_borders == OP_BORDERS_CANNY:                   # Umbralización o Canny
    img_borders = canny_filter(img_gray_board, min_thr=15, max_thr=25)
elif op_borders == OP_BORDERS_ADAPTATIVE_THRESHOLD:  # Umbralización o Canny
    img_borders = umbralizacion_adaptativa(
        img_gray_board, cv2.THRESH_BINARY_INV, 25, 5)
elif op_borders == OP_BORDERS_THRESHOLD:             # Umbralización o Canny
    img_borders = umbralizacion(img_gray_board, 160, cv2.THRESH_BINARY_INV)
elif op_borders == OP_BORDERS_SOBEL:                 # Umbralización o Sobel TODO: Pendiente de implementar
    pass
show_window("Sudoku board PROCESSED", img_borders, force_square=True)
cv2.waitKey(tDelay)


# Deteccion de lineas
lines = cv2.HoughLines(img_borders, 1, np.pi/2, int(img_height*0.2))
for line in lines:
    rho, theta = line[0]
    v = np.cos(theta), np.sin(theta)
    p0 = (int(v[0] * rho), int(v[1] * rho))
    p1 = (int(p0[0]), int(p0[1]))
    p2 = (int(p0[0] + img_width * v[1]), int(p0[1] + img_height * v[0]))
    cv2.line(img_borders, p1, p2, (255, 255, 255), 2)
    cv2.line(img_orig_board, p1, p2, (0, 0, 255), 2)
    show_window("Sudoku board PROCESSED", img_borders, force_square=True)
    show_window("Sudoku board ORIGINAL", img_orig_board, force_square=True)
    cv2.waitKey(tDelay//3)

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

min_area = img_orig_board.shape[0] * img_orig_board.shape[1] * 0.6 / (9*9)
max_area = img_orig_board.shape[0] * img_orig_board.shape[1] / (9*9)
cells = []
for c in contornos[0]:
    if min_area < cv2.contourArea(c) and cv2.contourArea(c) < max_area:
        # Obtenemos el rectángulo que engloba al contorno
        (x, y, w, h) = cv2.boundingRect(c)
        cells.append((x, y, w, h))
        # cv2.drawContours(img_orig_board, [c], -1, (0, 255, 0), 2)
        # show_window("Sudoku board ORIGINAL", img_orig_board, force_square=True)
        # cv2.waitKey(int(tDelay/10))

# Comprobacion del numero de celdas encontradas
if len(cells) != 9*9:
    raise CellsError("No se encontraron todos los contornos de las celdas")

# Ordenar las celdas
cells = sorted(cells, key=lambda x: x[1])
for i in range(9):
    cells[i*9:i*9+9] = sorted(cells[i*9:i*9+9], key=lambda x: x[0])

# Dibujar las celdas
for c in cells:
    (x, y, w, h) = c
    cv2.rectangle(img_orig_board, (x, y), (x + w, y + h), (0, 255, 0), 3)
    show_window("Sudoku board ORIGINAL", img_orig_board, force_square=True)
    cv2.waitKey(int(tDelay/10))

# Leer el contenido de las celdas
for c in cells:
    x, y, w, h = c
    # Obtenemos la imagen de la celda
    m = int(w*0.05) # margen para evitar bordes
    points = np.array([[m+x, m+y], [x-m+w, m+y],
                      [x-m+w, y-m+h], [m+x, y-m+h]])
    cell = four_point_transform(img_gray_board, points)
    show_window("Sudoku cell", cell, size=200, force_square=True)
    cv2.waitKey(int(tDelay/10))
    cell = cv2.GaussianBlur(cell, (5, 5), 1)
    show_window("Sudoku cell", cell, size=200, force_square=True)
    cv2.waitKey(int(tDelay/10))
    cell = umbralizacion_adaptativa(cell, cv2.THRESH_BINARY_INV, 40, 5)
    cell = cv2.bitwise_not(cell)
    cell = cv2.GaussianBlur(cell, (7, 7), 1)
    show_window("Sudoku cell", cell, size=200, force_square=True)
    cv2.waitKey(int(tDelay/10))
    try:
        custom_config = r'--oem 3 --psm 6 outputbase digits'
        text = ocr.image_to_string(cell, config=custom_config)
    except Exception as e:
        tesseract_error(e)
    text = text[0] if text != '' else '0'
    print(text)
    show_window("Sudoku cell", cell, size=200, force_square=True)
    cv2.waitKey(int(tDelay/10))


# else:
#     cv2.drawContours(img_borders, [c], -1, (255, 0, 0), 2)
#     cv2.drawContours(img_orig_board, [c], -1, (255, 0, 0), 2)

# # approximate the contour
# peri = cv2.arcLength(c, True)
# approx = cv2.approxPolyDP(c, 0.02 * peri, True)
# cv2.drawContours(img_orig_board, [c], -1, (0, 255, 0), 2)
# show_window("Sudoku board ORIGINAL", img_orig_board, force_square=True)
# cv2.waitKey(tDelay)


cv2.destroyAllWindows()
# #################################################
# FIN
# #################################################
exit(0)

# Buscamos contorno en la imagen
contornos, hierarchy = cv2.findContours(
    sudoku_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Recorremos todos los contornos encontrados
for c in contornos:
    # Eliminamos los contornos más pequeños
    if cv2.contourArea(c) < 500:
        continue

    # Obtenemos el bounds del contorno, el rectángulo mayor que engloba al contorno
    (x, y, w, h) = cv2.boundingRect(c)
    # Dibujamos el rectángulo del bounds
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


cv2.waitKey(tDelay)


adp_img = umbralizacion_adaptativa(img.copy())
canny_img = canny_filter(img.copy())
sobel_img = filtro_sobel(img.copy(), SOBEL_X)
dil_img = dilate_image(sobel_img.copy())
gaus_img = gaussian_filter(img.copy())

# r = find_contours(dil_img)

show_window("Sudoku adp_img", adp_img)
show_window("Sudoku canny_img", canny_img)
show_window("Sudoku dil_img", dil_img)
show_window("Sudoku sobel_img", sobel_img)
show_window("Sudoku gaus_img", gaus_img)

# canny_filter(img)
# show_window("Sudoku", img)


cv2.waitKey(tDelay)
cv2.destroyAllWindows()
