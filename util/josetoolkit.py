# Importamos las librerías necesarias
from matplotlib import pyplot as plt

import cv2
import numpy as np
import webbrowser
from imutils.perspective import four_point_transform
from imutils import grab_contours
from skimage.segmentation import clear_border
import tkinter
from tkinter import messagebox


# #############################################################################
# Constantes
# #############################################################################
# OP_BORDERS_CANNY = 0                 # Bordes mediante el filtro de Canny
# OP_BORDERS_SOBEL = 1                 # Bordes mediante el filtro de Sobel
# OP_BORDERS_THRESHOLD = 2             # Bordes mediante umbralización básica
# OP_BORDERS_ADAPTATIVE_THRESHOLD = 3  # Bordes mediante umbralización adaptativa

SOBEL_X = 0
SOBEL_Y = 1

FIXED_WIDTH = 2560  # Ancho fijo de la imagen a escalar

# Imagen de entrada
DEF_SUDOKU_IMG = ".\img\sudoku_01.png"

# Ruta de salida
SUDOKU_OUT = ".\sudoku_out.png"

# #############################################################################
# Inicialización opciones
# #############################################################################
# op_borders = OP_BORDERS_CANNY
WAIT_DELAY = 10

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
def show_window(name, image, size=640, force_square=False, wait=-1):
    """ Muestra una ventana con la imagen
    @param name: Nombre de la ventana
    @param image: Imagen a mostrar
    @param size: Tamaño de la ventana (Ancho fijo)
    @param force_square: Si es True, asegura que la imagen sea cuadrada
    @param wait: Tiempo de espera antes de cerrar la ventana
    @return: Devuelve la imagen de entrada sin alterarla. Esto permite usar una 
    funcion de procesamiento directamente como parametro `image` de entrada y 
    guardar su resultado por la salida.
    """
    # Escalar imagen a tamaño fijo
    scale_factor = size / image.shape[1]
    # Calcular ancho y alto de la imagen
    width = int(image.shape[1] * scale_factor)
    height = width if force_square else int(image.shape[0] * scale_factor)
    dim = (width, height)
    # resize image
    resized = cv2.resize(image.copy(), dim, interpolation=cv2.INTER_AREA)
    cv2.imshow(name, resized)
    cv2.namedWindow(name, cv2.INTER_LINEAR)
    if wait != -1:
        cv2.waitKey(int(wait))
    return image


# #############################################################################
def resize_image(image):
    # Para reducir una imagen, generalmente se verá mejor con la interpolación
    # INTER_AREA, mientras que para agrandar una imagen, generalmente se verá
    # mejor con INTER_CUBIC (lento) o INTER_LINEAR (más rápido peor calidad).
    # Escalar imagen a ancho fijo
    scale_factor = FIXED_WIDTH / image.shape[1]
    # Calcular ancho y alto de la imagen
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    dim = (width, height)
    interpolation = cv2.INTER_CUBIC if scale_factor > 1 else cv2.INTER_AREA
    # resize image
    return cv2.resize(image, dim, interpolation=interpolation)


def show_image(path):
    img_original = show_window(
        "Sudoku",
        cv2.imread(path, cv2.IMREAD_UNCHANGED)
    )
    img_height, img_width = img_original.shape[:2]
    print("Dimensiones de la imagen: {}x{}".format(img_width, img_height))
    return img_original


def std_resize(image):
    img_resized = resize_image(image)
    img_height, img_width = img_resized.shape[:2]
    min_dim = min(img_height, img_width)
    print("Dimensiones de la imagen redimensionada: {}x{}".format(
        img_width, img_height))
    return img_resized, img_height, img_width, min_dim


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
    return cv2.GaussianBlur(image, (size, size), 0)


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
    umbral = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, type, size, c_substract)
    return cv2.bitwise_not(umbral)


def umbralizacion(image, thr=50, type=cv2.THRESH_BINARY_INV):
    # Aplicamos un umbral
    umbral = cv2.threshold(image, thr, 255, type)[1]
    return clear_border(umbral)


def dilate_image(image):
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=2)
    return dilated


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


def show_hist(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    max_value = np.max(hist)  # Valor máximo del histograma
    max_i = np.argmax(hist)  # Índice del valor máximo
    pixels = image.shape[0]*image.shape[1]
    # show_window("Histograma", hist, size=hist.shape[1])
    plt.plot(hist/pixels)
    plt.show()
    return hist, max_value, max_i


def ordenar_puntos(points):
    """
    Ordena los puntos en forma de cuadrado

    [p1, p2, 

     p3, p4]"""
    p = np.array(points)
    if len(p) != 4:
        raise BoardError("Se esperaban 4 puntos")
    # Ordenar las puntos
    up = sorted(p, key=lambda x: x[1])[:2]
    down = sorted(p, key=lambda x: x[1])[2:]
    p1, p2 = sorted(up, key=lambda x: x[0])
    p3, p4 = sorted(down, key=lambda x: x[0])
    return [p1[0], p1[1]], [p2[0], p2[1]], [p3[0], p3[1]], [p4[0], p4[1]]


def write_coords(images, name, point, colors):
    """
    Escribe un nombre junto a las coordenadas de un punto en todas las imagenes de entrada.
    @param images: Lista de imagenes
    @param name: Nombre del punto
    @param point: Coordenadas del punto a escribir
    @param colors: Lista de colores
    """
    for i in range(len(images)):
        cv2.putText(
            img=images[i],
            text="{}:({}, {})".format(name, point[0], point[1]),
            org=point,
            fontFace=cv2.FONT_HERSHEY_TRIPLEX,
            fontScale=2,
            color=colors[i],
            thickness=2,
            lineType=cv2.LINE_AA)


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
