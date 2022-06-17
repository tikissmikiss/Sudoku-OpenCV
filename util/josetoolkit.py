# Importamos las librerías necesarias
import os
import threading
import tkinter
import webbrowser
from tkinter import messagebox

import cv2
import numpy as np
from skimage.segmentation import clear_border

try:
    import pytesseract as ts
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

SOBEL_X = 0
SOBEL_Y = 1

FIXED_WIDTH = 2560  # Ancho fijo de la imagen a escalar

# #############################################################################
# Inicialización opciones
# #############################################################################

# Imagen de entrada por defecto
DEF_SUDOKU_IMG = ".\img\sudoku_01.png"

# Ruta de salida por defecto
SUDOKU_OUT = ".\sudoku_out.png"

# Tiempo de espera por defecto
WAIT_DELAY = 1000

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


def intersect(a, b):
    """
    Calcula la intersección de dos rectas
    @param a: Recta 1 (2 puntos (x, y))
    @param b: Recta 2 (2 puntos (x, y))
    @return: Coordenadas de la intersección
    """
    x1, y1 = a[0]
    x2, y2 = a[1]
    x3, y3 = b[0]
    x4, y4 = b[1]
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denom == 0:
        return None
    ma = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    mb = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    x = x1 + ma * (x2 - x1)
    y = y1 + ma * (y2 - y1)
    return (int(x), int(y))


def ocr(cell):
    """ TESSERACT
    OCR Engine modes (--oem):
        0    Legacy engine only.
        1    Neural nets LSTM engine only.
        2    Legacy + LSTM engines.
        3    Default, based on what is available.
    Tesseract Page segmentation modes (--psm):
        0    Orientation and script detection (OSD) only.
        1    Automatic page segmentation with OSD.
        2    Automatic page segmentation, but no OSD, or OCR.
        3    Fully automatic page segmentation, but no OSD. (Default)
        4    Assume a single column of text of variable sizes.
        5    Assume a single uniform block of vertically aligned text.
        6    Assume a single uniform block of text.
        7    Treat the image as a single text line.
        8    Treat the image as a single word.
        9    Treat the image as a single word in a circle.
        10    Treat the image as a single character.
        11    Sparse text. Find as much text as possible in no particular order.
        12    Sparse text with OSD.
        13    Raw line. Treat the image as a single text line,
                bypassing hacks that are Tesseract-specific.
    """
    reads = []
    cnfgs = [
        r'--psm 6 -c tessedit_char_whitelist=0123456789',
        r'--psm 8 -c tessedit_char_whitelist=0123456789',
        r'--psm 10 -c tessedit_char_whitelist=0123456789',
    ]
    imgs = process_cell(cell)
    try:
        thrds = []
        for i in imgs:
            show_window('Sudoku cell', i, size=200, wait=1)
            for c in cnfgs:
                thrds.append(threading.Thread(
                    target=ocr_thread, args=(i, c, reads)))
                thrds[-1].start()
            # Esperamos que acaben todos los hilos
            for i in range(len(thrds)):
                thrds[i].join()
    except Exception as e:
        tesseract_error(e)
    return best(reads)


def ocr_thread(img, cnfg, reads):
    try:
        reads.append(ts.image_to_string(img, config=cnfg))
    except Exception as e:
        tesseract_error(e)


def process_cell(cell):
    pixels = cell.shape[0]*cell.shape[1]
    brightness = int(np.sum(cell)/pixels)
    imgs = [cell]
    if brightness < 3:
        return []
    for p in range(90, 211, 15):
        thr = p
        c = gaussian_filter(cell, 25)
        c = umbralizacion(c, thr=thr, type=cv2.THRESH_BINARY)
        c = gaussian_filter(c, 7)
        # show_window('Sudoku cell', c, size=200, wait=1)
        imgs.append(c)
    return imgs


def best(values):
    matches = ['1\n', '2\n', '3\n', '4\n', '5\n', '6\n', '7\n', '8\n', '9\n']
    reads = [v[0] for v in values if v in matches]
    print("OCR: {}".format(reads))
    if len(reads) != 0:
        dic = {'1': 0, '2': 0, '3': 0, '4': 0,
               '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
        for r in reads:
            dic[r] += 1
        return max(dic, key=dic.get)
    else:
        return '0'
