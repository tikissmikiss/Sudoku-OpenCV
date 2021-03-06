from argparse import ArgumentParser as AP

import cv2
import numpy as np
from cv2 import drawContours
from imutils.perspective import four_point_transform

import util.josetoolkit as jtk
from util.josetoolkit import DEF_SUDOKU_IMG, SUDOKU_OUT, WAIT_DELAY, intersect
from util.sudoku_solver import print_board, solve

# #############################################################################
# Menú del programa.
# #############################################################################

ap = AP()
ap.add_argument('-i', '--image', default=DEF_SUDOKU_IMG, required=False,
                help='Ruta de la imagen de entrada. Si se omite se usa "'+str(DEF_SUDOKU_IMG)+'".')
ap.add_argument('-w', '--wait-delay', default=WAIT_DELAY, required=False,
                help='Tiempo de espera entre etapas del procesado (mseg). Si es igual a 0, \
                    se espera indefinidamente. Pulsar una tecla fuerza el avance. \
                    Si se omite se usa ' + str(WAIT_DELAY) + '.')
ap.add_argument('-o', '--output', default=SUDOKU_OUT, required=False,
                help='Ruta de la imagen de salida. Si se omite se usa "'+str(SUDOKU_OUT)+'".')
args = vars(ap.parse_args())

DELAY = 10 if int(args['wait_delay']) in range(
    1, 10) else int(args['wait_delay'])
IMG_IN = args['image']
IMG_OUT = args['output']


# #############################################################################
# #############################################################################
# Leer y mostrar imagen
# #############################################################################
# #############################################################################

img_original = jtk.show_image(IMG_IN)

# Redimensionar imagen para trabajar siempre con el mismo ancho
img_resized, img_height, img_width, min_dim = jtk.std_resize(img_original)

# Mostrar lienzo en color y copiar imagen original
img_color = jtk.show_window("Sudoku", img_resized.copy(), wait=DELAY)


# #############################################################################
# #############################################################################
# Buscar tablero
# #############################################################################
# #############################################################################

# Convertir imagen a escala de grises
print("Convertir imagen a escala de grises")
img_gray = jtk.show_window(
    "PROCESS",
    cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY),
    wait=DELAY)

# Aplicar filtro gausiano para eliminar ruido
print("Aplicar filtro gausiano para eliminar ruido")
img_denoise = jtk.show_window(
    "PROCESS",
    jtk.gaussian_filter(img_gray, 7),
    wait=DELAY)

# Umbralización segun brillo medio de la imagen
print("Umbralización")
pixels = img_height*img_width
brightness = int(np.sum(img_denoise)/pixels)
thr = brightness * 0.8
img_denoise = jtk.show_window(
    "PROCESS",
    jtk.umbralizacion(img_denoise, thr=thr, type=cv2.THRESH_BINARY_INV),
    wait=DELAY)

# Detección de bordes
print("Detección de bordes")
img_borders = jtk.show_window(
    "PROCESS",
    jtk.canny_filter(img_denoise, min_thr=15, max_thr=30),
    wait=DELAY)

# Dilatar bordes
print("Dilatar bordes")
img_dilate = jtk.show_window(
    "PROCESS",
    cv2.dilate(
        img_borders,
        cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    ),
    wait=DELAY)

# Erosionar bordes
print("Erosionar bordes")
img_erode = jtk.show_window(
    "PROCESS",
    cv2.erode(
        img_dilate,
        cv2.getStructuringElement(cv2.MORPH_RECT, (24, 24))
    ),
    wait=DELAY)


# #############################################################################
# Deteccion de lineas
# #############################################################################

# Crear matriz de ceros para mascara de bordes
print("Generar base para mascara de bordes")
mask = np.zeros((img_height, img_width), np.uint8)
jtk.show_window("Mask", mask, wait=DELAY)

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
jtk.show_window("PROCESS", img_color)
jtk.show_window("Mask", mask, wait=DELAY)

# Agrupar lineas dilatando para juntar lineas cercanas y erosionando de nuevo
print("Eliminar lineas finas")
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
mask = cv2.dilate(mask, kernel, iterations=1)
jtk.show_window("Mask", mask, wait=DELAY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
mask = cv2.erode(mask, kernel, iterations=1)
jtk.show_window("Mask", mask, wait=DELAY)

# Contornos
contornos, h = cv2.findContours(
    mask,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE)
contorno = max(contornos, key=cv2.contourArea)
perimetro = cv2.arcLength(contorno, True)
poligon = cv2.approxPolyDP(contorno, 0.1 * perimetro, closed=True)


# #############################################################################
# Extraer y dibujar puntos
# #############################################################################

# Extraer puntos
points = []
for i in range(0, len(poligon)):
    x, y = poligon[i][0]
    points.append((x, y))
    cv2.circle(img_color, (x, y), 20, (0, 0, 255), 10)
jtk.show_window("PROCESS", img_color)
# Ordenar los puntos
points = jtk.ordenar_puntos(points)
# Dibujar coordenadas puntos
i = 0
print(points)
for p in points:
    i += 1
    jtk.write_coords([mask, img_color], "P{}".format(i),
                     p, [(128), (255, 0, 0)])
x, y = intersect([points[0], points[3]], [points[1], points[2]])
cv2.circle(img_color, (x, y), 20, (0, 0, 255), -1)
jtk.write_coords([mask, img_color], "Center", (x, y), [(128), (255, 0, 0)])
jtk.show_window("Mask", mask)
jtk.show_window("PROCESS", img_color, wait=DELAY)

# Transformacion de encuadre
h, w = img_height, img_width
p_orig = points
p_dest = [[0, 0], [w, 0], [w, h], [0, h]]
p_orig = jtk.ordenar_puntos(p_orig)
p_dest = jtk.ordenar_puntos(p_dest)
print("\nPuntos origen:\n", p_orig)
print("\nPuntos destino:\n", p_dest)
transform_matrix = cv2.getPerspectiveTransform(
    np.float32(p_orig), np.float32(p_dest))
img_inframe = cv2.warpPerspective(img_resized, transform_matrix, (w, h))
jtk.show_window("PROCESS", img_inframe, wait=DELAY)


# #############################################################################
# #############################################################################
# Leer sudoku
# #############################################################################
# #############################################################################

# Convertir imagen a escala de grises
print("Convertir imagen a escala de grises")
img_process = jtk.show_window(
    "PROCESS",
    cv2.cvtColor(img_inframe, cv2.COLOR_BGR2GRAY),
    wait=DELAY)

# Aplicar filtro gausiano para eliminar ruido
print("Aplicar filtro gausiano para eliminar ruido")
img_process = jtk.show_window(
    "PROCESS",
    jtk.gaussian_filter(img_process, 25),
    wait=DELAY)

# Umbralizacion adaptativa
print("Umbralizacion adaptativa")
img_process = jtk.show_window(
    "PROCESS",
    jtk.umbralizacion_adaptativa(img_process, cv2.THRESH_BINARY, 150, 20),
    wait=DELAY)

# # Umbralización segun brillo medio de la imagen
# print("Umbralización")
# pixels = img_height*img_width
# brightness = int(np.sum(img_process)/pixels)
# thr = brightness * 0.7
# img_process = jtk.show_window(
#     "PROCESS",
#     jtk.umbralizacion(img_process, thr=thr, type=cv2.THRESH_BINARY_INV),
#     wait=DELAY)


# #############################################################################
# Dubujar lineas teoricas
# #############################################################################

m = 5
w = img_width//9
h = img_height//9
img_process[:m, :] = 255
img_process[-m:, :] = 255
img_process[:, :m] = 255
img_process[:, -m:] = 255
for i in range(10):
    img_process[i*h-m:i*h+m:, :] = 255
    img_process[:, i*w-m:i*w+m:] = 255
mask = jtk.show_window("Mask", img_process, wait=DELAY)

# Contornos
# There are four types in retrieval mode in OpenCV.
# - cv2.RETR_LIST → Retrieve all contours
# - cv2.RETR_EXTERNAL → Retrieves external or outer contours only
# - cv2.RETR_COMP → Retrieves all in a 2-level hierarchy
# - cv2.RETR_TREE → Retrieves all in the full hierarchy
# Hierarchy is stored in the following format[next, previous, First child, parent].
# Buscar contornos
contornos, hierarchy = cv2.findContours(
    mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
hierarchy = hierarchy[0]
drawContours(img_inframe, contornos, -1, (0, 255, 0), 2)
jtk.show_window("PROCESS", img_inframe, wait=DELAY)


# #############################################################################
# Buscar celdas
# #############################################################################

# Pintar contornos de primer nivel (Nivel Raiz = 0)
next, previous, child, parent = 0, 1, 2, 3
celdas, numero = [], []
root, i = contornos[0], hierarchy[0, child]
drawContours(img_inframe, contornos, 0, (0, 255, 0), 2)
jtk.show_window("PROCESS", img_inframe, wait=1)
drawContours(img_inframe, contornos, 0, (0), -1)
drawContours(mask, contornos, 0, (0), -1)
jtk.show_window("Mask", mask)
jtk.show_window("PROCESS", img_inframe, wait=1)

# Filtrar por area
mean = 0
max_area = img_width * img_height / 81
min_area = max_area / 2
celdas = [i for i in range(len(contornos))
            if min_area < cv2.contourArea(contornos[i]) 
            and cv2.contourArea(contornos[i]) < max_area]

# Contornos celdas
for i in celdas:
    drawContours(img_inframe, contornos, i, (255, 0, 0), -1)
    drawContours(mask, contornos, i, (0), -1)
    jtk.show_window("Mask", mask)
    jtk.show_window("PROCESS", img_inframe,wait=1)
    if hierarchy[i, child] != -1:
        h = hierarchy[i, child]
        c = []
        while h != -1:
            if cv2.contourArea(contornos[h]) > mean*0.4:
                c.append(h)
            h = hierarchy[h, next]
        if len(c)>0:
            numero.append([x for x in c if cv2.contourArea(contornos[x]) == max([cv2.contourArea(contornos[j]) for j in c])][0])
            mean = np.mean([cv2.contourArea(contornos[j]) for j in numero])

# Contornos numeros
for h in numero:
    if cv2.contourArea(contornos[h]) > mean*0.5:
        drawContours(img_inframe, contornos,h, (255, 255, 255), -1)
        drawContours(mask, contornos, h, (255), -1)
        jtk.show_window("Mask", mask)
        jtk.show_window("PROCESS", img_inframe, wait=1)
    if hierarchy[h, child] != -1:
        j = hierarchy[h, child]
        while j != -1:
            drawContours(img_inframe, contornos, j, (255, 0, 0), -1)
            drawContours(mask, contornos, j, (0), -1)
            jtk.show_window("Mask", mask)
            jtk.show_window("PROCESS", img_inframe, wait=1)
            j = hierarchy[j, next]

# Rectangulos celdas
cells = [cv2.boundingRect(contornos[i]) for i in celdas]

# Comprobacion del numero de celdas encontradas
if len(cells) != 81:
    raise jtk.CellsError("No se encontraron todos los contornos de las celdas")

# Ordenar las celdas
cells = sorted(cells, key=lambda x: x[1])
for i in range(9):
    cells[i*9:i*9+9] = sorted(cells[i*9:i*9+9], key=lambda x: x[0])

# Dibujar las celdas
for c in cells:
    (x, y, w, h) = c
    cv2.rectangle(img_inframe, (x, y), (x + w, y + h), (0, 0, 255), 3)
jtk.show_window("PROCESS", img_inframe, wait=1)


# ###############################################################################
# Leer el contenido de las celdas
# ###############################################################################

window_cell = "Sudoku cell"
board_data = []

for c in cells:
    x, y, w, h = c
    # Obtenemos la imagen de la celda
    m = int(w*0.1)  # margen para evitar bordes
    m = 0  # margen para evitar bordes
    points = np.array([[m+x, m+y], [x-m+w, m+y], [x-m+w, y-m+h], [m+x, y-m+h]])
    cell = four_point_transform(mask, points)
    jtk.show_window(window_cell, cell, size=200, wait=1)
    text = jtk.ocr(cell)
    if text != '0':
        jtk.draw_number(img_inframe, c, text, (0, 0, 255))
    jtk.show_window("PROCESS", img_inframe)
    board_data.append(text)
    jtk.show_window(window_cell, cell, size=200, wait=1)
cv2.destroyWindow(window_cell)
board_data = np.array(board_data).reshape(9, 9).astype(int)
print()
print(board_data)
cv2.waitKey(DELAY)


# ###############################################################################
# Resolver sudoku
# ###############################################################################

print_board('Problem', board_data)
cv2.waitKey(DELAY)
solution = solve(board_data)
print_board('Solution', solution)


# ###############################################################################
# Imprimir solucion
# ###############################################################################

# Crear mascara de ceros para superponer  resultado
mask = np.zeros((img_height, img_width, img_resized.shape[2]), np.uint8)
mask = np.zeros((img_height, img_width), np.uint8)
jtk.show_window("Mask", mask, wait=DELAY)
# Transformacion de restablecimiento de encuadre
p_orig, p_dest = p_dest, p_orig
transform_matrix = cv2.getPerspectiveTransform(
    np.float32(p_orig), np.float32(p_dest))

for r in range(9):
    for c in range(9):
        if board_data[r, c] == 0:
            jtk.draw_number(
                img_inframe, cells[r*9+c], solution[r][c], (14, 168, 12))
            jtk.draw_number(mask, cells[r*9+c], solution[r][c], (255))
            m_over = cv2.warpPerspective(
                mask, transform_matrix, (img_width, img_height))
            iChannels = cv2.split(img_resized)
            iChannels[0][m_over == 255] = 14
            iChannels[1][m_over == 255] = 168
            iChannels[2][m_over == 255] = 12
            img_solution = cv2.merge(iChannels)
            jtk.show_window("PROCESS", img_inframe)
            jtk.show_window("Mask", m_over)
            jtk.show_window("Sudoku", img_solution, wait=100)

# Guardan solucion en archivo
cv2.imwrite(IMG_OUT, img_solution)


# #################################################
# FIN
# #################################################

cv2.waitKey(5000)
cv2.destroyWindow("Mask")
cv2.destroyWindow("PROCESS")

cv2.putText(
    img_solution,
    'Pulsar cualquier tecla para salir !!!',
    (350, img_height-50),
    cv2.FONT_HERSHEY_COMPLEX,
    fontScale=3,
    color=(0, 0, 255),
    thickness=5)
jtk.show_window("Sudoku", img_solution, wait=0)

cv2.destroyAllWindows()
exit(0)
