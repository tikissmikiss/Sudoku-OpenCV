import os
import threading

# import cv2

files = [
    ".\img\sudoku_01.png",
    ".\img\sudoku_02.jpg",
    ".\img\sudoku_03.jpg",
    ".\img\sudoku_04.jpg",
    ".\img\sudoku_05.jpg",
    ".\img\sudoku_06.jpg",
    ".\img\sudoku_07.jpg",
    ".\img\sudoku_08.jpg",
    ".\img\sudoku_09.jpg",
    ".\img\sudoku_10.jpg",
    ]

def run_thread(file, i):
    os.system("python sudoku.py -w 10 -o .\solutions\solucion{}.png -i {}".format(i+1, file))

threads = []
for i, file in enumerate(files):
    t = threading.Thread(target=run_thread, args=(file, i))
    threads.append(t)
    t.start()
    print("Thread {} started".format(i+1))
for t in threads:
    t.join()

print("Threads joined")
