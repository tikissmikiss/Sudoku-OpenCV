import time

board = [[5, 3, 0, 0, 7, 0, 0, 0, 0],
         [6, 0, 0, 1, 9, 5, 0, 0, 0],
         [0, 9, 8, 0, 0, 0, 0, 6, 0],
         [8, 0, 0, 0, 6, 0, 0, 0, 3],
         [4, 0, 0, 8, 0, 3, 0, 0, 1],
         [7, 0, 0, 0, 2, 0, 0, 0, 6],
         [0, 6, 0, 0, 0, 0, 2, 8, 0],
         [0, 0, 0, 4, 1, 9, 0, 0, 5],
         [0, 0, 0, 0, 8, 0, 0, 7, 9]]


def print_board(title, board):
    """ Print the board state with a heading title """
    print('\r\n', ' '*(16-len(title)//2), title)
    for r in range(len(board)):
        if (r == 0): print('╔', '═'*3, '╤', '═'*3, '╤', '═'*3, '╦', '═'*3, '╤', '═'*3, '╤',
                  '═'*3, '╦', '═'*3, '╤', '═'*3, '╤', '═'*3, '╗', end='\r\n║', sep='')
        elif (r % 3 == 0): print('\r\n╠', '═'*3, '╪', '═'*3, '╪', '═'*3, '╬', '═'*3, '╪', '═'*3, '╪',
                  '═'*3, '╬', '═'*3, '╪', '═'*3, '╪', '═'*3, '╣', end='\r\n║', sep='')
        else: print('\r\n╟', '─'*3, '┼', '─'*3, '┼', '─'*3, '╫', '─'*3, '┼', '─'*3, '┼',
                  '─'*3, '╫', '─'*3, '┼', '─'*3, '┼', '─'*3, '╢', end='\r\n║', sep='')
        for c in range(len(board[r])): 
            if (board[r][c] == 0): print('   ', end='')
            else: print('', board[r][c], '',  end='')
            if (c % 3 == 2): print('║', end='')
            else: print('│', end='')
    print('\r\n╚', '═'*3, '╧', '═'*3, '╧', '═'*3, '╩', '═'*3, '╧', '═' *
          3, '╧', '═'*3, '╩', '═'*3, '╧', '═'*3, '╧', '═'*3, '╝', sep='')


def is_valid(board, r, c, n):
    """ Devuelve True si el número n es válido en la celda (r,c) """
    if (n in board[r]): return False
    if (n in [board[r][c] for r in range(len(board))]): return False
    r_init, c_init = (r//3)*3, (c//3)*3
    if (n in [board[r][c] for r in range(r_init, r_init+3) for c in range(c_init, c_init+3)]): return False
    return True


def total_solution(sol):
    """ Devuelve True si el Sudoku está resuelto """
    for r in range(len(sol)):
        for c in range(len(sol[r])):
            if sol[r][c] == 0: return False
    return True


def successors(sol):
    """ Devuelve la expansion de un nodo con todos los valores validos de una celda """
    for r in range(len(sol)):
        for c in range(len(sol[r])):
            if (not sol[r][c]):
                for n in range(1, 10):
                    if (is_valid(sol, r, c, n)):
                        sol[r][c] = n
                        yield sol # Devuelve un sucesor
                sol[r][c] = 0 # Restablece el valor original
                return # Fin generador de sucesores
    return None # No existen sucesores


def solve(board):
    """ Use backtracking to solve the puzzle in the board """
    if total_solution(board): return True
    for s in successors(board): 
        board = solve(s)
        if board != False: return True
    return False


print_board('Problem', board)
start = time.time()
solve(board)
end = time.time()
print_board('Solution', board)
print('Time to solve:', end-start, "\r\n")

