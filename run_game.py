import chess
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def simulate_random_game(_):
    board = chess.Board()
    while not board.is_game_over(claim_draw=True):
        move = random.choice(list(board.legal_moves))
        board.push(move)
    return board.result()

if __name__ == '__main__':
    num_cores = 8
    with Pool(num_cores) as p:
        results = list(tqdm(p.imap(simulate_random_game, range(1000)), total=1000))
    for result in results:
        print(result)
