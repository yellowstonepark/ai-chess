import chess
import numpy as np
import torch
import torch.nn as nn

ACTION_SPACE_SIZE = 64 * 64  # 4096 possible moves

# Define the piece values
PIECE_VALUES = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 100}

# Neural Network for DQN
class ChessCNN(nn.Module):
    def __init__(self, action_size=ACTION_SPACE_SIZE):
        super(ChessCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, action_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Function to encode a move into an integer
def encode_move(move):
    return move.from_square * 64 + move.to_square

# Function to decode an integer back into a move
def decode_move(encoded_move, legal_moves):
    from_square = encoded_move // 64
    to_square = encoded_move % 64
    for move in legal_moves:
        if move.from_square == from_square and move.to_square == to_square:
            return move
    return None

# Function to encode the board state
def encode_board(board):
    # Create a 8x8 numpy array to represent the board
    board_array = np.zeros((1, 8, 8), dtype=np.float32)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            value = PIECE_VALUES.get(piece.symbol().lower(), 0)
            if piece.color == chess.WHITE:
                board_array[0, i // 8, i % 8] = value
            else:
                board_array[0, i // 8, i % 8] = -value
    return board_array

# Load the model
model = ChessCNN(ACTION_SPACE_SIZE)
model.load_state_dict(torch.load('chess_model.pth'))
model.eval()  # Set the model to evaluation mode

# Function to choose a move with the model
def choose_move_with_model(board, model):
    state = encode_board(board)
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state_tensor)

    legal_moves = list(board.legal_moves)
    best_move = None
    best_q_value = -float('inf')

    for move in legal_moves:
        encoded_move = encode_move(move)
        if encoded_move < ACTION_SPACE_SIZE:
            q_value = q_values[0, encoded_move].item()
            if q_value > best_q_value:
                best_q_value = q_value
                best_move = move

    return best_move

# Example of using the model in a game
board = chess.Board()
while not board.is_game_over():
    print(board)  # Print the current state of the board
    move = choose_move_with_model(board, model)
    print("Move:", move)
    if move:
        board.push(move)
    else:
        print("No valid moves found by the model")
        break

# After the game is over, print the result
game_result = board.result()
print("Game over. Result:", game_result)

if game_result == "1-0":
    print("White wins!")
elif game_result == "0-1":
    print("Black wins!")
else:
    print("The game is a draw.")