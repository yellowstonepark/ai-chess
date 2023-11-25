import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTION_SPACE_SIZE = 4672

# Define the piece values
PIECE_VALUES = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 100}

class EnhancedChessCNN(nn.Module):
    def __init__(self, action_size):
        super(EnhancedChessCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(512, action_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x

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

# Load White's model
model_white = EnhancedChessCNN(ACTION_SPACE_SIZE)
model_white.load_state_dict(torch.load('chess_model100000_3.pth'))
model_white.eval()

# Load Black's model
model_black = EnhancedChessCNN(ACTION_SPACE_SIZE)
model_black.load_state_dict(torch.load('chess_model100000_5.pth'))
model_black.eval()

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

def get_user_move(board):
    """
    Prompt the user for a move and return it.
    """
    print(board)
    move = input("Enter your move: ")
    try:
        chess_move = chess.Move.from_uci(move)
        if chess_move in board.legal_moves:
            return chess_move
        else:
            print("Illegal move. Please try again.")
            return get_user_move(board)
    except ValueError:
        print("Invalid move format. Please try again.")
        return get_user_move(board)

# Function to play a game against the model
def play_against_model(model_white, model_black):
    board = chess.Board()

    while not board.is_game_over():
        if board.turn == chess.BLACK:
            # Model's turn (BLACK)
            move = choose_move_with_model(board, model_black)
            print("Model's move:", move)
        else:
            # Your turn (Black)
            move = get_user_move(board)

        if move:
            board.push(move)
        else:
            print("No valid moves found.")
            break

    print(board)
    game_result = board.result()
    print("Game over. Result:", game_result)

    if game_result == "1-0":
        print("White wins!")
    elif game_result == "0-1":
        print("Black wins!")
    else:
        print("The game is a draw.")

# Play against the model
play_against_model(model_white, model_black)
