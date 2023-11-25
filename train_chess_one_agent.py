import chess
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

# Define the piece values
PIECE_VALUES = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 100}

ACTION_SPACE_SIZE = 4096  # Adjust as needed

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

# Chess environment
class ChessEnv:
    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        self.board.reset()
        return encode_board(self.board)

    def step(self, move):
        taken_piece = self.board.piece_at(move.to_square)
        self.board.push(move)
        reward = 0
        if taken_piece:
            reward = PIECE_VALUES.get(taken_piece.symbol().lower(), 0)
        done = self.board.is_game_over()
        return encode_board(self.board), reward, done

    def legal_moves(self):
        return list(self.board.legal_moves)

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

# DQN Agent
class DQNAgent:
    def __init__(self, action_size):
        self.model = ChessCNN(action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.epsilon = 0.9
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99  # Discount factor

    def select_action(self, state, legal_moves):
        if random.random() <= self.epsilon:
            # Choose a random legal move within the action space limit
            filtered_legal_moves = [move for move in legal_moves if encode_move(move) < ACTION_SPACE_SIZE]
            return encode_move(random.choice(filtered_legal_moves)) if filtered_legal_moves else None
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_tensor)

            # Filter for legal moves within the action space limit
            legal_q_values = []
            legal_encoded_moves = []
            for move in legal_moves:
                encoded_move = encode_move(move)
                if encoded_move < ACTION_SPACE_SIZE:
                    legal_q_values.append(q_values[0, encoded_move])
                    legal_encoded_moves.append(encoded_move)

            # Select the move with the highest q-value
            if legal_q_values:
                best_move_q_value = max(legal_q_values)
                best_move_index = legal_q_values.index(best_move_q_value)
                return legal_encoded_moves[best_move_index]
            else:
                return None

    def train(self, state, action_idx, reward, next_state, done):
        legal_moves = env.legal_moves()
        action = decode_move(action_idx, legal_moves)
        if action is None:
            # Handle invalid action index if necessary
            return
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor(action, dtype=torch.int64).unsqueeze(0)
        reward_tensor = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
        done_tensor = torch.tensor(done, dtype=torch.float32).unsqueeze(0)

        # Get predicted Q-values for the current state
        current_q_values = self.model(state_tensor).gather(1, action_tensor)

        # Compute the target Q-values
        with torch.no_grad():
            max_next_q_values = self.model(next_state_tensor).max(1)[0].unsqueeze(1)
            target_q_values = reward_tensor + (self.gamma * max_next_q_values * (1 - done_tensor))

        # Calculate loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Main training loop
env = ChessEnv()
action_size = ACTION_SPACE_SIZE  # Use the defined ACTION_SPACE_SIZE
agent = DQNAgent(action_size)

for i in tqdm(range(100000)):  # Number of games
    state = env.reset()
    done = False

    while not done:
        legal_moves = env.legal_moves()
        action_idx = agent.select_action(state, legal_moves)
        action = decode_move(action_idx, legal_moves)

        if action not in legal_moves:
            continue

        next_state, reward, done = env.step(action)
        agent.train(state, action_idx, reward, next_state, done)
        state = next_state

# Save the model
torch.save(agent.model.state_dict(), 'chess_model.pth')