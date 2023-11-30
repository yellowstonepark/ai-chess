import chess
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import collections

# Define the piece values
PIECE_VALUES = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 30}

# Updated action space size
ACTION_SPACE_SIZE = 4672

# Number of agents and games
num_agents = int(input("Enter the number of agents (default is 5): "))
num_games = int(input("Enter the number of games (default is 10000): "))

# Enhanced Neural Network for DQN
class ChessCNN(nn.Module):
    def __init__(self, action_size=ACTION_SPACE_SIZE):
        super(ChessCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
    
class EnhancedChessCNN(nn.Module):
    def __init__(self, action_size):
        super(EnhancedChessCNN, self).__init__()
        
        # Input is now 12 channels, one for each piece type for both colors
        self.conv1 = nn.Conv2d(12, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # You might experiment with the number of fully connected layers and their sizes
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.dropout1 = nn.Dropout(p=0.3)  # Adjust dropout rate as needed
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, action_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# Function to encode the board state
def encode_board(board):
    # Define a mapping from pieces to channels
    piece_to_channel = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }

    # Create a 12x8x8 numpy array to represent the board
    board_array = np.zeros((12, 8, 8), dtype=np.float32)

    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            channel = piece_to_channel[piece.symbol()]
            board_array[channel, i // 8, i % 8] = 1  # Place the piece in the corresponding channel

    return board_array

# Chess environment
class ChessEnv:
    def __init__(self):
        self.board = chess.Board()
        self.last_capture_map = self._piece_capture_map()

    def reset(self):
        self.board.reset()
        self.last_capture_map = self._piece_capture_map()
        return encode_board(self.board)

    def _piece_capture_map(self):
        """Creates a map of pieces on the board with their values."""
        capture_map = {}
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece:
                capture_map[i] = PIECE_VALUES.get(piece.symbol().lower(), 0)
        return capture_map

    def step(self, move):
        # Determine if any piece is lost before making the move
        lost_piece_value = self._calculate_lost_piece_value(move)

        self.board.push(move)
        reward = -lost_piece_value  # Negative reward for losing a piece

        # Positional and mobility rewards
        mobility_reward = len(self.legal_moves()) / 100  # Scale down the reward
        center_control_reward = self._evaluate_center_control(move) / 100

        # Update reward with new factors
        reward += mobility_reward + center_control_reward

        # Check if any opponent's piece is captured
        taken_piece = self.board.piece_at(move.to_square)
        if taken_piece:
            reward += PIECE_VALUES.get(taken_piece.symbol().lower(), 0)

        self.last_capture_map = self._piece_capture_map()  # Update capture map after the move

        done = self.board.is_game_over()
        if done:
            if self.board.result() == "1-0":
                reward += 10  # Positive reward for White's victory
            elif self.board.result() == "0-1":
                reward += 10  # Positive reward for Black's victory
            else:
                reward -= 10   # Negative reward for a draw

        return encode_board(self.board), reward, done

    def _calculate_lost_piece_value(self, move):
        """Calculate the value of the piece lost in the current move, if any."""
        current_piece_value = self.last_capture_map.get(move.from_square, 0)
        new_piece_value = PIECE_VALUES.get(self.board.piece_at(move.to_square).symbol().lower(), 0) if self.board.piece_at(move.to_square) else 0
        return current_piece_value - new_piece_value if current_piece_value > new_piece_value else 0

    def legal_moves(self):
        return list(self.board.legal_moves)
    
    def _evaluate_center_control(self, move):
        center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
        control_score = 0
        for square in center_squares:
            if move.to_square == square:
                control_score += 1
        return control_score

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

class DQNAgent:
    def __init__(self, action_size, buffer_capacity=10000, batch_size=64):
        self.model = EnhancedChessCNN(action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.epsilon = 0.9
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99  # Discount factor
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size

    def select_action(self, state, legal_moves):
        if random.random() <= self.epsilon:
            # Randomly choose from the legal moves
            return encode_move(random.choice(legal_moves))
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_tensor)

            # Create a dictionary of Q-values for legal moves
            legal_q_values = {encode_move(move): q_values[0, encode_move(move)] for move in legal_moves}

            if legal_q_values:
                # Select the move with the highest Q-value among legal moves
                best_move = max(legal_q_values, key=legal_q_values.get)
                return best_move
            else:
                # In case there are no legal Q-values (highly unlikely), choose a random legal move
                return encode_move(random.choice(legal_moves)) if legal_moves else None

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, action_idxs, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Correcting the conversion to tensors
        states = torch.tensor(np.stack(states), dtype=torch.float32)
        next_states = torch.tensor(np.stack(next_states), dtype=torch.float32)
        actions = torch.tensor(action_idxs, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Get predicted Q-values for the current state
        current_q_values = self.model(states).gather(1, actions)

        # Compute the target Q-values
        with torch.no_grad():
            max_next_q_values = self.model(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        # Calculate loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Global variable to keep track of the best model
current_best_model = None
improvement_history = []

def conduct_tournament(agents, num_matches=3):
    scores = {agent: 0 for agent in agents}
    for _ in range(num_matches):
        for agent_a in agents:
            for agent_b in agents:
                if agent_a != agent_b:
                    result = compete(agent_a, agent_b, ChessEnv())
                    # print(result)
                    if result == "1-0":
                        scores[agent_a] += 1
                    elif result == "0-1":
                        scores[agent_b] += 1
                    # For a draw, no points are awarded

    # Return the agent with the highest score
    best_agent = max(scores, key=scores.get)
    return best_agent

def compete(agent_white, agent_black, env):
    state = env.reset()
    done = False
    total_moves = 0  # Counter for the total number of moves

    while not done and total_moves < 200:  # 100 moves for each player
        # White's turn
        action_idx_white = agent_white.select_action(state, env.legal_moves())
        action_white = decode_move(action_idx_white, env.legal_moves())
        if action_white in env.legal_moves():
            state, _, done = env.step(action_white)
        else:
            return "0-1"  # Illegal move by white, black wins

        total_moves += 1

        if done:
            break  # Game over

        # Black's turn
        action_idx_black = agent_black.select_action(state, env.legal_moves())
        action_black = decode_move(action_idx_black, env.legal_moves())
        if action_black in env.legal_moves():
            state, _, done = env.step(action_black)
        else:
            return "1-0"  # Illegal move by black, white wins

        total_moves += 1

    return env.board.result() if total_moves < 200 else "1/2-1/2"  # Draw if 200 moves are reached


def compare_models(model_a, model_b, num_games=7):
    wins_a = 0
    wins_b = 0

    for _ in range(num_games):
        result = compete(model_a, model_b, ChessEnv())
        if result == "1-0":
            wins_a += 1
        elif result == "0-1":
            wins_b += 1

    if wins_a > wins_b:
        return model_a
    elif wins_b > wins_a:
        return model_b
    else:
        return random.choice([model_a, model_b])  # Randomly choose in case of a tie


# Initialize two sets of agents
agents_white = [DQNAgent(ACTION_SPACE_SIZE) for _ in range(num_agents)]
agents_black = [DQNAgent(ACTION_SPACE_SIZE) for _ in range(num_agents)]

# Main training loop with multiple agents
for i in tqdm(range(num_games)):
    env = ChessEnv()
    state = env.reset()
    done = False
    move_count = 0  # Initialize move count

    while not done:
        move_count += 1

        # White's turn
        agent_white = random.choice(agents_white)  # Randomly select an agent
        action_idx_white = agent_white.select_action(state, env.legal_moves())
        action_white = decode_move(action_idx_white, env.legal_moves())

        if action_white in env.legal_moves():
            next_state, reward, done = env.step(action_white)
            # Push this experience to the replay buffer
            agent_white.replay_buffer.push(state, action_idx_white, reward, next_state, done)
            # Train the agent
            agent_white.train()

            if done:
                result = env.board.result()
                if result == "1-0":
                    reward += 50 - move_count  # Bonus for fast victory
                elif result == "0-1":
                    reward -= 10  # Penalty for losing
                break  # Game over

        # Black's turn
        agent_black = random.choice(agents_black)  # Randomly select an agent
        action_idx_black = agent_black.select_action(state, env.legal_moves())
        action_black = decode_move(action_idx_black, env.legal_moves())

        if action_black in env.legal_moves():
            next_state, reward, done = env.step(action_black)
            # Push this experience to the replay buffer
            agent_black.replay_buffer.push(state, action_idx_black, reward, next_state, done)
            # Train the agent
            agent_black.train()

            if done:
                result = env.board.result()
                if result == "0-1":
                    reward += 50 - move_count  # Bonus for fast victory
                elif result == "1-0":
                    reward -= 10  # Penalty for losing
                break  # Game over

        state = next_state

        # Write moves to moves.txt file
        with open("moves.txt", "a") as file:
            file.write(f"{num_games}_{num_agents}_{i};White: {action_white}, Black: {action_black}\n")

    # Write the game result to moves.txt file
    with open("moves.txt", "a") as file:
        file.write(f"{num_games}_{num_agents}_{i};Result: {env.board.result()}\n")

    # Every 50 rounds, conduct a tournament
    if i != 0 and i % 50 == 0:
        best_white = conduct_tournament(agents_white)
        best_black = conduct_tournament(agents_black)

        if current_best_model is None:
            current_best_model = best_white  # Initially set to the first best model found
            improvement_history.append(1)  # Mark improvement
        else:
            # Compare the new best models with the current global best
            new_best = compare_models(current_best_model, best_white)
            if new_best != current_best_model:
                # Doesnt REALLY make it better
                # current_best_model = new_best
                improvement_history.append(1)  # Mark improvement
            else:
                improvement_history.append(0)  # No improvement
        
        # Save the model
        torch.save(best_white.model.state_dict(), f'chess_model{num_games}_{num_agents}_white_{i}.pth')
        torch.save(best_black.model.state_dict(), f'chess_model{num_games}_{num_agents}_black_{i}.pth')

        current_best_model = best_white

# Print the improvement history at the end
print("Improvement history:", improvement_history)

# Save the model
torch.save(current_best_model.model.state_dict(), f'chess_model{num_games}_{num_agents}.pth')
