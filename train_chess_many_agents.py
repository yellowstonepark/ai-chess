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


# Rest of the functions remain the same
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
        if done:
            if self.board.result() == "1-0":
                reward += 10  # Positive reward for White's victory
            elif self.board.result() == "0-1":
                reward += 10  # Positive reward for Black's victory
            else:
                reward -= 20   # Negative reward for a draw
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
        self.model = EnhancedChessCNN(action_size)
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
            # Apply additional reward or penalty at the end of the game
            if done:
                result = env.board.result()
                if result == "1-0":
                    reward += 50 - move_count  # Bonus for fast victory
                elif result == "0-1":
                    reward -= 10  # Penalty for losing
                agent_white.train(state, action_idx_white, reward, next_state, done)

        if done:
            break  # Game over

        # Black's turn
        agent_black = random.choice(agents_black)  # Randomly select an agent
        action_idx_black = agent_black.select_action(state, env.legal_moves())
        action_black = decode_move(action_idx_black, env.legal_moves())

        if action_black in env.legal_moves():
            next_state, reward, done = env.step(action_black)
            # Apply additional reward or penalty at the end of the game
            if done:
                result = env.board.result()
                if result == "0-1":
                    reward += 50 - move_count  # Bonus for fast victory
                elif result == "1-0":
                    reward -= 10  # Penalty for losing
                agent_black.train(state, action_idx_black, reward, next_state, done)

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
                current_best_model = new_best
                improvement_history.append(1)  # Mark improvement
            else:
                improvement_history.append(0)  # No improvement

# Print the improvement history at the end
print("Improvement history:", improvement_history)

# Save the model
torch.save(current_best_model.model.state_dict(), f'chess_model{num_games}_{num_agents}.pth')
