import chess
import random
import numpy as np
import keras
import os

current_directory = os.path.dirname(os.path.abspath(__file__))

# Function to convert the chess board to a numeric representation
def board_to_array(board):
    # Create a 8x8x6 array to represent the board
    board_array = np.zeros((8, 8, 6), dtype=np.uint8)

    # Define piece types as per their index
    piece_dict = {'r': 0, 'n': 1, 'b': 2, 'q': 3, 'k': 4, 'p': 5,
                  'R': 0, 'N': 1, 'B': 2, 'Q': 3, 'K': 4, 'P': 5}

    for i in range(64):
        piece = board.piece_at(i)
        if piece is not None:
            board_array[i // 8][i % 8][piece_dict[piece.symbol()]] = 1

    return board_array

# Function to create a simple neural network
def create_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(8, 8, 6)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')  # Output layer for move prediction
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to generate training data
def generate_training_data(num_samples):
    X_train = []
    y_train = []

    for _ in range(num_samples):
        board = chess.Board()
        moves = list(board.legal_moves)

        if len(moves) > 0:
            move = random.choice(moves)
            board.push(move)
            X_train.append(board_to_array(board))
            y_train.append(1)

    return np.array(X_train), np.array(y_train)

# Train the model
def train_model(model, X_train, y_train, epochs):
    model.fit(X_train, y_train, epochs=epochs, batch_size=64, validation_split=0.2)

# Generate training data
X_train, y_train = generate_training_data(10000)

if os.path.isdir('model_chess'):
    # Load model if it exists
    print('Model Loading...')
    model = keras.models.load_model(os.path.join(current_directory,'model_chess'))
    print('Model Loaded!')
else:
    # Create and train the model
    model = create_model()
    train_model(model, X_train, y_train, epochs=10)
    model.save(os.path.join(current_directory,'model_chess'))

# Function to make predictions using the trained model
def make_prediction(board, model):
    board_array = np.expand_dims(board_to_array(board), axis=0)
    prediction = model.predict(board_array)
    return prediction

# Use the prediction function to make moves
def ai_move(board, model):
    moves = list(board.legal_moves)
    if len(moves) > 0:
        move_predictions = {}
        for move in moves:
            board.push(move)
            move_predictions[move] = make_prediction(board, model)
            board.pop()
        
        best_move = max(move_predictions, key=move_predictions.get)
        return best_move
    return None

# Play against the AI
board = chess.Board()
print(board)
print("\nA","B","C","D","E","F","G","H")

while not board.is_game_over():
    if board.turn == chess.WHITE:
        moves = list(board.legal_moves)
        if len(moves) > 0:
            move = input("Enter your move in UCI format (e.g., 'e2e4'): ")
            try:
                if (move == "moves"):
                    print(list(board.legal_moves))
                elif chess.Move.from_uci(move) in moves:
                    board.push(chess.Move.from_uci(move))
                    print(board)
                    print("\nA","B","C","D","E","F","G","H")
                else:
                    print(move, "is an invalid move!")
            except:
                print("Put a valid move!")
        else:
            print("You have no legal moves. Game over. AI wins again.")
    else:
        best_move = ai_move(board, model)
        moves = list(board.legal_moves)
        if len(moves) > 0:
            board.push(best_move)
            print("AI's move:", best_move)
            print(board)
            print("\nA","B","C","D","E","F","G","H")
        else:
            print("AI has no legal moves. Game over. For the first time ever. You win.")
            break
