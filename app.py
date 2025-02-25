import anvil.server
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the trained Connect 4 model
cnn_model = tf.keras.models.load_model("connect4_cnn_model.h5")

#--------------Hybrid Model Configs--------------
class PatchIndex(tf.keras.layers.Layer):
    """Returns (batch, num_patches) with values [0..num_patches-1]."""
    def __init__(self, num_patches, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches

    def call(self, x):
        bs = tf.shape(x)[0]
        idx = tf.range(self.num_patches)
        idx = tf.expand_dims(idx, 0)  # shape (1, num_patches)
        idx = tf.tile(idx, [bs, 1])   # shape (batch, num_patches)
        return idx

class ClassTokenIndex(tf.keras.layers.Layer):
    """Returns shape (batch,1) all zeros for the class token embedding."""
    def call(self, x):
        bs = tf.shape(x)[0]
        idx = tf.range(1)        # [0]
        idx = tf.expand_dims(idx, 0)  # (1,1)
        idx = tf.tile(idx, [bs, 1])    # (batch,1)
        return idx

class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    1) Warmup from 0 to initial_lr over warmup_steps
    2) Cosine decay from initial_lr down to 0 over (total_steps - warmup_steps).
    """
    def __init__(self, initial_lr, warmup_steps, total_steps):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = float(warmup_steps)
        self.total_steps = float(total_steps)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        # Warmup phase
        def warmup_fn():
            return self.initial_lr * (step / self.warmup_steps)

        # Cosine decay phase
        def cosine_fn():
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return 0.5 * self.initial_lr * (1.0 + tf.cos(np.pi * progress))

        return tf.cond(step < self.warmup_steps, lambda: warmup_fn(), lambda: cosine_fn())

    def get_config(self):
        return {
            'initial_lr': self.initial_lr,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps
        }

# Now load the saved model. Replace the path below with the path to your saved model.
hybrid_model_path = 'Hybrid_best_67_val.keras'

# Use the custom_objects parameter to let Keras know about your custom classes.
hybrid_model = tf.keras.models.load_model(
    hybrid_model_path,
    custom_objects={
        'PatchIndex': PatchIndex,
        'ClassTokenIndex': ClassTokenIndex,
        'WarmupCosineSchedule': WarmupCosineSchedule
    }
)


#--------Pure Transformer Configs---------------------
# # Define the custom layers used in your model
# class PositionalIndex(tf.keras.layers.Layer):
#     def call(self, x):
#         bs = tf.shape(x)[0]
#         number_of_vectors = tf.shape(x)[1]
#         indices = tf.range(number_of_vectors)
#         indices = tf.expand_dims(indices, 0)
#         return tf.tile(indices, [bs, 1])

# class ClassTokenIndex(tf.keras.layers.Layer):
#     def call(self, x):
#         bs = tf.shape(x)[0]
#         indices = tf.range(1)
#         indices = tf.expand_dims(indices, 0)
#         return tf.tile(indices, [bs, 1])

# class GetItem(tf.keras.layers.Layer):
#     def __init__(self, index, **kwargs):
#         super().__init__(**kwargs)
#         self.index = index

#     def call(self, inputs):
#         return inputs[:, self.index]

#     def get_config(self):
#         config = super().get_config()
#         config.update({"index": self.index})
#         return config

# # Create a dictionary of custom objects
# custom_objects = {
#     'PositionalIndex': PositionalIndex,
#     'ClassTokenIndex': ClassTokenIndex,
#     'GetItem': GetItem
# }

# # Load the saved model with the custom objects
# transformer_model = keras.models.load_model('best_transformer_model2.keras',
#                                          custom_objects=custom_objects)

# Updated custom layers (as shown above)
class CLSTokenLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, **kwargs):
        super(CLSTokenLayer, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.cls_token = self.add_weight(
            shape=(1, 1, hidden_dim),
            initializer="random_normal",
            trainable=True,
            name="cls_token"
        )

    def call(self, x):
        batch_size = tf.shape(x)[0]
        cls_token = tf.tile(self.cls_token, [batch_size, 1, 1])
        return tf.concat([cls_token, x], axis=1)
    
    def get_config(self):
        config = super(CLSTokenLayer, self).get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, num_patches, hidden_dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.hidden_dim = hidden_dim
        self.pos_embed = self.add_weight(
            shape=(1, num_patches + 1, hidden_dim),
            initializer="random_normal",
            trainable=True,
            name="pos_embed"
        )

    def call(self, x):
        return x + self.pos_embed

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            "num_patches": self.num_patches,
            "hidden_dim": self.hidden_dim
        })
        return config

# Load the model
transformer_model = tf.keras.models.load_model(
    'final_best_vit_67_val.keras',
    custom_objects={
        'CLSTokenLayer': CLSTokenLayer,
        'PositionalEncoding': PositionalEncoding
    }
)

# Helper function for patch extraction (as used during training)
def extract_patches(inputs, patch_size=(3, 4), stride=(3, 3)):
    batch_size, height, width, channels = inputs.shape
    num_patches_h = (height - patch_size[0]) // stride[0] + 1
    num_patches_w = (width - patch_size[1]) // stride[1] + 1
    num_patches = num_patches_h * num_patches_w
    patch_dim = patch_size[0] * patch_size[1] * channels

    patches_tensor = tf.image.extract_patches(
        images=inputs,
        sizes=[1, patch_size[0], patch_size[1], 1],
        strides=[1, stride[0], stride[1], 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    patches = tf.reshape(patches_tensor, [batch_size, num_patches, patch_dim])
    return patches.numpy()



# Connect to Anvil using your uplink key (replace with your actual key)
anvil.server.connect("server_O67WWJTEX7KCRROTVANLQKNU-AIBDDZFW6WDSUHC5")


#server_GOXTCOWNLDPZ5QIAPXJSTEY3-LPWSO6HF6YFCSOAY
#server_QAAKVRJNC73VDWRBPFHOMYPP-YJIPXDMXAFKYQLUB
#server_EQDNEF4NYDK5V6KTONNRV5YK-3EM4PJCO3IYPQJ67"
#"server_TCI33OEOB6NVAP34AFU2NIMK-3EM4PJCO3IYPQJ67"



# def predict_best_move(input_array):
#     """
#     Predict the class (0 to 6) from a 6x7x2 input array using the loaded model.

#     Args:
#         input_array (np.array): A numpy array with shape (6, 7, 2).

#     Returns:
#         int: Predicted class label between 0 and 6.
#     """
#     # Ensure the input has the correct shape
#     #if input_array.shape != (6, 7, 2):
#     #    raise ValueError(f"Expected input shape (6, 7, 2) but got {input_array.shape}")

#     # Reshape to (1, 42, 2) as the model expects a batch dimension
#     #reshaped_input = np.array(input_array).reshape(1, 42, 2)
    
#     # Get the model predictions (logits or probabilities)
#     predictions = model.predict(np.array(input_array))
    
#     predicted_col = int(np.argmax(predictions, axis=-1)[0])
#     # Use argmax to get the predicted class index (0 to 6)
#     #predicted_class = np.argmax(predictions, axis=-1)[0]
#     #best_move =  int(predicted_class)
#     return {"best_move": predicted_col}



# def predict_best_move(board_state):
    # board_array = np.array(board_state).reshape(1, 6, 7, 2)
    # prediction = model.predict(board_array)
    # best_move = int(np.argmax(prediction))
    # return {"best_move": best_move}

def convert_connect4(board):
    # Convert the board to a NumPy array
    board_np = np.array(board)
    
    # Create a layer for human discs: 1 where board == 1, else 0
    human_layer = (board_np == 1).astype(int)
    
    # Create a layer for AI discs: 1 where board == 2, else 0
    ai_layer = (board_np == 2).astype(int)
    
    # Stack the layers along a new last axis to get shape (6,7,2)
    converted_board = np.stack([human_layer, ai_layer], axis=-1)
    return converted_board

@anvil.server.callable
def predict_best_move(board_state, model_type = 'CNN'):
    """
    Predicts the next move (column index 0-6) for a Connect 4 board state.
    
    Args:
        board_state (np.array): A NumPy array of shape (6, 7, 2) representing the board state.
        model (tf.keras.Model): The loaded Keras model.
    
    Returns:
        int: The predicted move (column index between 0 and 6).
    """
    # Verify input shape
    # if board_state.shape != (6, 7, 2):
    #     raise ValueError("board_state must be of shape (6, 7, 2)")

    if np.array(board_state).shape == (6,7):
        board_state = convert_connect4(board_state)
    else:
        print("yolo")
    

    
    

    if model_type == "Hybrid":

    
        # Add a batch dimension (model expects a 4D tensor: (batch, height, width, channels))
        board_state_batch = np.expand_dims(np.array(board_state), axis=0)
        
        # Get prediction probabilities from the model (expected shape: (1, 7))
        predictions = hybrid_model.predict(board_state_batch)
        
        # Choose the column with the highest probability
        predicted_move = int(np.argmax(predictions, axis=1)[0])
        print("using Hybrid")
        return {"best_move": predicted_move}

    elif model_type == 'CNN':
        board_array = np.array(board_state).reshape(1, 6, 7, 2)
        prediction = cnn_model.predict(board_array)
        best_move = int(np.argmax(prediction))
        print("using CNN")
        return {"best_move": best_move}

    else:
        # # Reshape board state from (6,7,2) to (42,2)
        # reshaped_state = np.array(board_state).reshape(42, 2)

        # # Add batch dimension: shape becomes (1,42,2)
        # input_state = np.expand_dims(reshaped_state, axis=0)

        # # Use the loaded model to predict probabilities for each move (0 to 6)
        # predictions = transformer_model.predict(input_state)
        # Add batch dimension: (1, 6, 7, 2)
        board_state_batch = np.expand_dims(board_state, axis=0)
    
        # Convert board state into patches (expected shape: (1,4,24))
        patches = extract_patches(board_state_batch)
    
        # Use the loaded model to get prediction probabilities over 7 moves (columns)
        predictions = transformer_model.predict(patches)

        # predictions has shape (1, 7); choose the column with the highest probability
        predicted_move = np.argmax(predictions, axis=1)[0]
        print("using Transformer")
        return {"best_move": predicted_move}



print("Backend connected to Anvil. Waiting for requests...")

# Keep the connection alive
anvil.server.wait_forever()





#model.summary()

# Example usage: Make predictions on dummy data.
#dummy_input = np.random.rand(1, 42, 2).astype(np.float32)
# predictions = model.predict(dummy_input)
# predicted_class = np.argmax(predictions, axis=-1)

# print("Predicted class:", predicted_class)  # For t
