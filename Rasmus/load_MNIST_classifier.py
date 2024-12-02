# import tensorflow as tf
# import tf2onnx
# import tensorflow_hub as hub
# import torch
# import torch.nn as nn
# import os
# import onnxruntime as ort

# class ONNXWrapper(nn.Module):
#     def __init__(self, onnx_model_path):
#         super(ONNXWrapper, self).__init__()
#         self.session = ort.InferenceSession(onnx_model_path)
#         self.activations = {}

#     def forward(self, x):
#         if isinstance(x, torch.Tensor):
#             x = x.detach().cpu().numpy()
#         ort_inputs = {self.session.get_inputs()[0].name: x}
#         ort_outs = self.session.run(None, ort_inputs)
#         return torch.tensor(ort_outs[0])

#     def get_activations(self, x):
#         activations = []
#         def activation_hook(layer_name):
#             def hook(module, input, output):
#                 activations.append(output.clone().detach())
#                 self.activations[layer_name] = output.clone().detach()
#             return hook

#         dummy_hook = activation_hook("output")
#         dummy_hook(None, x, self.forward(x))
#         return activations

# def convert_and_save_tfhub_to_pytorch(tfhub_url, onnx_path, pytorch_save_path):
#     # Step 1: Download TensorFlow Hub model and save locally with signature
#     print("Downloading TensorFlow Hub model...")
#     model_dir = "./tfhub_model"
#     os.makedirs(model_dir, exist_ok=True)
#     hub_model = hub.load(tfhub_url)

#     # Define a wrapper module with a signature
#     class MyModel(tf.Module):
#         def __init__(self, model):
#             super(MyModel, self).__init__()
#             self.model = model

#         @tf.function(input_signature=[tf.TensorSpec(shape=[None, 28, 28, 1], dtype=tf.float32)])
#         def __call__(self, x):
#             # Ensure input is float32
#             x = tf.cast(x, tf.float32)
#             return self.model(x)

#     my_model = MyModel(hub_model)

#     # Save the model with the signature
#     tf.saved_model.save(my_model, model_dir, signatures=my_model.__call__)

#     print(f"TensorFlow Hub model saved to {model_dir}")

#     # Step 2: Load model locally
#     print("Loading TensorFlow Hub model...")
#     model = tf.saved_model.load(model_dir)
#     print("Available signatures:", list(model.signatures.keys()))

#     # Retrieve the concrete function from the loaded model
#     infer = model.signatures["serving_default"]

#     # Step 3: Export to ONNX using from_function
#     print(f"Exporting to ONNX: {onnx_path}")
#     # Convert the concrete function to ONNX
#     onnx_model_proto, _ = tf2onnx.convert.from_function(
#         infer,
#         input_signature=[tf.TensorSpec(shape=[None, 28, 28, 1], dtype=tf.float32)],
#         opset=13,
#         output_path=onnx_path
#     )
#     print("ONNX model exported successfully.")

#     # Step 4: Wrap ONNX model in PyTorch
#     print("Wrapping ONNX model in PyTorch...")
#     pytorch_model = ONNXWrapper(onnx_path)

#     # Step 5: Save PyTorch model
#     print(f"Saving PyTorch model to {pytorch_save_path}...")
#     torch.save(pytorch_model, pytorch_save_path)
#     print("PyTorch model saved successfully.")

# if __name__ == "__main__":
#     TFHUB_URL = "https://tfhub.dev/tensorflow/tfgan/eval/mnist/logits/1"
#     ONNX_PATH = "mnist_classifier.onnx"
#     PYTORCH_SAVE_PATH = "mnist_classifier.pt"

#     convert_and_save_tfhub_to_pytorch(TFHUB_URL, ONNX_PATH, PYTORCH_SAVE_PATH)


import tensorflow as tf
import tensorflow_hub as hub
import os

# Define the TensorFlow Hub URL
TFHUB_URL = "https://tfhub.dev/tensorflow/tfgan/eval/mnist/logits/1"

# Load the model from TensorFlow Hub
hub_model = hub.load(TFHUB_URL)

# Define a wrapper module with a signature
class MyModel(tf.Module):
    def __init__(self, model):
        super(MyModel, self).__init__()
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 28, 28, 1], dtype=tf.float32)])
    def __call__(self, x):
        x = tf.cast(x, tf.float32)
        return self.model(x)

my_model = MyModel(hub_model)

# Save the model with the signature
model_dir = "./tfhub_model"
os.makedirs(model_dir, exist_ok=True)
tf.saved_model.save(my_model, model_dir, signatures=my_model.__call__)