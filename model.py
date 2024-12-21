import torch
torch.backends.quantized.engine = 'qnnpack'
def load_model(model_path: str):
    try:
        model = torch.jit.load(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
