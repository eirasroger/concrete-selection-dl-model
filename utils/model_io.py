import torch
import os

def save_model(model, filename="ranking_model.pt"):
    
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_file_dir, ".."))
    save_dir = os.path.join(project_root, "stored_models")
    
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), path)
    print(f"Model saved at: {path}")


def load_model(filename="ranking_model.pt"):
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_file_dir, ".."))
    save_dir = os.path.join(project_root, "stored_models")
    path = os.path.join(save_dir, filename)
    
    model = torch.jit.load(path)
    model.eval()
    return model

def test_loaded_model(path="ranking_model.pt", dummy_input=None, dummy_mask=None):
    model = load_model(path)
    model.eval()
    
    if dummy_input is None:
        raise ValueError("Must provide dummy_input because model requires multiple inputs")
    
    if dummy_mask is None:
        batch_size = dummy_input.size(0)
        seq_len = dummy_input.size(1)
        dummy_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    with torch.no_grad():
        output = model(dummy_input, dummy_mask)
    print("Output from loaded model:", output)
    return output