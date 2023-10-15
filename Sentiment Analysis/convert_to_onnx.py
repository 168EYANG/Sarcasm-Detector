import torch

from model import SarcasmModel

def main():
    pytorch_model = SarcasmModel(300)
    state_dict = torch.load("./trained_model.pt")
    pytorch_model.load_state_dict(state_dict)
    # pytorch_model.load_state_dict(torch.load("C:/Users/Ethan/Desktop/Help/trained_model.pt"))
    pytorch_model.eval()
    dummy_input = torch.zeros(1, 1, 28, 28)
    torch.onnx.export(pytorch_model, dummy_input, 'onnx_model.onnx', verbose=True)
    
if __name__ == '__main__':
    main()