import torch
import onnx

from model import SarcasmModel

def main():
    pytorch_model = SarcasmModel(300)
    state_dict = torch.load("trained_model.pt")
    pytorch_model.load_state_dict(state_dict)
    # pytorch_model.load_state_dict(torch.load("C:/Users/Ethan/Desktop/Help/trained_model.pt"))
    pytorch_model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dummy_input = torch.zeros(12544).to(torch.int64).to(device)
    dummy_input.view(dummy_input.size(0), -1)
    torch.onnx.export(pytorch_model, dummy_input, 'onnx_model.onnx', verbose=True)
    
    onnx_model = onnx.load("onnx_model.onnx")
    onnx.checker.check_model(onnx_model)
if __name__ == '__main__':
    main()