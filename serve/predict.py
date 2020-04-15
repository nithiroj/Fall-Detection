import json
import os
import logging

import torch
import torch.nn.functional as F

from model import FDNet2

from utils import bytearray_to_pil, preprocess

logger = logging.getLogger(__name__)

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    logger.info('Loading the model.')

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('Current device: {}'.format(device))
    model = FDNet2()

    # Load the store model parameters.
    model_path = os.path.join(model_dir, 'fdnet2-40.pt')
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device).eval()

    logger.info("Done loading model.")
    return model

def input_fn(serialized_input_data, content_type):
    logger.info('Deserializing the input data.')
    if content_type == 'image/png':
        data = serialized_input_data
        return data
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

def output_fn(prediction_output, accept):
    logger.info('Serializing the generated output.')
    labels = ['fall', 'not_fall']
    prob, label = prediction_output
    output = {'prob': prob.item(), 'label': labels[label.item()]}
    logger.info('Output Dict: {}'.format(output))
    result = json.dumps(output)
    logger.info('Result JSON: {}'.format(result))
    
    return result

def predict_fn(input_data, model):
    logger.info('Inferring input data.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('Current device: {}'.format(device))
    
    input_tensor = preprocess(bytearray_to_pil(input_data))
    logger.info('Input Tensor Size: {}'.format(input_tensor.size()))

    model.to(device)
    model.eval()

    output = model(input_tensor)
    logger.info('Output Tensor: {}'.format(output))
    result = torch.max(F.softmax(output, dim=1), dim=1)
    logger.info('Result Tensor: {}'.format(result))

    return result