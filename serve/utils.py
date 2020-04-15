import io
from PIL import Image
import logging


from torchvision import transforms

logger = logging.getLogger(__name__)

def bytearray_to_pil(img_bytearray):
    ''' Read iput bytearray to PIL image'''
    logger.info('Convert byte to PIL')

    pil_image = Image.open(io.BytesIO(img_bytearray))
    logger.info('PIL Image Size {}'.format(pil_image.size))
    
    return pil_image

def preprocess(pil_image):
    ''' Transform PIL image to required tensor'''
    logger.info('Convert PIL to tensor')

    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # repeat 1 to 3 channels
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(pil_image)
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor