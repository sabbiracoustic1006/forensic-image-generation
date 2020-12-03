#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 17:17:05 2020

@author: ratul
"""

import torch, os, argparse
import numpy as np
from models import GenerativeModel
from matplotlib import pyplot as plt

attribute_list = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',' Bags_Under_Eyes', 'Bald', 'Bangs', 
                  'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
                  'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                  'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
                  'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
                  'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 
                  'Wearing_Necktie', 'Young']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attributes', default='brown_hair heavy_makeup attractive no_beard young eyeglasses smiling', 
                        type=str, help='facial attributes given as input')
    parser.add_argument('--encoder', default='saved_models/vae.pth', type=str, help='weight path of encoder')
    parser.add_argument('--generator', default='saved_models/generator.pth', type=str, help='weight path of generator')
    parser.add_argument('--device', default='cpu', help='device to use for training')
    args = parser.parse_args()
    
    os.makedirs('generated-imgs',exist_ok=True)
    
    model = GenerativeModel()
    generator_weight = torch.load(args.generator)
    encoder_weight = torch.load(args.encoder)
    model.vae.load_state_dict(encoder_weight)
    model.generator.load_state_dict(generator_weight)
    
    input_attributes = [attr.lower() for attr in args.attributes.split()]
    
    attribute_vector = []
    for attr in attribute_list:
        if attr.lower() in input_attributes:
            attribute_vector.append(1)
        else:
            attribute_vector.append(0)
   
    attribute_vector = torch.tensor(attribute_vector).float()
    output = model(attribute_vector)
    image = np.uint8(255*output.data.permute(0,2,3,1).numpy())[0]
    
    plt.imshow(image)
    plt.axis('off')
    filename = "_".join(args.attributes.split())
    plt.savefig(f'generated-imgs/{filename}.jpg')
    
    
    