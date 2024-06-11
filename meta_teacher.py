"""META-TEACHER"""

"""
Author: Leonardo Antiqui <leonardoantiqui@gmail.com>

"""


"""Libraries"""

from itertools import product
import skimage
import random
import torch
import torchvision

from generator import *


"""Global variables"""

#Weights for Gower's distance for attribute matching pretext task
distance_weights = [1.0,0.8,0.6,0.4,0.3,0.2]

#Difficulty threshold for triplet validation
difficulty = 0.1


def set_setting(variable, value):

    """
    Change weights or difficulty threshold
    
    Args:
        variable (str): variable to adjustment
        value (float): new value for the variable
    
    """

    if variable == 'distance':
        global distance_weights
        distance_weights = value
    elif variable == 'difficulty':
        global difficulty
        difficulty = value
        

"""Pattern recognition"""
  
def example_pattern_recognition(spec, image_width, image_height):

    """
    Generate a triplet of synthetic images of the pretext task pattern recognition
    
    Args:
        spec (tuple of int, int, int, int, int, int, int): image specification
        image_width (int): image width
        image_height (int): image height
        
    Returns:
        anchor (PIL.Image)
        positive (PIL.Image)
        negative (PIL.Image)
    
    """

    code = rand_code_pattern(spec)
    bg_color = code[0]

    positive = create_image_pattern(image_width, image_height, code)
    patch_size = random.randint(int(image_width*0.4), int(image_width*0.6))
    anchor, patch = cutout(positive, patch_size)

    code = rand_code_pattern(spec)
    code[0] = bg_color  #to ensure positive and negative samples share same background color

    negative = create_image_pattern(image_width, image_height, code)

    return anchor, positive, negative
    
    
"""Pattern completion"""

def example_pattern_completion(spec, image_width, image_height):

    """
    Generate a triplet of synthetic images of the pretext task pattern completion
    
    Args:
        spec (tuple of int, int, int, int, int, int, int): image specification
        image_width (int): image width
        image_height (int): image height
        
    Returns:
        anchor (PIL.Image)
        positive (PIL.Image)
        negative (PIL.Image)
    
    """
    
    code = rand_code_pattern(spec)
    bg_color = code[0]
    filter = code[-1]
    generated_image = create_image_pattern(image_width, image_height, code)
    patch_size = random.randint(int(image_width*0.2), int(image_width*0.4))
    anchor, positive = cutout(generated_image, patch_size)

    code = rand_code_pattern(spec)

    code[0] = bg_color #to ensure positive and negative samples share same background color
    code[-1] = filter #to ensure positive and negative samples share same filter

    generated_image = create_image_pattern(image_width, image_height, code)
    image, negative = cutout(generated_image, patch_size)

    return anchor, positive, negative
    

"""Jigsaw"""

def jigsaw_puzzle(image, num_pieces=4):

    """
    Split the image in panels that are randomly relocated 
    
    Args:
        image (PIL.image)
        num_pieces (int): number of panels. Default 4
        
    Returns:
        anchor (PIL.Image)
        positive (PIL.Image)
        negative (PIL.Image)
    
    """


    image_width = image.width
    image_height = image.height

    piece_w = image_width // int(num_pieces ** 0.5)
    piece_h = image_height // int(num_pieces ** 0.5)

    # Calculate the dimensions of the final puzzle
    final_height = piece_h * int(num_pieces ** 0.5)
    final_width = piece_w * int(num_pieces ** 0.5)

    img = np.array(image)

    pieces = [img[x:x + piece_h, y:y + piece_w] for x in range(0, final_height, piece_h)
                  for y in range(0, final_width, piece_w)]

    np.random.shuffle(pieces)

    # Create a blank canvas to reconstruct the puzzle
    puzzle = np.zeros((final_height, final_width, 3), dtype=np.uint8)

    # Reconstruct the puzzle
    idx = 0
    for i in range(0, final_height, piece_h):
        for j in range(0, final_width, piece_w):
            puzzle[i:i + piece_h, j:j + piece_w] = pieces[idx]
            idx += 1

    new_image = Image.fromarray(puzzle)

    return new_image.resize((image_width, image_height), Image.BILINEAR)

def example_pattern_jigsaw(spec, image_width, image_height):

    """
    Generate a triplet of synthetic images of the pretext task jigsaw puzzle
    
    Args:
        spec (tuple of int, int, int, int, int, int, int): image specification
        image_width (int): image width
        image_height (int): image height
        
    Returns:
        anchor (PIL.Image)
        positive (PIL.Image)
        negative (PIL.Image)
    
    """

    code = rand_code_pattern(spec)
    bg_color = code[0]
    filter = code[-1]

    anchor = create_image_pattern(image_width, image_height, code)
    num_pieces = random.choice([4,9])
    positive = jigsaw_puzzle(anchor, num_pieces)

    code = rand_code_pattern(spec)
    code[0] = bg_color #to ensure positive and negative samples share same background color
    code[-1] = filter #to ensure positive and negative samples share same filter

    generated_image = create_image_pattern(image_width, image_height, code)
    negative = jigsaw_puzzle(generated_image, num_pieces)

    return anchor, positive, negative
    

"""Distance metrics for attribute matching pretext task"""

def deltaECIE2000(color1_rgb, color2_rgb):

    """
    Compute delta E distance between two colors passed as rgb as given by CIEDE 2000 standard
    
    Args:
        color1_rgb (tuple of int, int, int): color 1
        color2_rgb (tuple of int, int, int): color 2
        
    Returns:
        deltaE (int): value representing the delta E distance
    
    """

    image1 = Image.new(mode="RGB", size=(1, 1), color=color1_rgb)
    image2 = Image.new(mode="RGB", size=(1, 1), color=color2_rgb)

    color1_lab = skimage.color.rgb2lab(image1)
    color2_lab = skimage.color.rgb2lab(image2)

    deltaE = skimage.color.deltaE_ciede2000(color1_lab, color2_lab)

    return deltaE.item()


def GD_color(color1_rgb, color2_rgb):
    
    """
    Compute the normalized similarity between two colors passed rgb
    
    Args:
        color1_rgb (tuple of int, int, int): color 1
        color2_rgb (tuple of int, int, int): color 2
        
    Returns:
        value (float): value representing the normilized similarity between the colors
    
    """

    diff = deltaECIE2000(color1_rgb, color2_rgb)
    return 1-abs(diff)/100

#Categorical variable
def GD_cat(var1, var2):

    """
    Compute the simple matching coefficient between the variables
    
    Args:
        var1 (any): variable 1
        var2 (any): variable 2
        
    Returns:
        value (int): binary value representing the matching or not of the variable
    
    """

    return int(var1 == var2)


def G_distances(code1, code2, code3):

    """
    Compute the Gower's distance between the codes
    
    Args:
        code1 (tuple): vector code of image 1
        code2 (tuple): vector code of image 2
        code3 (tuple): vector code of image 3
        
    Returns:
        value (float): Gower's distance between the codes
    
    """
    
    codes = [(code1, code2), (code1, code3), (code2, code3)]

    Gdistances = list()

    for codeA, codeB in codes:

        total = 0
        i = 0.001

        #Color background
        if not (code1[0] == code2[0] == code3[0]):
            if distance_weights[4] > 0:
                total += distance_weights[4]*GD_color(codeA[0], codeB[0])
                i += 1

        #Background pattern
        if not (code1[1] == code2[1] == code3[1]):
            if distance_weights[5] > 0:
                total += distance_weights[5]*GD_cat(codeA[1], codeB[1])
                i += 1

        #Shape type
        if not (code1[2] == code2[2] == code3[2]):
            if distance_weights[0] > 0:
                total += distance_weights[0]*GD_cat(codeA[2], codeB[2])
                i += 1

        #Color shape
        if not (code1[3] == code2[3] == code3[3]):
            if distance_weights[2] > 0:
                total += distance_weights[2]*GD_color(codeA[3], codeB[3])
                i += 1

        #Shape pattern
        if not (code1[4] == code2[4] == code3[4]):
            if distance_weights[3] > 0:
                total += distance_weights[3]*GD_cat(codeA[4], codeB[4])
                i += 1

        #Number of shapes
        if not (code1[8] == code2[8] == code3[8]):
            if distance_weights[1] > 0:
                total += distance_weights[1]*GD_cat(codeA[8], codeB[8])
                i += 1

        Gdistances.append(1-total/i) #the lower the more similar

    return Gdistances

"""2D shapes"""

def example_2Dshape(spec, image_width, image_height):

    """
    Generate a triplet of synthetic images of the pretext task attribute matching for general 2d shapes
    
    Args:
        spec (tuple of int, int, int, int, int, int, int, int, int): image specification
        image_width (int): image width
        image_height (int): image height
        
    Returns:
        anchor (PIL.Image)
        positive (PIL.Image)
        negative (PIL.Image)
    
    """

    code_anchor, code_positive, code_negative = triplet_codes(spec, rand_code_2Dshape)

    anchor = create_image_2Dshape(image_width, image_height, code_anchor)

    positive = create_image_2Dshape(image_width, image_height, code_positive)

    negative = create_image_2Dshape(image_width, image_height, code_negative)

    return anchor, positive, negative


"""Triplet code generation"""

def triplet_codes(spec, rand_code):
    
    """
    Generate a triplet of suitable codes for the pretext task attribute matching
    
    Args:
        spec (tuple of int, int, int, int, int, int, int, int, int): image specification
        rand_code (function): function to generate a random code based on the spec
        
    Returns:
        code_anchor (tuple): vector code of anchor image
        code_positive (tuple): vector code of positive image
        code_negative (tuple): vector code of negative image
    
    """
    
    #Hardness adjustment
    continuar = True
    while continuar:
        code1 = rand_code(spec)
        code2 = rand_code(spec)
        code3 = rand_code(spec)

        if random.uniform(0, 1) > 0.5:
            code2[2] = code1[2]
            if random.uniform(0, 1) > 0.6:
                code3[2] = code1[2]

        if random.uniform(0, 1) > 0.5:
            code2[0] = code1[0]
            if random.uniform(0, 1) > 0.6:
                code3[0] = code1[0]

        if random.uniform(0, 1) > 0.5:
            code2[3] = code1[3]
            if random.uniform(0, 1) > 0.6:
                code3[3] = code1[3]

        Gdistances = G_distances(code1, code2, code3)

        if max(Gdistances) - min(Gdistances) > difficulty:
            continuar = False

    idx_min = Gdistances.index(min(Gdistances))

    if idx_min == 0:
        code_anchor = code1
        code_positive = code2
        code_negative = code3
    elif idx_min == 1:
        code_anchor = code1
        code_positive = code3
        code_negative = code2
    else:
        code_anchor = code2
        code_positive = code3
        code_negative = code1

    return code_anchor, code_positive, code_negative
    
    
"""Digits"""

def duplet_2Ddigit(spec, image_width, image_height):

    """
    Generate a tuple of a synthetic image of a digit and its label
    
    Args:
        spec (tuple of int, int, int, int, int, int, int, int, int): image specification
        image_width (int): image width
        image_height (int): image height
        
    Returns:
        digit (PIL.Image)
        label (str): label
    
    """

    code = rand_code_2Ddigit(spec)

    digit = create_image_2Ddigit(image_width, image_height, code)

    label = code[2]

    return digit, label


def triplet_2Ddigit(spec, image_width, image_height):

    """
    Generate a triplet of synthetic images of the pretext task attribute matching for digits
    
    Args:
        spec (tuple of int, int, int, int, int, int, int, int, int): image specification
        image_width (int): image width
        image_height (int): image height
        
    Returns:
        anchor (PIL.Image)
        positive (PIL.Image)
        negative (PIL.Image)
    
    """

    code_anchor, code_positive, code_negative = triplet_codes(spec, rand_code_2Ddigit)

    anchor = create_image_2Ddigit(image_width, image_height, code_anchor)

    positive = create_image_2Ddigit(image_width, image_height, code_positive)

    negative = create_image_2Ddigit(image_width, image_height, code_negative)

    return anchor, positive, negative
    

def digit_dataset(num_examples, task, specs, image_size):

    """
    Generate a dataset with batches of tuples of digit images and their labels
    The tuples are loaded in the memory
    
    Args:
        num_examples (int): total number of examples in the dataset
        task (function): function to generate tuples of digits and their labels (duplet_2Ddigit)
        specs (list of tuples): list with vector specifications
        image_size (tuple of int, int): image size
        
    Returns:
        dataset (Dataset): batch of tuples loaded in the memory
    
    """

    image_width = image_size[0]
    image_height = image_size[1]

    anchors = list()
    labels = list()

    n_specs = len(specs)

    for _ in range(num_examples):

        if n_specs > 4:
            spec = random.choice(specs)
        else:
            weights = [8,4,2,1] #replay weight
            spec = random.choices(specs, weights=weights[:n_specs])[0]

        anchor, label = task(spec, image_width, image_height)

        anchor = torchvision.transforms.ToTensor()(anchor)

        anchors.append(anchor)
        labels.append(int(label))

    anchors = torch.stack(anchors, dim=0)
    labels = torch.tensor(labels)

    dataset = torch.utils.data.TensorDataset(anchors, labels)

    return dataset


"""Auxiliary functions"""

def custom_collate(batch, task, specs, image_size):

    """
    Generate a batch of triplets of images for any of the pretext tasks
    Collate function for data loader
    
    Args:
        batch (int): batch size
        task (function): function generator of triplet
        specs (list of tuples): list with vector specifications
        image_size (tuple of int, int): image size
        
    Returns:
        anchors (torch.Tensor): batch of anchors images
        positives (torch.Tensor): batch of positive images
        negatives (torch.Tensor): batch of negative images
        spec_list (torch.Tensor): batch of specifications (irrelevant)
    
    """

    image_width = image_size[0]
    image_height = image_size[1]

    anchors = list()
    positives = list()
    negatives = list()
    spec_list = list()
    
    n_specs = len(specs)

    for _ in batch:

        if n_specs > 4:
            spec = random.choice(specs)
        else:
            weights = [8,4,2,1] #replay weight
            spec = random.choices(specs, weights=weights[:n_specs])[0]

        anchor, positive, negative = task(spec, image_width, image_height)

        anchor = torchvision.transforms.ToTensor()(anchor)
        positive = torchvision.transforms.ToTensor()(positive)
        negative = torchvision.transforms.ToTensor()(negative)

        anchors.append(anchor)
        positives.append(positive)
        negatives.append(negative)
        spec_list.append(spec)

    anchors = torch.stack(anchors, dim=0)
    positives = torch.stack(positives, dim=0)
    negatives = torch.stack(negatives, dim=0)
    spec_list = torch.tensor(spec_list)

    return anchors, positives, negatives, spec_list
    
    
    
def triplet_dataset(num_examples, task, specs, image_size):

    """
    Generate a dataset with batches of triplets of images for any of the pretext tasks
    The triplets are loaded in the memory
    
    Args:
        num_examples (int): total number of examples in the dataset
        task (function): function generator of triplet
        specs (list of tuples): list with vector specifications
        image_size (tuple of int, int): image size
        
    Returns:
        dataset (Dataset): batch of triplets loaded in the memory
    
    """

    image_width = image_size[0]
    image_height = image_size[1]

    anchors = list()
    positives = list()
    negatives = list()
    spec_list = list()

    n_specs = len(specs)

    for _ in range(num_examples):

        if n_specs > 4:
            spec = random.choice(specs)
        else:
            weights = [8,4,2,1] #replay weight
            spec = random.choices(specs, weights=weights[:n_specs])[0]

        anchor, positive, negative = task(spec, image_width, image_height)

        anchor = torchvision.transforms.ToTensor()(anchor)
        positive = torchvision.transforms.ToTensor()(positive)
        negative = torchvision.transforms.ToTensor()(negative)

        anchors.append(anchor)
        positives.append(positive)
        negatives.append(negative)
        spec_list.append(spec)

    anchors = torch.stack(anchors, dim=0)
    positives = torch.stack(positives, dim=0)
    negatives = torch.stack(negatives, dim=0)
    spec_list = torch.tensor(spec_list)

    dataset = torch.utils.data.TensorDataset(anchors, positives, negatives, spec_list)

    return dataset
    
    
