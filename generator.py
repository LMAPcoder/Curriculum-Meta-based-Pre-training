"""GENERATOR"""

"""
Author: Leonardo Antiqui <leonardoantiqui@gmail.com>

"""

"""Libraries"""

from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageFont
import random
import numpy as np
import math
import subprocess
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from shapely.geometry import Point, Polygon



"""install noise library if it is not installed"""
try:
    import noise
except:
    subprocess.check_call(["pip", "install", "--quiet", "noise"])
    import noise


    
def fog_effect(image):

    """
    
    Add fog effect on an image
    
    Args:
        image (PIL.Image)
        
    Returns:
        image (PIL.Image)
    
    """

    # Create a white image with the same size as the input image
    fog_layer = Image.new('RGB', image.size, (255, 255, 255))

    # Create a drawing context
    draw = ImageDraw.Draw(fog_layer)
    
    # Calculate the maximum distance from the reference point (top-left corner)
    max_distance = (image.width ** 2 + image.height ** 2) ** 0.5

    center_x = random.randint(0, image.width)
    center_y = random.randint(0, image.height)

    # Iterate over each pixel in the image
    for y in range(image.height):
        for x in range(image.width):
            distance = 0.7 + (((x-center_x) ** 2 + (y-center_y) ** 2) ** 0.5) / max_distance
            opacity = 0.58 * distance
            draw.point((x, y), fill=(int(255 * opacity), int(255 * opacity), int(255 * opacity)))
    
    # Blend the fog layer with the input image
    return Image.blend(image, fog_layer, 0.2)
    

def distortion(image, filter):

    """
    
    Add a filtering effect on an image
    
    Args:
        image (PIL.Image)
        filter (str): the filtering effect
        
    Returns:
        image (PIL.Image)
    
    """

    if filter == 'blur':
        # image = image.filter(ImageFilter.BLUR)
        image = image.filter(ImageFilter.GaussianBlur(radius=1))
    elif filter == 'smooth':
        image = image.filter(ImageFilter.SMOOTH)
    elif filter == 'noise':
        image_array = np.array(image)
        noise_level = random.uniform(1, 10)
        noise = np.random.normal(loc=0, scale=noise_level, size=image_array.shape)
        noisy_image_array = np.clip(image_array + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(noisy_image_array)
    elif filter == 'fog':
        image = fog_effect(image)

    return image
    

def cutout(image, patch_size):

    """
    
    Add a gray patch located randomly on an image
    
    Args:
        image (PIL.Image)
        patch_size (tuple of int)
        
    Returns:
        image_patch (PIL.Image): patched image
        patch (PIL.Image): image crop at the patch location
    
    """

    image_width, image_height = image.size

    patch_position = (random.randint(0, image_width-patch_size), random.randint(0, image_height-patch_size))

    patch = image.crop((patch_position[0], patch_position[1], patch_position[0]+patch_size, patch_position[1]+patch_size))

    patch = patch.resize((image_width, image_height), Image.LANCZOS)

    color = random.randint(0, 255)
    patch_color = (color, color, color)

    image_patch = image.copy()

    image_patch.paste(patch_color, (patch_position[0], patch_position[1], patch_position[0]+patch_size, patch_position[1]+patch_size))

    return image_patch, patch

    
"""Noise pattern"""

def generate_noise_2d(shape, params):

    """
    
    Generate a 2d array with perlin or simplex noise
    
    Args:
        shape (tuple of int): size of the 2d array
        params (tuple of str, float, float, float, int): parameters for noise function
    
    Returns:
        world (np.ndarray): 2d array with perlin or simplex noise
    
    """

    noise_type, scale, octaves, persistence, lacunarity, seed = params

    if noise_type == 'perlin':
        noise_function = noise.pnoise2
    elif noise_type == 'simplex':
        noise_function = noise.snoise2

    world = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            world[i][j] = noise_function(i / scale,
                                         j / scale,
                                         octaves=octaves,
                                         persistence=persistence,
                                         lacunarity=lacunarity,
                                         repeatx=1024,
                                         repeaty=1024,
                                         base=seed)

    return world



"""Geometrical patterns"""


def get_parallel_lines(image, angles, color='black'):

    """
    
    Add parallel lines to an image
    
    Args:
        image (PIL.Image)
        angles (tuple of int): angles of the first and second lines
    
    Returns:
        image (PIL.Image) 
    
    """

    image_width = image.width
    image_height = image.height
    draw = ImageDraw.Draw(image)

    center_x, center_y = image_width // 2, image_height // 2

    radius = (image_width/2)*(2**(1/2))

    width =  random.randint(1, 2) #line width

    step_min = math.degrees(math.asin(7/radius)) #minimum distance between lines = 7

    for angle in angles:
        if angle == -1:
            continue

        step = random.uniform(step_min, 25)
        decay = random.uniform(0.999, 1)
        alpha = angle
        beta = angle

        while alpha > angle-180 and beta < angle+180:

            alpha = alpha - step
            beta = beta + step

            x1 = center_x + radius * math.cos(math.radians(alpha))
            y1 = center_y - radius * math.sin(math.radians(alpha))

            x2 = center_x + radius * math.cos(math.radians(beta))
            y2 = center_y - radius * math.sin(math.radians(beta))

            draw.line([(x1, y1), (x2, y2)], fill=color, width=width)

            if step * decay > 1:
                step *= decay

    return image



def get_convergent_lines(image, color='black'):

    """
    
    Add convergent lines to an image
    
    Args:
        image (PIL.Image)
        color (str): color of the line. Default 'black'
    
    Returns:
        image (PIL.Image) 
    
    """

    image_width = image.width
    image_height = image.height

    draw = ImageDraw.Draw(image)

    # width =  random.randint(1, 2)
    width = 1

    center_x = random.randint(image_width//3, 2*image_width//3)
    center_y = random.randint(image_height//3, 2*image_height//3)

    radius = 1.5 * (image_width/2)*(2**(1/2))

    num_lines = random.randint(8, 16)

    step = (2 * math.pi) / num_lines

    for i in range(num_lines):
        angle = i * step
        x = center_x + radius * math.cos(angle)
        y = center_y - radius * math.sin(angle)

        draw.line([(center_x, center_y), (x, y)], fill=color, width=width)

    image

    return image



def get_parallel_curves(image, angles, color='black'):

    """
    
    Add parallel curves to an image
    
    Args:
        image (PIL.Image)
        angles (tuple of int, int)
        color (str): color of the line. Default 'black'
    
    Returns:
        image (PIL.Image) 
    
    """

    image_width = image.width
    image_height = image.height
    draw = ImageDraw.Draw(image)
    width =  random.randint(1, 2)

    step = random.randint(4, image_height//4)
    mrg =  random.randint(0, image_width)

    for i in range(-image_height, image_height, step):
        draw.arc([(-image_width+mrg, i), (2*image_width-mrg, image_height+i)], start=angles[0], end=angles[1], fill=color, width=width)

    return image



def get_parallel_cosines(image, color='black'):

    """
    
    Add parallel cosine curves to an image
    
    Args:
        image (PIL.Image)
        color (str): color of the line. Default 'black'
    
    Returns:
        image (PIL.Image) 
    
    """

    image_width = image.width
    image_height = image.height
    draw = ImageDraw.Draw(image)
    width =  random.randint(1, 2)

    amplitude = random.uniform(5, image_width/3)  # Amplitude of the cosine curve

    frequency = random.uniform(0, amplitude/1000)  # Frequency of the cosine curve (controls the number of oscillations)
    phase_shift = random.uniform(0, 5)  # Phase shift of the curve

    step = random.randint(5, image_height//4)

    # Calculate points along the cosine curve
    for j in range(0, image_height, step):
        points = []
        for x in range(image_width):
            y = amplitude * math.cos(2 * math.pi * frequency * x + phase_shift) + j
            points.append((x, y))

        # Draw lines connecting the points to form the curve
        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill=color, width=width)

    return image



def get_concentric_polygons(image, N_edges, color='black'):

    """
    
    Add concentric regular polygons to an image
    
    Args:
        image (PIL.Image)
        N_edges (int): number of edges in the polygon
        color (str): color of the line. Default 'black'
    
    Returns:
        image (PIL.Image) 
    
    """

    image_width = image.width
    image_height = image.height

    draw = ImageDraw.Draw(image)

    center_x = random.randint(image_width//3, 2*image_width//3)
    center_y = random.randint(image_height//3, 2*image_height//3)

    center = (center_x, center_y)

    R = image_width*(2**(1/2))
    step = random.randint(4, image_width//4)
    N = int(R/step)

    for i in range(1,N+1):
        draw.regular_polygon(bounding_circle=(center, step*i), n_sides=N_edges, rotation=0, fill=None, outline='black')

    return image



def get_concentric_ellipses(image, color='black'):

    """
    
    Add concentric ellipses to an image
    
    Args:
        image (PIL.Image)
        color (str): color of the line. Default 'black'
    
    Returns:
        image (PIL.Image) 
    
    """

    image_width = image.width
    image_height = image.height

    draw = ImageDraw.Draw(image)

    center_x = random.randint(image_width//3, 2*image_width//3)
    center_y = random.randint(image_height//3, 2*image_height//3)

    radius = (image_width/2)*(2**(1/2))

    step = random.randint(4, image_width//4)

    x = random.uniform(1, 2)
    y = random.uniform(1, 2)

    for r in range(int(radius), 0, -step):
        draw.ellipse([(center_x - r*x, center_y - r*y),
                      (center_x + r*x, center_y + r*y)],
                      outline=color)

    return image



def get_dots(image, r, color='black'):

    """
    
    Add dots to an image
    
    Args:
        image (PIL.Image)
        r (float): vertical alignment between consecutive dots
        color (str): color of the line. Default 'black'
    
    Returns:
        image (PIL.Image) 
    
    """

    image_width = image.width
    image_height = image.height

    draw = ImageDraw.Draw(image)

    x_spacing = random.randint(7, image_width//4)
    # y_spacing = random.randint(5, image_height//4)
    y_spacing = x_spacing

    num_dots = image_width//x_spacing

    # dot_radius = random.randint(1, 2)
    dot_radius = random.uniform(0.8,1.2)

    lag = x_spacing * r

    for i in range(image_width//x_spacing + 1):
        x = i * x_spacing
        for j in range(image_height//y_spacing + 1):
            y = j * y_spacing + lag*(i%2)
            draw.ellipse((x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius), fill=color)

    return image



def get_stripes(image, tipo, color='black'):

    """
    
    Add stripes to an image
    
    Args:
        image (PIL.Image)
        tipo (str): type of stripe (vertical, horizontal or both)
        color (str): color of the line. Default 'black'
    
    Returns:
        image (PIL.Image) 
    
    """


    image_width = image.width
    image_height = image.height
    draw = ImageDraw.Draw(image)

    step = random.randint(5, image_width//4)

    if tipo == 'h' or tipo == 'b':
        x = random.randint(0, step)
        for i in range(0, image_height//step+1):
            if i%2 == 0:
                draw.rectangle([(0, i*step+x), (image_width, (i+1)*step+x)], fill =color)

    if tipo == 'v' or tipo == 'b':
        x = random.randint(0, step)
        for i in range(0, image_width//step+1):
            if i%2 == 0:
                draw.rectangle([(i*step+x, 0), ((i+1)*step+x, image_height)], fill =color)

    return image
    


def create_image_pattern(image_width, image_height, code):

    """
    
    Create an image with regular pattern according to the vector code
    
    Args:
        image_width (int): image width
        image_height (int): image height
        code (tuple): image vector code
    
    Returns:
        image (PIL.Image) 
    
    """

    background_color = code[0]
    filter = code[-1]

    image = Image.new(mode="RGB", size=(image_width, image_height), color=background_color)

    for c in code[1:-1]:

        if c is None:
            continue

        pattern = c[0]
        confg = c[1]

        if pattern == 'linear':
            angles = confg
            image = get_parallel_lines(image, angles)

        elif pattern == 'convergent_lines':
            image = get_convergent_lines(image)

        elif pattern == 'stripes':
            tipo = confg
            image = get_stripes(image, tipo)

        elif pattern == 'concentric_polygons':
            N_edges = confg
            image = get_concentric_polygons(image, N_edges)

        elif pattern == 'curved':
            angles = confg
            image = get_parallel_curves(image, angles)

        elif pattern == 'cosine':
            image = get_parallel_cosines(image)

        elif pattern == 'concentric_ellipses':
            image = get_concentric_ellipses(image)

        elif pattern == 'dots':
            lag = confg
            image = get_dots(image, lag)

    if filter:
        image = distortion(image, filter)

    return image
    


def rand_code_pattern(spec):

    """
    
    Create a vector code from vector specification
    
    Args:
        spec (tuple of int, int, int, int, int, int, int): image specification
    
    Returns:
        code (tuple): vector code
    
    """

    attr_N = len(spec) #Number of attributes

    code = list() #code

    for attr_idx in range(attr_N):

        attr_D = spec[attr_idx] # Attribute difficulty

        #Background full color
        if attr_idx == 0:

            #Grey background
            if attr_D == 1:
                color = random.randint(50, 255)
                color_tuple = (color, color, color)

            # #Primary colors
            # elif attr_D == 2:
            #     color_tuple = random.choice([(255, 0, 0), (0, 255, 0),(0, 0, 255)])

            #Multicolor background
            elif attr_D == 2:
                color_tuple = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            #White background
            else:
                color_tuple = (255, 255, 255)

            code.append(color_tuple)

        if attr_idx == 1:
            pattern = 'linear'

            if attr_D == 1:
                angles = (random.choice([0, 90]), -1)
                pattern_tuple = (pattern, angles)
            elif attr_D == 2:
                angles = (random.randint(0, 360), -1)
                pattern_tuple = (pattern, angles)
            elif attr_D == 3:
                angles = (random.randint(0, 90), random.randint(91, 180))
                pattern_tuple = (pattern, angles)
            elif attr_D == 4:
                pattern = 'convergent_lines'
                pattern_tuple = (pattern, None)
            else:
                pattern_tuple = None

            code.append(pattern_tuple)

        if attr_idx == 2:
            pattern = 'stripes'
            if attr_D == 1:
                tipo = random.choice(['h','v'])
                pattern_tuple = (pattern, tipo)

            elif attr_D == 2:
                tipo = random.choice(['h','v','b'])
                pattern_tuple = (pattern, tipo)

            else:
                pattern_tuple = None

            code.append(pattern_tuple)

        if attr_idx == 3:

            if attr_D == 1:
                pattern = 'curved'
                angles = random.choice([(0, 180), (180, 360)])
                pattern_tuple = (pattern, angles)

            elif attr_D == 2:
                pattern = 'cosine'
                pattern_tuple = (pattern, None)

            else:
                pattern_tuple = None

            code.append(pattern_tuple)

        if attr_idx == 4:

            if attr_D == 1:
                pattern = 'concentric_polygons'
                N_edges = random.randint(3, 6)
                pattern_tuple = (pattern, N_edges)

            elif attr_D == 2:
                pattern = 'concentric_ellipses'
                pattern_tuple = (pattern, None)

            else:
                pattern_tuple = None

            code.append(pattern_tuple)

        if attr_idx == 5:
            pattern = 'dots'
            if attr_D == 1:
                r = random.choice((0, 0.5))
                pattern_tuple = (pattern, r)
            elif attr_D == 2:
                r = random.uniform(0, 1)
                pattern_tuple = (pattern, r)
            else:
                pattern_tuple = None

            code.append(pattern_tuple)

        #Image filtering
        if attr_idx == 6:
            filter = None
            if attr_D == 1:
                filter = random.choices(['blur', 'smooth', 'noise', 'fog', None], weights=[2, 2, 2, 2, 1])[0]
            code.append(filter)

    return code
    

"""2D shapes"""

#Function to draw regular polygons
def draw_reg_polygon(image, N_edges):

    """
    
    Create a regular polygon on an image
    
    Args:
        image (PIL.Image)
        N_edges (int): number of edges in the polygon
    
    Returns:
        image (PIL.Image)
    
    """

    draw = ImageDraw.Draw(image)

    radius = image.width//2

    center_x = image.width//2
    center_y = image.height//2

    draw.regular_polygon(
                bounding_circle=(center_x, center_y, radius), #a tuple defined by a point and radius
                n_sides=N_edges,
                rotation=0,
                fill=255,
                outline=None
                )

    return image



def superformula_polar(a, b, y, z, n1, n2, n3, phi):

    """
    
    Compute the radius of a point using a generalization of Gielis' superformula
    
    Args:
        a (float): coefficient
        b (float): coefficient
        y (float): coefficient
        z (float): coefficient
        n1 (float): coefficient
        n2 (float): coefficient
        n3 (float): coefficient
        phi (float): angle in radians
    
    Returns:
        (r, phi) (tuple of float, float)
    
    """

    t1 = math.cos(y * phi / 4.0) / a
    t1 = abs(t1)
    t1 = pow(t1, n2)

    t2 = math.sin(z * phi / 4.0) / b
    t2 = abs(t2)
    t2 = pow(t2, n3)

    t3 = -1 / float(n1)
    r = pow(t1 + t2, t3)
    if abs(r) == 0:
        return (0, 0)
    else:
      #     return (r * cos(phi), r * sin(phi))
        return (r, phi)
        


def supershape_points(params, image_size=(100,100), point_count=1000):

    """
    
    Generate a list of points for the contour of a super shape
    
    Args:
        params (tuple): Gielis' coefficients
        image_size (tuple of int, int): image size. Default (100, 100)
        point_count (int): number of points in the contour. Default 1000
    
    Returns:
        points (list of tuples)
    
    """

    a, b, y, z, n1, n2, n3 = params

    phis = np.linspace(0, 2 * np.pi, point_count)

    points = [superformula_polar(a, b, y, z, n1, n2, n3, phi) for phi in phis]

    rmax = max(points, key=lambda x: x[0])[0]

    width = image_size[0]
    height = image_size[1]

    # scale and transpose...
    path = []
    for r, a in points:
        x = (width/2) * (r/rmax) * math.cos(a)
        y = (height/2) * (r/rmax) * math.sin(a)
        path.append((x, y))

    points = [(int(width/2+px), int(height/2+py)) for px, py in path]

    return points
    


def supershape_image(image, params):
    
    """
    
    Add a 2d shape on an image
    
    Args:
        image (PIL.Image)
        params (tuple): Gielis' coefficients
    
    Returns:
        image (PIL.Image)
    
    """

    draw = ImageDraw.Draw(image)
    image_size = (image.width, image.height)
    points = supershape_points(params, image_size)
    draw.polygon(points, fill=255)
    return image
    
    
"""Panel positions"""


def panels_positions(image_width, image_height, shapes_count, shape_size=None, occlusion=None):

    """
    
    Provide random locations for the panels
    
    Args:
        image_width (int): image width
        image_height (int): image height
        shapes_count (int): number of panels to generate
        shape_size (float): proportion of the maximum panel size possible
        occlusion (bool): if partial occlusion between shapes is allowed
    
    Returns:
        panels (list of tuples): panel locations
    
    """

    panel_size_initial = image_width
    if shapes_count > 1:
        panel_size_initial = image_width//2

    delta = 0 #No occlusion
    panel_size = panel_size_initial
    panels = list()

    i = 0

    while i < shapes_count:

        forbidden = True
        tries = 1
        while forbidden:

            if shape_size:
                panel_size = int(panel_size_initial*shape_size)
                x1 = random.randint(0, image_width-panel_size)
                y1 = random.randint(0, image_height-panel_size)
                shape_size = random.uniform(0.7, 1)
            else:
                if shapes_count == 1:
                    x1 = 0
                    y1 = 0
                else:
                    if random.uniform(0, 1) > 0.5:
                        x1 = random.choice((0, panel_size))
                        if random.uniform(0, 1) > 0.6:
                            y1 = random.choice((0, panel_size))
                        else:
                            y1 = random.randint(0, image_height-panel_size)
                    else:
                        y1 = random.choice((0, panel_size))
                        if random.uniform(0, 1) > 0.6:
                            x1 = random.choice((0, panel_size))
                        else:
                            x1 = random.randint(0, image_width-panel_size)


            if occlusion:
                delta = panel_size*random.uniform(0, 0.2)

            k = 0
            for j in range(i):

                panel = panels[j]

                p1 = (panel[0]-panel_size+delta, panel[1]-panel_size+delta)
                p2 = (panel[2], panel[1]-panel_size+delta)
                p3 = (panel[2], panel[3])
                p4 = (panel[0]-panel_size+delta, panel[3])

                panel = Polygon((p1, p2, p3, p4))

                if Point(x1, y1).within(panel):
                    break
                k += 1

            if k == i:
                forbidden = False
            else:
                tries += 1

            if tries > 10:
                i = 0
                panels = list()
                break

        x2 = x1 + panel_size
        y2 = y1 + panel_size
        panel = (x1, y1, x2, y2)

        panels.append(panel)
        i+=1

    return panels



def create_image_2Dshape(image_width, image_height, code):

    """
    
    Create an image with a 2d Gielis' shape according the vector code
    
    Args:
        image_width (int): image width
        image_height (int): image height
        code (tuple): image vector code
    
    Returns:
        image (PIL.Image) 
    
    """

    background_color = code[0]

    image = Image.new(mode="RGB", size=(image_width, image_height), color=background_color)
    # draw = ImageDraw.Draw(image)

    background_pattern  = code[1]
    shape_type = code[2]
    shape_color = code[3]
    shape_pattern = code[4]
    shape_size = code[5]
    shape_rotation = code[6]
    filter = code[7]
    shapes_count = code[8]


    color = random.randint(0, 200)
    color_inner = (color, color, color)

    #Background pattern
    if background_pattern:
        world = generate_noise_2d((image_width, image_height), eval(background_pattern))
        world = ((world - world.min()) / (world.max() - world.min()) * 255).astype(np.uint8)
        world = Image.fromarray(world, mode='L')
        image.paste(world, (0, 0), world)

    if shape_type:

        #Panels
        panels = panels_positions(image_width, image_height, shapes_count, shape_size)

        #Drawing
        for panel in panels:

            panel_size = (panel[2]-panel[0], panel[3]-panel[1])

            # Shape pattern
            shape_background = Image.new(mode="RGB", size=panel_size, color=shape_color)
            color = random.randint(0, 200)
            color_inner = (color, color, color)

            if shape_pattern:
                world = generate_noise_2d(panel_size, eval(shape_pattern))
                world = ((world - world.min()) / (world.max() - world.min()) * 255).astype(np.uint8)
                world = Image.fromarray(world, mode='L')
                shape_background.paste(world, (0, 0), world)


            shape_panel = Image.new(mode="L", size=panel_size, color=0)

            if shape_type:                
                shape = supershape_image(shape_panel, eval(shape_type))

            #Rotation
            if shape_rotation:
                shape = shape.rotate(angle=shape_rotation, resample=Image.Resampling.BILINEAR, expand=0)
                shape_rotation = random.randint(0, 360)

            image.paste(shape_background, #Source image or pixel value
                        box=panel, #4-tuple giving the region to paste into
                        mask=shape #mask image (updates only the regions indicated by the mask)
                        )

    if filter:
        image = distortion(image, filter)
        
    if random.uniform(0, 1) > 0.9:
        patch_size = random.randint(int(image.width*0.2), int(image.height*0.4))
        image, patch = cutout(image, patch_size)

    return image

    
def rand_code_2Dshape(spec):

    """
    
    Create a vector code from vector specification
    
    Args:
        spec (tuple of int, int, int, int, int, int, int, int, int): image specification
    
    Returns:
        code (tuple): vector code
    
    """

    attr_N = len(spec) #Number of attributes
    code = list() #code

    for attr_idx in range(attr_N):

        attr_D = spec[attr_idx] # Attribute difficulty

        #Background full color
        if attr_idx == 0:

            #Grey background
            if attr_D == 1:
                color = random.randint(50, 255)
                color_tuple = (color, color, color)
            # #Primary colors
            # elif attr_D == 2:
            #     color_tuple = random.choice([(255, 0, 0), (0, 255, 0),(0, 0, 255)])
            #Multicolor background
            elif attr_D == 2:
                color_tuple = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            #White background
            else:
                color_tuple = (255, 255, 255)

            code.append(color_tuple)

        #Background pattern
        if attr_idx == 1:
            pattern = None
            if attr_D == 1:
                # pattern = random.choice([None, 'linear', 'dots'])
                pattern = str(random.choice(texture_noise))

            code.append(pattern)

        #Shape type
        if attr_idx == 2:
            shape_type = None
            if attr_D == 1:
                shape_type = str(random.choice(super_shapes[:10]))
            elif attr_D == 2:
                shape_type = str(random.choice(super_shapes))

            code.append(shape_type)

        #Shape color
        if attr_idx == 3:
            color_tuple = None
            if shape_type:
                #Grey shape
                if attr_D == 1:
                    color = random.randint(0, 200)
                    color_tuple = (color, color, color)
                # #Primary colors
                # elif attr_D == 2:
                #     color_tuple = random.choice([(255, 0, 0), (0, 255, 0),(0, 0, 255)])
                #Multicolor shape
                elif attr_D == 2:
                    color_tuple = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                #Black shape
                else:
                    color_tuple = (0, 0, 0)

                if color_tuple == code[0]:
                    color_tuple = (255, 255, 255) #white shape

            code.append(color_tuple)

        #Shape pattern
        if attr_idx == 4:
            pattern = None
            if shape_type:
                if attr_D == 1:
                    pattern = str(random.choice(texture_noise))
            code.append(pattern)

        #Shape size
        if attr_idx == 5:
            shape_size = None
            if shape_type:
                if attr_D == 1:
                    shape_size = round(random.uniform(0.7, 1),2)
            code.append(shape_size)

        #Rotation
        if attr_idx == 6:
            rotation = None
            if shape_type:
                if attr_D == 1:
                    rotation = random.randint(0, 360)
            code.append(rotation)

        #Image filtering
        if attr_idx == 7:
            filter = None
            if attr_D == 1:
                # filter = random.choices(['blur', 'smooth', 'noise', 'fog', None], weights=[2, 2, 2, 2, 1])[0]
                filter = random.choices(['blur', 'smooth', None], weights=[2, 2, 1])[0]
            code.append(filter)

        #Shape count
        if attr_idx == 8:
            shapes_count = None
            if shape_type:
                shapes_count = 1
                if attr_D == 1:
                    shapes_count = random.randint(2, 4)
            code.append(shapes_count)

    return code
    

"""2d digits"""


def digit(image, digit):

    """
    
    Add a 2d digit on an image
    
    Args:
        image (PIL.Image)
        digit (str): digit to be drawn
    
    Returns:
        image (PIL.Image)
    
    """
    
    draw = ImageDraw.Draw(image)

    font_path = random.choices(fm.findSystemFonts(fontpaths=None, fontext='ttf'))[0]

    init_size = 250
    wrongsize = True

    while wrongsize:

        font = ImageFont.truetype(font_path, init_size)

        length = font.getlength(digit, direction='ltr')
        height = font.getlength(digit, direction='ttb')

        if length > image.width or height > image.height:
            init_size -= 10
        else:
            wrongsize = False

    draw.text(
        xy=(image.width//2, image.height//2), #The anchor coordinates of the text.
        text=digit,
        fill='white', 
        font=font,
        anchor='mm'
        )
    
    return image


def create_image_2Ddigit(image_width, image_height, code):

    """
    
    Create an image with a 2d digit according vector code
    
    Args:
        image_width (int): image width
        image_height (int): image height
        code (tuple): image vector code
    
    Returns:
        image (PIL.Image) 
    
    """

    background_color = code[0]

    image = Image.new(mode="RGB", size=(image_width, image_height), color=background_color)
    # draw = ImageDraw.Draw(image)

    background_pattern  = code[1]
    shape_type = code[2]
    shape_color = code[3]
    shape_pattern = code[4]
    shape_size = code[5]
    shape_rotation = code[6]
    filter = code[7]
    shapes_count = code[8]


    color = random.randint(0, 200)
    color_inner = (color, color, color)

    #Background pattern
    if background_pattern:
        world = generate_noise_2d((image_width, image_height), eval(background_pattern))
        world = ((world - world.min()) / (world.max() - world.min()) * 255).astype(np.uint8)
        world = Image.fromarray(world, mode='L')
        image.paste(world, (0, 0), world)

    if shape_type:

        #Panels
        panels = panels_positions(image_width, image_height, shapes_count, shape_size)

        #Drawing
        for panel in panels:

            panel_size = (panel[2]-panel[0], panel[3]-panel[1])

            # Shape pattern
            shape_background = Image.new(mode="RGB", size=panel_size, color=shape_color)
            color = random.randint(0, 200)
            color_inner = (color, color, color)

            if shape_pattern:
                world = generate_noise_2d(panel_size, eval(shape_pattern))
                world = ((world - world.min()) / (world.max() - world.min()) * 255).astype(np.uint8)
                world = Image.fromarray(world, mode='L')
                shape_background.paste(world, (0, 0), world)


            shape_panel = Image.new(mode="L", size=panel_size, color=0)

            if shape_type:
                shape = digit(shape_panel, shape_type)

            #Rotation
            if shape_rotation:
                shape = shape.rotate(angle=shape_rotation, resample=Image.Resampling.BILINEAR, expand=0)
                shape_rotation = random.randint(0, 360)

            image.paste(shape_background, #Source image or pixel value
                        box=panel, #4-tuple giving the region to paste into
                        mask=shape #mask image (updates only the regions indicated by the mask)
                        )

    if filter:
        image = distortion(image, filter)

    if random.uniform(0, 1) > 0.9:
        patch_size = random.randint(int(image.width*0.2), int(image.height*0.4))
        image, patch = cutout(image, patch_size)

    return image



def rand_code_2Ddigit(spec):

    """
    
    Create a vector code from vector specification
    
    Args:
        spec (tuple of int, int, int, int, int, int, int, int, int): image specification
    
    Returns:
        code (tuple): vector code
    
    """

    attr_N = len(spec) #Number of attributes
    code = list() #code

    for attr_idx in range(attr_N):

        attr_D = spec[attr_idx] # Attribute difficulty

        #Background full color
        if attr_idx == 0:

            #Grey background
            if attr_D == 1:
                color = random.randint(50, 255)
                color_tuple = (color, color, color)
            # #Primary colors
            # elif attr_D == 2:
            #     color_tuple = random.choice([(255, 0, 0), (0, 255, 0),(0, 0, 255)])
            #Multicolor background
            elif attr_D == 2:
                color_tuple = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            #White background
            else:
                color_tuple = (255, 255, 255)

            code.append(color_tuple)

        #Background pattern
        if attr_idx == 1:
            pattern = None
            if attr_D == 1:
                # pattern = random.choice([None, 'linear', 'dots'])
                pattern = str(random.choice(texture_noise))

            code.append(pattern)

        #Shape type
        if attr_idx == 2:
            shape_type = None
            if attr_D == 1:
                shape_type = str(random.randint(0, 9))
            code.append(shape_type)

        #Shape color
        if attr_idx == 3:
            color_tuple = None
            if shape_type:
                #Grey shape
                if attr_D == 1:
                    color = random.randint(0, 200)
                    color_tuple = (color, color, color)
                # #Primary colors
                # elif attr_D == 2:
                #     color_tuple = random.choice([(255, 0, 0), (0, 255, 0),(0, 0, 255)])
                #Multicolor shape
                elif attr_D == 2:
                    color_tuple = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                #Black shape
                else:
                    color_tuple = (0, 0, 0)

                if color_tuple == code[0]:
                    color_tuple = (255, 255, 255) #white shape

            code.append(color_tuple)

        #Shape pattern
        if attr_idx == 4:
            pattern = None
            if shape_type:
                if attr_D == 1:
                    pattern = str(random.choice(texture_noise))
            code.append(pattern)

        #Shape size
        if attr_idx == 5:
            shape_size = None
            if shape_type:
                if attr_D == 1:
                    shape_size = round(random.uniform(0.7, 1),2)
            code.append(shape_size)

        #Rotation
        if attr_idx == 6:
            rotation = None
            if shape_type:
                if attr_D == 1:
                    rotation = random.randint(0, 360)
            code.append(rotation)

        #Image filtering
        if attr_idx == 7:
            filter = None
            if attr_D == 1:
                # filter = random.choices(['blur', 'smooth', 'noise', 'fog', None], weights=[2, 2, 2, 2, 1])[0]
                filter = random.choices(['blur', 'smooth', None], weights=[2, 2, 1])[0]
            code.append(filter)

        #Shape count
        if attr_idx == 8:
            shapes_count = None
            if shape_type:
                shapes_count = 1
                if attr_D == 1:
                    shapes_count = random.randint(2, 4)
            code.append(shapes_count)

    return code
    

"""Variables"""

"""Gielis' superformula coefficients used in this study"""
#a, b, y, z, n1, n2, n3
super_shapes = [
    (2,2,2,2,2,2,2), #Circle
    (1,1,4,4,1,1,1), #Square
    (1,1,3,3,0.5,1,1), #Triangle
    (1,1,5,5,1.6,1,1), #Pentagon
    (1,1,6,6,2.4,1,1), #Hexagon
    (1,1.5,4,4,1,1,1), #Rhombus
    (0.5,1,4,4,100,100,100), #Rectangle
    (1,1,2,2,2,7,7), #Oval
    (1,1,3,3,-0.5,1,1), #Trefoil
    (1,1,4,4,-0.7,1,1), #Quatrefoil
    (1,1,4,4,30,15,15), #Smoothed square1
    (1,1,4,4,12,15,15), #Smoothed square2
    (1,1,6,6,-1.5,1,1), #Flower1
    (81,1.045,16,16,0.2,0,30), #Flower2
    (1,1,2,10,-1.5,1,1), #Flower3
    (1,1,8,8,0.5,0.5,8), #Flower4
    (81,1,16,16,0.1,0,10), #Flower5
    (1,1,6,6,40,90,90), #Star1
    (1,1,6,6,60,90,90), #Star2
    (1,1,5,5,30,90,90), #Star3
    (1,1,5,5,3,10,10), #Star4
    (1,1,2,10,1.5,1,1), #Skewed star
    (1,1,5,5,2,7,7), #Smoothed star
    (1,1,5,5,2,13,13), #Smoothed star
    (1,1,5,5,1,1,1), #Pointed star
    (1,1,8,8,1,1,1), #Pointed star
    (1,1,3,3,4.5,10,10), #Smoothed Triangle
    (3,3,6,6,60,50,20), #Smoothed Triangle
    (1,1,3,3,0.5,0.5,0.5), #Pointed triangle
    (1,1,16,16,0.5,0.5,16), #Virus1
    (2,1,30,30,10,10,35), #Virus2
    (1,1,30,30,6,15,15), #Virus3
    (1,1,30,30,6,20,20), #Virus4
    (1,1,30,30,5,30,30), #Virus5
    (2,2,30,30,50,1,20), #Virus6
    (1,1,2,2,0.5,0.5,0.5), #Eye1
    (1,1,2,2,1,0.7,0.7), #Eye2
    (1,1,4,4,0.5,0.5,4), #spinning top1
    (1,1,4,4,1,1,4), #spinning top2
    (1.5,2,8,8,0.9,1.8,0.8), #spinning top3
    (1,1,4,4,1,1,2), #spinning top4
    (1,1,4,4,0.5,0.5,10), #spinning top4
    (10,1,6,6,1,1,50), #deltas
    (1,5,4,4,50,100,50), #Hourglass1
    (1,10,4,4,50,100,50), #Hourglass2
    (7,5,2,1,100,20,80), #drop
    (1,1,2,44,-0.2,1,1), #sea shell1
    (1,3,2,44,-0.2,1,1), #sea shell2
    (1,1,2,44,-0.3,1,1), #sea shell3
    (1,1,2,44,-0.2,1,0.3), #sea shell4
                ]
print("Number of shapes:", len(super_shapes))


"""Noise pattern coefficients used in this study"""
#noise_type, scale, octaves, persistence, lacunarity, seed
texture_noise = [
      ('simplex',3.9,8,0.9,2.5,4),#granite
      #('simplex',7.1,9,0.1,2.8,36),#caves1
      ('simplex',10.5,6,0.3,2.8,40),#caves2
      #('perlin',52.2,5,0.6,2.6,15),#dark night
      ('perlin',73.3,1,1.0,2.5,9),#lights1
      ('perlin',26.2,1,0.3,2.7,29),#lights2
      ('simplex',56.2,4,0.8,2.3,10),#fire
      #('perlin',13.8,2,0.3,2.1,13),#cloud
      ('perlin',10.7,10,0.1,2.7,34),#water
      ('perlin',29.2,4,0.9,1.7,26),#ceramic
      ('perlin',96.8,1,0.1,2.5,53),#stage
      ('perlin',66.4,9,1.0,2.3,96),#metal
      ('perlin',37.3,10,0.9,1.5,33),#rays
      ('perlin',8.0,8,0.9,2.4,3),#grass
]

print("Number of textures:", len(texture_noise))