# Eddie Du
# 260 986 386 
import os
import skimage.io as io
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import sobel
import matplotlib.pyplot as plt
import math
    
# This function is provided to you. You will need to call it.
# You should not need to modify it.
def seedfill(im, seed_row, seed_col, fill_color,bckg):
    """ Runs the seedfill algorithm
    Args:
        im (graytone image): The image on which to perform the seedfill algorithm
        seed_row (int) and seed_col (int): position of the seed pixel
        fill_color (float between 0 and 1): Color for the fill
        bckg (float between 0 and 1): Color of the background, to be filled
    Returns: 
        int: Number of pixels filled
    Behavior:
        Modifies the greytone of some pixels by performing seedfill
    """
    
    # check that the image if a greyscale image
    if type(im[0,0])!=np.float64:
        raise TypeError("This is not a greyscale image. Aborting.")

    # check that the fill_color is not the same as bckg
    if fill_color==bckg:
        raise ValueError("fill_color can't be the same as bckg")

    
    size=0  # keep track of patch size
    n_row, n_col = im.shape
    front={(seed_row,seed_col)}  # initial front
    while len(front)>0:
        r, c = front.pop()  # remove an element from front
        if im[r, c]==bckg: 
            im[r, c]=fill_color  # color the pixel
            size+=1
            # look at all neighbors
            for i in range(max(0,r-1), min(n_row,r+2)):
                for j in range(max(0,c-1),min(n_col,c+2)):
                    # if background, add to front
                    if im[i,j]==bckg and (i,j) not in front:
                        front.add((i,j))
    return size


# QUESTION 4
def fill_cells(edge_image):
    """ Fills each enclosed region with a different grayscale value
    Args:
        edge_image (grayscale): An image, with black background and
                                white edges
    Returns: 
        A new grayscale image where each close region is filled with a different
        grayscale value
    """
    n_row, n_col = edge_image.shape
    copy_image = edge_image.copy()
    
    m = 0   # number of times a cell has been colored has been used
    
    # Checks each pixel if graytone of black
    for i in range(n_row):
        for j in range(n_col):
            
            # Fill colour to fill cells
            fill_colour = 0.5+(0.001*m)    
            
            # If pixel is black, execute seedfill with new fill_colour
            if copy_image[i,j] == 0:
                    
                    seedfill(copy_image,i,j,fill_colour,0)
                    m += 1      # Number of times cells colored + 1 
                    
    return copy_image

# QUESTION 5
def classify_cells(original_image, labeled_image, \
                   min_size=1000, max_size=5000, \
                   infected_grayscale=0.5, min_infected_percentage=0.02):
    """ Classifies and counts infected and non-infected cells
    Args:
        original_image (grayscale): The original grayscale image
        labeled_image (grayscale): Image where each enclosed region is colored
                       with a different grayscal value
        min_size (int), max_size (int): The min and max size of a region to be 
                                        called a cell
        infected_grayscale (float): Maximum grayscale value for a pixel to be 
                                    called infected
        min_infected_percentage (float): Smallest fraction of dark pixels 
                                         needed to call a cell infected
    Returns: 
        tuple of two sets: (grayscale values of infected cells, grayscale values
                            of uninfected cells)
    """
    n_row, n_col = labeled_image.shape
    # Create a set of the cells
    set_grayscale = {labeled_image[r,c] for r in range(n_row) \
                     for c in range(n_col)}
    
    # Initilize empty sets     
    infected = set()
    not_infected = set()
    
    # Iterates through each gray_scale color
    for i in set_grayscale:
        
        # Initialize counting variables
        n_dark_pixel = 0
        n_light_pixel = 0
        
        # Checks every pixel of labeled_image for gray_scale color
        for l in range(n_row):
            for j in range(n_col):
                
                # If pixel has the same color, checks original grayscale
                # image to see if dark pixel or light pixel
                if labeled_image[l,j] == i:
                    
                    if original_image[l,j] <= infected_grayscale:
                        n_dark_pixel += 1
                    else:
                        n_light_pixel += 1
        
        # Checks if has requirements to be a cell
        if (min_size <= (n_dark_pixel + n_light_pixel) <= max_size):
            
            # Checks if it is an infected cell
            if n_dark_pixel >= min_infected_percentage*\
                (n_dark_pixel+n_light_pixel):
                
                # If infected, add to infected_set
                infected.add(i)
            
            # Else, add to not_infected 
            else:
                not_infected.add(i)
                        
    return (set(infected),set(not_infected))


# QUESTION 6
def annotate_image(color_image, labeled_image, infected, not_infected):
    """ Annotates the cells in the image, using green for uninfected cells
        and red for infected cells.
    Args: Labeled infected cells in red, and uninfected cells in green
        color_image (color image): An image of cells
        labeled_image (grayscale): Image where each closed region is colored
                       with a different grayscal value
        infected (set of float): A set of graytone values of infected cells
        not_infected (set of float): A set of graytone values of non-infcted cells
    Returns: 
        color image: An image with infected cells highlighted in red
             and non-infected cells highlighted in green
    """     
    
    # Positions of neighboring cells
    x_axis = [-1,-1,-1,0,0,1,1,1]
    y_axis = [1,0,-1,1,-1,1,0,-1]
    n_row, n_col = labeled_image.shape
    
    # Iterates through infected_cells colors
    for color in infected:
        
        # Checks every pixel in labeled_image for the color
        for i in range(n_row):
            for j in range(n_col):
                if labeled_image[i,j] == color:
                    
                    # Checks if neighboring cells color == white 
                    for p in range(len(x_axis)):
                        
                        # If there are, then change original pixel to red
                        if labeled_image[i + x_axis[p], j+y_axis[p]] == 1:
                            color_image[i,j] = (255,0,0)
                            
    # Iterates through not_infected_cells colors                        
    for color in not_infected:
        
        # Checks every pixel in labeled_image for the color
        for i in range(n_row):
            for j in range(n_col):
                if labeled_image[i,j] == color:
                    
                    # Checks if neighboring cell color == white
                    for p in range(len(x_axis)):
                        
                        # If there are, then change original pixel to green
                        if labeled_image[i + x_axis[p], j+y_axis[p]] == 1:
                            color_image[i,j] = (0,255,0)
    
    return color_image

#%%
if __name__ == "__main__":  # do not remove this line   

    # get the directory where this Python file is located
    # and make it the working director (this makes reading the input file easier)
    os.chdir(os.path.dirname(os.path.realpath(__file__)))    

    # QUESTION 1: WRITE YOUR CODE HERE
image = io.imread("malaria-1.jpg")      # Reads image
gray_image = rgb2gray(image)            # Converts image into gray_scale
edge_sobel = sobel(gray_image)          # Converts gray_scale into sobel

io.imsave("Q1_Sobel.jpg",edge_sobel)    # Saves the sobel_image

#%%    
    # QUESTION 2: WRITE YOUR CODE HERE

n_row, n_col = edge_sobel.shape     # Get image dimensions
edge_image = edge_sobel.copy()      # Creates a copy

# For each pixel, checks if edginess >= 0.05 (threshold)
for i in range(n_row):
    for j in range(n_col):
        
        # If it is, then pixel gets value of 1.0
        if (edge_sobel[i,j]) >= 0.05:
            edge_image[i,j] = 1.0
        
        # Else, pixel gets value of 0
        else:
            edge_image[i,j] = 0
        
# Saves image
io.imsave("Q2_Sobel_T_0.05.jpg",edge_image)
            
#%%    
    # QUESTION 3: WRITE YOUR CODE HERE
    
# Copy of image
copy_edge_image = edge_image.copy()
n_row, n_col = gray_image.shape

# References for neighboring pixels
x_axis = [-1,-1,-1,0,0,1,1,1]
y_axis = [1,0,-1,1,-1,1,0,-1]

# Checks if pixel is a Plasmodium cell pixel
for i in range(n_row):
    for j in range(n_col):
        
        # If pixel is less than 0.5, then immidiately gets value of 0
        if gray_image[i,j] < 0.5:
            copy_edge_image[i,j] = 0
        
        # If not less than 0.5, then checks the neighboring pixels
        else:
            for l in range(len(x_axis)):
                
                # Checks if surpasses the image dimensions
                if (i + x_axis[l] in range(n_row)) and (j+ y_axis[l] \
                                                        in range(n_col)):
                    
                    # Checks if neighbor is less than 0.5
                    if gray_image[i+x_axis[l],j+y_axis[l]] < 0.5:
                        copy_edge_image[i,j] = 0

# Saves image
io.imsave("Q3_Sobel_T0.05_clean.jpg", copy_edge_image)


#%%    
    # QUESTION 4: WRITE YOUR CODE CALLING THE FILL_CELLS FUNCTION HERE

# Copy of image
edge_image_copy = copy_edge_image.copy()

# Creates the background with 0.1 graytone
seedfill(edge_image_copy,0,0,0.1,0)

# Executes fill_cells function
new_image = fill_cells(edge_image_copy)

# Saves image
io.imsave("Q4_Sobel_T_0.05_clean_filled.jpg", new_image)

#%%    
    # QUESTION 5: WRITE YOUR CODE CALLING THE CLASSIFY_CELLS FUNCTION HERE
    
two_sets = classify_cells(gray_image,new_image,1000,5000,0.5,0.02)
#%%
    # QUESTION 6: WRITE YOUR CODE CALLING THE ANNOTATE_IMAGE FUNCTION HERE
    
final_image = annotate_image(image,new_image,two_sets[0],two_sets[1])

io.imsave("Q6_annotated.jpg",final_image)
