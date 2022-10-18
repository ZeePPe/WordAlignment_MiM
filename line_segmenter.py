import os, shutil
import numpy as np
from skimage import io
from skimage import img_as_uint, img_as_ubyte
from skimage.color import rgb2gray, rgba2rgb
from skimage.filters import sobel, gaussian
from skimage.filters import threshold_otsu
from skimage.util import invert
from heapq import *
from tqdm import tqdm
import matplotlib.pyplot as plt

"""
https://github.com/muthuspark/line-segmentation-handwritten-doc/blob/master/A*%20Path%20Planning%20Line%20Segmentation%20Algorithm.ipynb

Va migliorata e aggiunta la divisione in vertical zone
  - detection zona di testo
  - divisione in zone verticale
  - ricostruzione confine di taglio
"""

# Parameters
SIGMA = 10      #sigma for gaussian filter
NUM_OF_VERTICAL_ZONES = 8

IN_FOLDER = os.path.join("03_binarized", "Sauvola")
#IN_FOLDER = os.path.join("03_binarized", "Niblack")
#IN_FOLDER = os.path.join("03_binarized", "Otsu")
IN_FOLDER = os.path.join("04_lines", "002.jpg", "err")
OUT_FOLDER = "04_lines"
OUT_FOLDER =  os.path.join("04_lines", "002.jpg", "2nd_round")

IN_FOLDER = "03_binarized\\new\out\\Sauvola"
OUT_FOLDER = "03_binarized\\new\outseg"

if os.path.exists(OUT_FOLDER):
    shutil.rmtree(OUT_FOLDER)
os.mkdir(OUT_FOLDER)

def horizontal_projections(image):
    return np.sum(image, axis=1) 


def vertical_projections(image):
    return np.sum(image, axis=0) 

def find_peak_regions(hpp, divider=2):
    """
    find peak in orizontal projection
    The “divider” parameter defaults to 2, which means the method will be thresholding the regions in 
    the middle of higher and lower peaks in the HPP.
    """
    threshold = (np.max(hpp)-np.min(hpp))/divider
    peaks = []
    peaks_index = []
    for i, hppv in enumerate(hpp):
        if hppv < threshold:
            peaks.append([i, hppv])
    return peaks


def get_hpp_walking_regions(peaks_index):
    """
    group the peaks into walking windows
    """
    hpp_clusters = []
    cluster = []
    for index, value in enumerate(peaks_index):
        cluster.append(value)

        if index < len(peaks_index)-1 and peaks_index[index+1] - value > 1:
            hpp_clusters.append(cluster)
            cluster = []

        #get the last cluster
        if index == len(peaks_index)-1:
            hpp_clusters.append(cluster)
            cluster = []
            
    return hpp_clusters


def get_binary(img):
    """
    Binarize image with OTSU
    """
    mean = np.mean(img)
    if mean == 0.0 or mean == 1.0:
        return img

    thresh = threshold_otsu(img)
    binary = img <= thresh
    binary = binary*1
    return binary

def path_exists(window_image):
    """
    very basic check first then proceed to A* check
    """
    if 0 in horizontal_projections(window_image):
        return True
    
    padded_window = np.zeros((window_image.shape[0],1))
    world_map = np.hstack((padded_window, np.hstack((window_image,padded_window)) ) )
    path = np.array(astar(world_map, (int(world_map.shape[0]/2), 0), (int(world_map.shape[0]/2), world_map.shape[1])))
    if len(path) > 0:
        return True
    
    return False

def get_road_block_regions(nmap):
    road_blocks = []
    needtobreak = False
    
    for col in range(nmap.shape[1]):
        start = col
        end = col+20 # fized step window
        if end > nmap.shape[1]-1:
            end = nmap.shape[1]-1
            needtobreak = True

        if path_exists(nmap[:, start:end]) == False:
            road_blocks.append(col)

        if needtobreak == True:
            break
            
    return road_blocks

def get_road_block_regions_project(nmap):
    road_blocks = []
    needtobreak = False

    projection = vertical_projections(nmap)
    road_blocks_indices = np.where(projection != 0)[0]

    i=0
    while i < len(road_blocks_indices):
        start = road_blocks_indices[i]
        end = start
        while i+1<len(road_blocks_indices) and road_blocks_indices[i+1]-end == 1:
            i += 1
            end = road_blocks_indices[i]
        
        if end > nmap.shape[1]-1:
            end = nmap.shape[1]-1
            needtobreak = True

        if path_exists(nmap[:, start:end]) == False:
             road_blocks.append([start, end])

        if needtobreak == True:
            break
        i += 1
            
    return road_blocks

def group_the_road_blocks(road_blocks):
    #group the road blocks
    road_blocks_cluster_groups = []
    road_blocks_cluster = []
    size = len(road_blocks)
    for index, value in enumerate(road_blocks):
        road_blocks_cluster.append(value)
        if index < size-1 and (road_blocks[index+1] - road_blocks[index]) > 1:
            road_blocks_cluster_groups.append([road_blocks_cluster[0], road_blocks_cluster[len(road_blocks_cluster)-1]])
            road_blocks_cluster = []

        if index == size-1 and len(road_blocks_cluster) > 0:
            road_blocks_cluster_groups.append([road_blocks_cluster[0], road_blocks_cluster[len(road_blocks_cluster)-1]])
            road_blocks_cluster = []

    return road_blocks_cluster_groups


#a star path planning algorithm 
def heuristic_SED(a, b):
    """
    Squared Euclidean Distance
    """
    return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2

#a star path planning algorithm 
def heuristic_manhattan(a, b):
    """
    Manhattan distance
    """
    return abs((b[0] - a[0])) + abs((b[1] - a[1]))

def astar(array, start, goal, heuristic=heuristic_SED):
    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []

    heappush(oheap, (fscore[start], start))
    
    while oheap:

        current = heappop(oheap)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j            
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:                
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue
                
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
                
            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heappush(oheap, (fscore[neighbor], neighbor))
                
    return []

def extract_line_from_image(image, lower_line, upper_line):
    lower_boundary = np.min(lower_line[:, 0])
    upper_boundary = np.max(upper_line[:, 0])
    img_copy = np.copy(image)
    r, c = img_copy.shape
    for index in range(c-1):
        img_copy[0:lower_line[index, 0], lower_line[index, 1]] = 255
        img_copy[upper_line[index, 0]:r, upper_line[index, 1]] = 255

        #img_copy[lower_line[index, 0], index] = 100
        #img_copy[upper_line[index, 0], index] = 100
    
    return img_copy[lower_boundary:upper_boundary, :]

def extract_line_from_image2(image, lower_line, upper_line):
    """
    extract line from boundary set
    """
    r, c = image.shape

    lower_boundary = np.min(lower_line[:, 0])
    upper_boundary = np.min(upper_line[:, 0])
    imr_row = np.ones((upper_boundary-lower_boundary, c), dtype=np.uint8)*255
   
    for index in range(c-1):
        col = image[lower_line[index, 0]:upper_line[index, 0], index]
        col_start = 0
        col_len = len(col)
        imr_row[col_start:col_len, index] = col # offset!
        #img_copy[0:lower_line[index, 0], index] = 255
        #img_copy[upper_line[index, 0]:r, index] = 255
    
    return imr_row


if __name__ == "__main__":
    for image_name in tqdm(os.listdir(IN_FOLDER)):
        os.mkdir(os.path.join(OUT_FOLDER, image_name))

        image = io.imread(os.path.join(IN_FOLDER, image_name))
        image = rgb2gray(image)

        inverse_image = gaussian(image, sigma=SIGMA)
        inverse_image = inverse_image > threshold_otsu(inverse_image)
        inverse_image = rgb2gray(inverse_image)
        inverse_image = invert(inverse_image)
       
        plt.imshow(image, cmap="gray")
        plt.show()
        
        hpp = horizontal_projections(inverse_image) 
        plt.plot(hpp)
        plt.show()

        peaks = find_peak_regions(hpp)
        peaks_index = np.array(peaks)[:,0].astype(int)

        segmented_img = np.copy(image)
        r,c = segmented_img.shape
        for ri in range(r):
            if ri in peaks_index:
                segmented_img[ri, :] = 0
                
        #plt.figure(figsize=(20,20) )
        #plt.title(image_name)
        #plt.imshow(segmented_img, cmap="gray")
        #plt.show()
        #io.imsave(os.path.join(OUT_FOLDER, image_name, "segment_"+image_name), segmented_img)

        # compute the line cluster basing on oriz projection
        hpp_clusters = get_hpp_walking_regions(peaks_index)

       

        # refine binarization and add doorways  (when there is no path in the row division (total ascender or descender))
        print(f"Binarization {image_name}")
        binary_image = get_binary(image)
        #io.imsave(os.path.join(OUT_FOLDER, image_name, image_name), binary_image)
        for cluster_of_interest in tqdm(hpp_clusters):
            nmap = binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest)-1],:]
            if nmap.shape[0]>0:
                #io.imsave(os.path.join(OUT_FOLDER, image_name, "line_"+image_name), nmap)
                #road_blocks = get_road_block_regions(nmap)
                #road_blocks_cluster_groups = group_the_road_blocks(road_blocks)
                road_blocks_cluster_groups = get_road_block_regions_project(nmap)
                #create the doorways
                for index, road_blocks in enumerate(road_blocks_cluster_groups):
                    window_offset = road_blocks[1]
                    while window_offset < image.shape[1] and np.sum(nmap[:,window_offset], axis=0) != 0:
                        window_offset +=1
                    window_image = nmap[:, road_blocks[0]: window_offset] # +1 OR NOT???
                    binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest)-1],:][:, road_blocks[0]: window_offset][int(window_image.shape[0]/2),:] *= 0
               

        #segment all the lines using the A* algorithm
        print(f"Line Segmentation {image_name}")
        line_segments = []
        for i, cluster_of_interest in tqdm(enumerate(hpp_clusters)):   
            nmap = binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest)-1],:]
            if nmap.shape[0]>0:
                #io.imsave(os.path.join(OUT_FOLDER, image_name, "line_"+str(i)+"_"+image_name), nmap)
                path = np.array(astar(nmap, (int(nmap.shape[0]/2), 0), (int(nmap.shape[0]/2),nmap.shape[1]-1)))
                offset_from_top = cluster_of_interest[0]
                if path.any():
                    path[:,0] += offset_from_top
                    line_segments.append(path)

        # view one line boundary
        cluster_of_interest = hpp_clusters[0]
        ##offset_from_top = cluster_of_interest[0]
        nmap = binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest)-1],:]
        path = np.array(astar(nmap, (int(nmap.shape[0]/2), 0), (int(nmap.shape[0]/2),nmap.shape[1]-1)))
        plt.imshow(invert(nmap), cmap="gray")
        try:
            plt.plot(path[:,1], path[:,0])
        except:
            pass

         # view all line boundary
        fig = plt.figure(figsize=(1, 1))
        offset_from_top = cluster_of_interest[0]
        for path in line_segments:
            plt.plot((path[:,1]), path[:,0], "r-")
        plt.axis("off")
        plt.imshow(image, cmap="gray")
        plt.show()
        #fig.savefig(os.path.join(OUT_FOLDER, "seg_"+image_name))

        ## add an extra line to the line segments array which represents the last bottom row on the image
        last_bottom_row = np.flip(np.column_stack(((np.ones((image.shape[1],))*image.shape[0]), np.arange(image.shape[1]))).astype(int), axis=0)
        line_segments.append(last_bottom_row)

        ## Lets divide the image now by the line segments passing through the image
        line_images = []

        line_count = len(line_segments)
        for line_index in range(line_count-1):
            line_image = extract_line_from_image(image, line_segments[line_index], line_segments[line_index+1])
            line_images.append(line_image)
            #io.imsave(os.path.join(OUT_FOLDER, image_name, str(line_index)+"_"+image_name), img_as_ubyte(line_image))


    print("Done")