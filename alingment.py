import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os, shutil
import numpy as np
import operator
import re
from math import ceil, inf
from statistics import mean
import cv2
from skimage import io
from skimage import draw
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.filters import threshold_otsu
import pytesseract
from difflib import SequenceMatcher
from heapq import *
from utils import save_alignments 
from tqdm import tqdm
import configs

"""
Text alignment
 the MiM method

"""


DOCUMENTS_FOLDER = configs.LINE_FOLDER
GT_FOLDER = configs.GT_FOLDER
OUT_FOLDER = configs.OUT_MIM_FOLDER
OUT_FILE = os.path.join(OUT_FOLDER, configs.OUT_MIM_FILENAME)
SAVE_OUT_IMG = False # save example of alignments in images

### PARAMETERS
THRESH_FIRST_SEGMENTATION = 0      # number of black pixel in the projection for the first word segmentation
MINIMUM_BB_SIZE = 5                # Minumun length for a bb
BLACLIST_CHARACTER = ".,:;!?/\"'"  # List of character to rmeove from the GT string of the row
CLEAN_TRANSCRIPT = True            # If True, the transcript of all words are "clened" from characters in blacklist
PARENTESIS = False                 # If True, in the consistency check are not considered letters in parentesis () 
FUSE_BB = True                     # If true, the bb are fused together while the num ob bb is less then equeal to the number of words in transcription
ACW_THRESH_MAX = 10                # threshold for max values for the consistency check
ACW_THRESH_MIN = 10                # threshold for min values for the consistency check
ACW_THRESH_CORR_A = 3              # Min number of letter to apply correction to acw thrasholds
ACW_THRESH_CORR_B = 0.005          # Percentage of correction for each character in the acw thresholds
EXTIMATE_ACW_THRS = False          # If True, ACW_THRESH_MAX and _MIN are estimated on the current document
USE_OCR = False                    # use OCR engine
ALLIGNMENT_MODE = 1                # 0=MiM - 1=Forward 
ORIGINAL = False                   # Test the original IGS method


### CONSTANTS
CONST_LESS = 2
CONST_GREAT = 1
CONST_OK = 0

def get_OCR(image, custom_config=r'--oem 3 --psm 6'):
    """
    --psm NUM             Specify page segmentation mode.
        0    Orientation and script detection (OSD) only.
        1    Automatic page segmentation with OSD.
        2    Automatic page segmentation, but no OSD, or OCR.
        3    Fully automatic page segmentation, but no OSD. (Default)
        4    Assume a single column of text of variable sizes.
        5    Assume a single uniform block of vertically aligned text.
        6    Assume a single uniform block of text.
        7    Treat the image as a single text line.
        8    Treat the image as a single word.
        9    Treat the image as a single word in a circle.
        10    Treat the image as a single character.
        11    Sparse text. Find as much text as possible in no particular order.
        12    Sparse text with OSD.
        13    Raw line. Treat the image as a single text line,
              bypassing hacks that are Tesseract-specific.

    --oem NUM             Specify OCR Engine mode.
        0    Legacy engine only.
        1    Neural nets LSTM engine only.
        2    Legacy + LSTM engines.
        3    Default, based on what is available.
    """
    text = pytesseract.image_to_string(image, config=custom_config)
    text = text.replace("\n", "")
    return text

def vertical_projections(image):
    return np.sum(image, axis=0) 

def get_black_pixel(image, oper=operator.gt):
    projection = vertical_projections(sobel(image))

    bounding_blocks_indices = []
    for index in range(len(projection)):
        black_pixels = projection[index]
        if oper(black_pixels, THRESH_FIRST_SEGMENTATION):
            bounding_blocks_indices.append(index)
    
    return bounding_blocks_indices

def get_white_pixel(image):
    #va sistemato un po - ci sta anche il filtro per i bb piccoli, fai qualcosa anche qui
    return get_black_pixel(image, oper=operator.le)

def get_bb_regions(image, use_oc=False):
    bounding_blocks = []
    transcript_interp = []
    needtobreak = False
    
    bounding_blocks_indices = get_black_pixel(image)

    i=0
    while i < len(bounding_blocks_indices):
        start = bounding_blocks_indices[i]
        end = start
        while i+1<len(bounding_blocks_indices) and bounding_blocks_indices[i+1]-end == 1:
            i += 1
            end = bounding_blocks_indices[i]
        
        if end > image.shape[1]-1:
            end = image.shape[1]-1
            needtobreak = True

        bounding_blocks.append([start, end])

        if needtobreak == True:
            break
        i += 1

    bounding_blocks = filter_bb_list(bounding_blocks, bb_min_size=MINIMUM_BB_SIZE)

    if use_oc:
        for bb in bounding_blocks:
            transcript = get_OCR(image[:,bb[0]:bb[1]])
            transcript_interp.append(transcript)

    return bounding_blocks, transcript_interp

def filter_bb_list(bb_list, bb_min_size):
    """
    Delete all the bbs in the bb_list with zile les than bb_min_size
    """
    new_bb_list = []

    for bb in bb_list:
        if bb[1]-bb[0] >= bb_min_size:
            new_bb_list.append(bb)
    
    return new_bb_list

def save_bb_image(line_image, bb_list, out_path, bb_color=150, word_list=None):
    """
    Save row image with bounding boxes
    """

    if not os.path.exists(os.path.dirname(out_path)):
        os.mkdir(os.path.dirname(out_path))


    bb_image = np.copy(line_image)

    for bb in bb_list:
        rr,cc = draw.rectangle_perimeter(start=(bb[0],0), end=(bb[1], bb_image.shape[0]), shape=bb_image.shape)
        rr,cc = draw.rectangle(start=(0, bb[0]), end=(bb_image.shape[0], bb[1]), shape=bb_image.shape)
        bb_image[rr, cc] = bb_color

    bb_image = np.ubyte(0.7*line_image + 0.3*bb_image)

    if word_list is not None:
        for bb, word in zip(bb_list, word_list):
            bb_image = cv2.putText(bb_image, word, (bb[0],25), cv2.FONT_HERSHEY_SIMPLEX, 1, 20, 2)

    
    io.imsave(out_path, bb_image)

def clean_gt(gt, char_list=BLACLIST_CHARACTER):
    """
    clean the gt string from blacklist characters
    """
    for ch in char_list:
        gt = gt.replace(ch, "")
    gt = re.sub(' +', ' ', gt)
    
    return gt

def extimate_acw_thrs(document_path):
    """
    Extimates the threshholds for the ACW on all the lines of a document
    """
    all_acw = []

    for line_name in os.listdir(document_path):
        doc_folder_name = os.path.basename(document_path)

        image = io.imread(os.path.join(document_path, line_name))
        gray_image = rgb2gray(image)
        bin_image = gray_image > threshold_otsu(gray_image)

        # load ground thruth
        gt_file_name = "gt_" + line_name.split(".")[0] + ".txt"
        with open(os.path.join(GT_FOLDER,doc_folder_name,gt_file_name), "r", encoding="utf8") as gt_file:
            line_gt = gt_file.readline().strip()
        line_gt = re.sub(' +', ' ', line_gt)     
        if CLEAN_TRANSCRIPT:
            line_gt = clean_gt(line_gt)
        line_gt = line_gt.strip()
        
        all_acw.append(compute_acw_on_line(bin_image, line_gt))
    
    acw_min_thr = round(mean(all_acw)) - min(all_acw)
    acw_max_thr = max(all_acw) - round(mean(all_acw))

    return acw_max_thr, acw_min_thr

def compute_acw_on_line(bin_image, gt):
    """
    returns the acw computed on the current line of text
    """
    gt_nospace = clean_gt(gt, char_list=" ")
    n_char = len(gt_nospace)
    black_pixel = get_black_pixel(bin_image)
    acw = ceil(len(black_pixel)/n_char)
    
    return acw

def fuse_bb(bb_words_list, line_gt):
    line_gt = clean_gt(line_gt)
    num_trans = len(line_gt.split(" "))

    while len(bb_words_list) > num_trans:
        if len(bb_words_list)<2:
            break
        else:
            min_dist = inf
            bb1_index = -1
            bb2_index = -1
            for i in range(len(bb_words_list)-1):
                test_dist = bb_words_list[i+1][0] - bb_words_list[i][-1]
                if test_dist < min_dist:
                    min_dist = test_dist
                    bb1_index = i
                    bb2_index = i+1

            bb_words_list[bb1_index] = [bb_words_list[bb1_index][0], bb_words_list[bb2_index][-1]]
            del bb_words_list[bb2_index]
    
    return bb_words_list

def delete_brackets(word):
    """
    delete parts in brackets in a word:
       qua(n)tum  ->  quatum 
    """
    new_word = ""
    add = True
    for ch in word:
        if ch == "(":
            add = False
        if add:
            new_word += ch
        if ch == ")":
            add = True
    
    return new_word
        
def consistency_check(word, bb, acw, acw_max_thr=ACW_THRESH_MAX, acw_min_thr=ACW_THRESH_MIN):
    """
    return = 2 (CONST_LESS) -> merge the bounding boxes
    return = 1 (CONST_GREAT) -> merge the transcript
    return = 0 (CONST_OK) -> consistency ok 
    """
    if PARENTESIS:
        if re.match(".*\(.*\).*", word) is not None:
            word = delete_brackets(word)


    thresh_correction = (len(word) - ACW_THRESH_CORR_A) * ACW_THRESH_CORR_B*acw
    if thresh_correction < 0:
        thresh_correction = 0

    current_acw = ceil((bb[1] - bb[0])/len(word))

    if current_acw < acw-acw_min_thr-thresh_correction:
        return CONST_LESS
    elif current_acw > acw+acw_max_thr+thresh_correction:
        return CONST_GREAT
    else:
        return CONST_OK

def box_segmentation(current_bb, line_img, n_cutting_edges=1):
    """
    detect the cutting edje for a bb. 
    The cutting edje is in the corrispondence of the minimum of the horizontal projection of blak pixel of the bb
    """
    projection = vertical_projections(sobel(line_img))[current_bb[0]:current_bb[1]]
    cutting_edje = projection.argsort()[:n_cutting_edges] + current_bb[0]

    return cutting_edje

def forward_allignment(bb_words_list, bin_line_img, line_gt, acw, acw_max_thr=ACW_THRESH_MAX, acw_min_thr=ACW_THRESH_MIN, use_oc=False):
    """
    Forward allignment algorithm ,from left to wright
    """
    all_words = line_gt.split(" ")

    current_bbIndex_left = 0
    current_wordIndex_left = 0

    allignments_bb = []
    allignments_word = []

    
    while current_bbIndex_left<len(bb_words_list) and current_wordIndex_left<len(all_words):
        current_word = all_words[current_wordIndex_left]
        current_bb = bb_words_list[current_bbIndex_left]

        test_consistency = consistency_check(current_word, current_bb, acw, acw_max_thr=acw_max_thr, acw_min_thr=acw_min_thr)
        while(test_consistency != CONST_OK and current_wordIndex_left<len(all_words) and current_bbIndex_left< len(bb_words_list)):
            if test_consistency == CONST_LESS:
                #merge bbs
                current_bbIndex_left += 1
                if current_bbIndex_left < len(bb_words_list):
                    current_bb = [current_bb[0], bb_words_list[current_bbIndex_left][1]]
            if test_consistency == CONST_GREAT:
                #test segment a box:
                cutting_edje = box_segmentation(bb_words_list[current_bbIndex_left], bin_line_img)[0]
                subtest_consistency = consistency_check(current_word, [current_bb[0], cutting_edje], acw,acw_max_thr=acw_max_thr, acw_min_thr=acw_min_thr)
                if subtest_consistency == CONST_OK:
                    #create new bb
                    current_bb = [current_bb[0], cutting_edje-1]
                    bb_words_list[current_bbIndex_left][0] = cutting_edje+1
                    current_bbIndex_left -= 1
                else:
                    # merge transcript
                    current_wordIndex_left += 1
                    if current_wordIndex_left < len(all_words):
                        current_word = current_word + " " +all_words[current_wordIndex_left]
            test_consistency = consistency_check(current_word, current_bb, acw,acw_max_thr=acw_max_thr, acw_min_thr=acw_min_thr)
        current_bbIndex_left += 1
        current_wordIndex_left += 1

        if use_oc:
            transcript = get_OCR(image[:,current_bb[0]:current_bb[1]])
            if len(transcript) > 0:
                #transcript = clean_gt(transcript)
                transcript_consistency = consistency_check(transcript, current_bb, acw,acw_max_thr=acw_max_thr, acw_min_thr=acw_min_thr)
                similarity = SequenceMatcher(None, transcript, current_word).ratio()
                print(f"transcript[{transcript}] - allign=[{current_word}] test={transcript_consistency}-{test_consistency} - simi:{similarity}]")

                #if similarity < 0.5:
                #    plt.imshow(image[:,current_bb[0]:current_bb[1]])
                #    plt.show()
                #    print()

                # if transcript_consistency == CONST_LESS:
                #     plt.imshow(image[:,current_bb[0]:current_bb[1]])
                #     plt.show()
                    
                #     cutting_edje = box_segmentation(current_bb, bin_line_img)[0]
                #     subtest_consistency = consistency_check(current_word, [current_bb[0], cutting_edje], acw,acw_max_thr=acw_max_thr, acw_min_thr=acw_min_thr)
                #     plt.imshow(image[:,current_bb[0]:cutting_edje])
                #     plt.show()
                #     if subtest_consistency == CONST_OK:
                #         #create new bb
                #         current_bb = [current_bb[0], cutting_edje-1]
                #         bb_words_list[current_bbIndex_left][0] = cutting_edje+1
                #         current_bbIndex_left -= 1
               


        allignments_bb.append(current_bb)
        allignments_word.append(current_word)

    if current_bbIndex_left <= len(bb_words_list):
        allignments_bb[-1][1] = bb_words_list[-1][1]
    
    return allignments_bb, allignments_word

def mim_allignment(bb_words_list, bin_line_img, line_gt, acw, acw_max_thr=ACW_THRESH_MAX, acw_min_thr=ACW_THRESH_MIN):
    """
    MiM allignment algorithm
    """
    all_words = line_gt.split(" ")

    current_bbIndex_left = 0
    current_bbIndex_right = len(bb_words_list)-1

    current_wordIndex_left = 0
    current_wordIndex_right = len(all_words)-1

    left_allignments_bb = []
    right_allignments_bb = []
    left_allignments_word = []
    right_allignments_word = []

    while current_bbIndex_left <= current_bbIndex_right and current_wordIndex_left <= current_wordIndex_right:
        #if current_wordIndex_left < current_wordIndex_right:
        #top left
        current_word = all_words[current_wordIndex_left]
        current_bb = bb_words_list[current_bbIndex_left]

        test_consistency = consistency_check(current_word, current_bb, acw, acw_max_thr=acw_max_thr, acw_min_thr=acw_min_thr)
        while(test_consistency != CONST_OK and current_wordIndex_left<len(all_words) and current_bbIndex_left<len(bb_words_list)):
            if test_consistency == CONST_LESS:
                #merge bbs
                current_bbIndex_left += 1
                if current_bbIndex_left<len(bb_words_list):
                    current_bb = [current_bb[0], bb_words_list[current_bbIndex_left][1]]
            if test_consistency == CONST_GREAT:
                #test segment a box:
                cutting_edje = box_segmentation(bb_words_list[current_bbIndex_left], bin_line_img)[0]
                subtest_consistency = consistency_check(current_word, [current_bb[0], cutting_edje], acw, acw_max_thr=acw_max_thr, acw_min_thr=acw_min_thr)
                if subtest_consistency == CONST_OK:
                    #create new bb
                    current_bb = [current_bb[0], cutting_edje-1]
                    bb_words_list[current_bbIndex_left][0] = cutting_edje+1
                    current_bbIndex_left -= 1
                else:
                    # merge transcript
                    current_wordIndex_left += 1
                    if current_bbIndex_left<len(all_words) and current_wordIndex_left<len(all_words):
                        current_word = current_word + " " +all_words[current_wordIndex_left]
            test_consistency = consistency_check(current_word, current_bb, acw, acw_max_thr=acw_max_thr, acw_min_thr=acw_min_thr)
        current_bbIndex_left += 1
        current_wordIndex_left += 1

        left_allignments_bb.append(current_bb)
        left_allignments_word.append(current_word)

        if current_bbIndex_left < current_bbIndex_right and current_wordIndex_left < current_wordIndex_right:
        #if current_wordIndex_left < current_wordIndex_right:
            #top right
            current_word = all_words[current_wordIndex_right]
            current_bb = bb_words_list[current_bbIndex_right]

            test_consistency = consistency_check(current_word, current_bb, acw, acw_max_thr=acw_max_thr, acw_min_thr=acw_min_thr)
            while(test_consistency != CONST_OK and current_wordIndex_right > 0 and 0 < current_bbIndex_right < len(bb_words_list)):
                if test_consistency == CONST_LESS:
                    #merge bbs
                    current_bbIndex_right -= 1
                    if current_bbIndex_right > 0:
                        current_bb = [bb_words_list[current_bbIndex_right][0], current_bb[1]]
                if test_consistency == CONST_GREAT:
                    #test segment a box:
                    cutting_edje = box_segmentation(bb_words_list[current_bbIndex_right], bin_line_img)[0]
                    subtest_consistency = consistency_check(current_word, [cutting_edje, current_bb[1]], acw, acw_max_thr=acw_max_thr, acw_min_thr=acw_min_thr)
                    if subtest_consistency == CONST_OK:
                        #create new bb
                        current_bb = [cutting_edje+1, current_bb[1]]
                        bb_words_list[current_bbIndex_right][1] = cutting_edje-1
                        current_bbIndex_right += 1
                    else:
                        # merge transcript
                        current_wordIndex_right -= 1
                        if current_wordIndex_right>0:
                            current_word = all_words[current_wordIndex_right] + " " + current_word
                test_consistency = consistency_check(current_word, current_bb, acw, acw_max_thr=acw_max_thr, acw_min_thr=acw_min_thr)
            current_bbIndex_right -= 1
            current_wordIndex_right -= 1

            right_allignments_bb.append(current_bb)
            right_allignments_word.append(current_word)
        
    right_allignments_bb.reverse()
    right_allignments_word.reverse()

    assign_last_bb(bb_words_list, left_allignments_bb, right_allignments_bb)
    
    return left_allignments_bb + right_allignments_bb, left_allignments_word + right_allignments_word

def forward_allignment_original(bb_words_list, line_gt, acw, acw_max_thr=ACW_THRESH_MAX, acw_min_thr=ACW_THRESH_MIN):
    """
    Forward allignment algorithm ,from left to wright
    """
    all_words = line_gt.split(" ")

    current_bbIndex_left = 0
    current_wordIndex_left = 0

    allignments_bb = []
    allignments_word = []

    while current_bbIndex_left<len(bb_words_list) and current_wordIndex_left<len(all_words):
        current_word = all_words[current_wordIndex_left]
        current_bb = bb_words_list[current_bbIndex_left]

        test_consistency = consistency_check(current_word, current_bb, acw, acw_max_thr=acw_max_thr, acw_min_thr=acw_min_thr)
        while(test_consistency != CONST_OK and current_wordIndex_left<len(all_words) and current_bbIndex_left< len(bb_words_list)):
            if test_consistency == CONST_LESS:
                #merge bbs
                current_bbIndex_left += 1
                if current_bbIndex_left < len(bb_words_list):
                    current_bb = [current_bb[0], bb_words_list[current_bbIndex_left][1]]
            if test_consistency == CONST_GREAT:
                # merge transcript
                current_wordIndex_left += 1
                if current_wordIndex_left < len(all_words):
                    current_word = current_word + " " +all_words[current_wordIndex_left]
            test_consistency = consistency_check(current_word, current_bb, acw,acw_max_thr=acw_max_thr, acw_min_thr=acw_min_thr)
        current_bbIndex_left += 1
        current_wordIndex_left += 1

        allignments_bb.append(current_bb)
        allignments_word.append(current_word)

    if current_bbIndex_left <= len(bb_words_list):
        allignments_bb[-1][1] = bb_words_list[-1][1]
    
    return allignments_bb, allignments_word

def mim_allignment_original(bb_words_list, line_gt, acw, acw_max_thr=ACW_THRESH_MAX, acw_min_thr=ACW_THRESH_MIN):
    """
    MiM allignment algorithm
    """
    all_words = line_gt.split(" ")

    current_bbIndex_left = 0
    current_bbIndex_right = len(bb_words_list)-1

    current_wordIndex_left = 0
    current_wordIndex_right = len(all_words)-1

    left_allignments_bb = []
    right_allignments_bb = []
    left_allignments_word = []
    right_allignments_word = []

    while current_bbIndex_left <= current_bbIndex_right and current_wordIndex_left <= current_wordIndex_right:
        #if current_wordIndex_left < current_wordIndex_right:
        #top left
        current_word = all_words[current_wordIndex_left]
        current_bb = bb_words_list[current_bbIndex_left]

        test_consistency = consistency_check(current_word, current_bb, acw, acw_max_thr=acw_max_thr, acw_min_thr=acw_min_thr)
        while(test_consistency != CONST_OK and current_wordIndex_left<len(all_words) and current_bbIndex_left<len(bb_words_list)):
            if test_consistency == CONST_LESS:
                #merge bbs
                current_bbIndex_left += 1
                if current_bbIndex_left<len(bb_words_list):
                    current_bb = [current_bb[0], bb_words_list[current_bbIndex_left][1]]
            if test_consistency == CONST_GREAT:
                # merge transcript
                current_wordIndex_left += 1
                if current_bbIndex_left<len(all_words) and current_wordIndex_left<len(all_words):
                    current_word = current_word + " " +all_words[current_wordIndex_left]
            test_consistency = consistency_check(current_word, current_bb, acw, acw_max_thr=acw_max_thr, acw_min_thr=acw_min_thr)
        current_bbIndex_left += 1
        current_wordIndex_left += 1

        left_allignments_bb.append(current_bb)
        left_allignments_word.append(current_word)

        if current_bbIndex_left < current_bbIndex_right and current_wordIndex_left < current_wordIndex_right:
        #if current_wordIndex_left < current_wordIndex_right:
            #top right
            current_word = all_words[current_wordIndex_right]
            current_bb = bb_words_list[current_bbIndex_right]

            test_consistency = consistency_check(current_word, current_bb, acw, acw_max_thr=acw_max_thr, acw_min_thr=acw_min_thr)
            while(test_consistency != CONST_OK and current_wordIndex_right > 0 and 0 < current_bbIndex_right < len(bb_words_list)):
                if test_consistency == CONST_LESS:
                    #merge bbs
                    current_bbIndex_right -= 1
                    if current_bbIndex_right > 0:
                        current_bb = [bb_words_list[current_bbIndex_right][0], current_bb[1]]
                if test_consistency == CONST_GREAT:
                    # merge transcript
                    current_wordIndex_right -= 1
                    if current_wordIndex_right>0:
                        current_word = all_words[current_wordIndex_right] + " " + current_word
                test_consistency = consistency_check(current_word, current_bb, acw, acw_max_thr=acw_max_thr, acw_min_thr=acw_min_thr)
            current_bbIndex_right -= 1
            current_wordIndex_right -= 1

            right_allignments_bb.append(current_bb)
            right_allignments_word.append(current_word)
        
    right_allignments_bb.reverse()
    right_allignments_word.reverse()

    assign_last_bb(bb_words_list, left_allignments_bb, right_allignments_bb)
    
    return left_allignments_bb + right_allignments_bb, left_allignments_word + right_allignments_word

def _overlap(box1, box2):
    if box2[0] < box1[1]:
        return True
    return False

def handle_overlapped_box(aligns):
    """
    correct the overlapped alignment computed by the MiM algotithm
    """
    for doc_id in aligns:
        lines_dic = aligns[doc_id]
        for line_id in lines_dic:
            boxes, transcrips = lines_dic[line_id]

            for ind in range(len(boxes)-1):
                box = boxes[ind]
                box_next = boxes[ind+1]

                trans = transcrips[ind]
                trans_next = transcrips[ind+1]

                if _overlap(box, box_next):
                    box1 = [box[0], box_next[0]]
                    box2 = [box_next[0]+1, box_next[1]]
                    boxes[ind] = box1
                    boxes[ind+1] = box2

                    tr1 = trans.removesuffix(trans_next).rstrip()
                    transcrips[ind] = tr1

def assign_last_bb(bb_words_list, left_allignments_bb, right_allignments_bb):
    """
    forse va riscritta un po' meglio
    controlla quali box non sono stati assegnati alle liste di destra e sinistra, e assegnali al bb più vicini
    """
    last_left_ind = 0
    last_left_end = bb_words_list[last_left_ind][-1]
    while last_left_end < left_allignments_bb[-1][-1]:
        last_left_ind += 1
        last_left_end = bb_words_list[last_left_ind][-1]

    last_right_ind = len(bb_words_list)-1
    last_right_start = bb_words_list[last_right_ind][0]
    while len(right_allignments_bb)>0 and last_right_start > right_allignments_bb[0][0]:
        last_right_ind -= 1
        last_right_start = bb_words_list[last_right_ind][0]

    left_allignments_bb[-1][-1] = bb_words_list[last_left_ind][-1]
    if len(right_allignments_bb)>0:
        right_allignments_bb[0][0] = bb_words_list[last_right_ind][0]


if __name__ == "__main__":
    if os.path.exists(OUT_FOLDER):
        shutil.rmtree(OUT_FOLDER)
    os.makedirs(OUT_FOLDER)

    all_alignment={}


    for doc_folder_name in tqdm(os.listdir(DOCUMENTS_FOLDER)):
        document_path = os.path.join(DOCUMENTS_FOLDER, doc_folder_name)

        if EXTIMATE_ACW_THRS:
            acw_max_thr, acw_min_thr = extimate_acw_thrs(document_path)
        else:
            acw_max_thr = ACW_THRESH_MAX
            acw_min_thr = ACW_THRESH_MIN

        line_alignments={}

        for line_name in os.listdir(document_path):
            # load image
            image = io.imread(os.path.join(document_path, line_name))
            gray_image = rgb2gray(image)
            bin_image = gray_image > threshold_otsu(gray_image)

            # load ground thruth
            gt_file_name = "gt_" + line_name.split(".")[0] + "_" + doc_folder_name + ".txt"
            with open(os.path.join(GT_FOLDER,doc_folder_name,gt_file_name), "r", encoding="utf8") as gt_file:
                line_gt = gt_file.readline().strip()
            if CLEAN_TRANSCRIPT:
                line_gt = clean_gt(line_gt)
            line_gt = re.sub(' +', ' ', line_gt)
            line_gt = line_gt.strip()

            #first word segemtation
            bb_words_list, transcript_bb_list = get_bb_regions(bin_image, use_oc=USE_OCR)
            #save_bb_image(image, bb_words_list, os.path.join(OUT_FOLDER, doc_folder_name, "bb_"+line_name), word_list=transcript_bb_list)

            if FUSE_BB:
                bb_words_list = fuse_bb(bb_words_list, line_gt)
                #save_bb_image(image, bb_words_list, os.path.join(OUT_FOLDER, doc_folder_name, "bb_post_"+line_name), word_list=transcript_bb_list)


            # alignment algorithm
            acw = compute_acw_on_line(bin_image, line_gt)
            if ALLIGNMENT_MODE == 0:
                if ORIGINAL:
                    allign_bb_list, allign_bb_word = mim_allignment_original(bb_words_list, line_gt, acw, acw_max_thr=acw_max_thr, acw_min_thr=acw_min_thr)
                else:
                    allign_bb_list, allign_bb_word = mim_allignment(bb_words_list, bin_image, line_gt, acw, acw_max_thr=acw_max_thr, acw_min_thr=acw_min_thr)
            elif ALLIGNMENT_MODE == 1:
                if ORIGINAL:
                    allign_bb_list, allign_bb_word = forward_allignment_original(bb_words_list, line_gt, acw, acw_max_thr=acw_max_thr, acw_min_thr=acw_min_thr)
                else:
                    allign_bb_list, allign_bb_word = forward_allignment(bb_words_list, bin_image, line_gt, acw, acw_max_thr=acw_max_thr, acw_min_thr=acw_min_thr, use_oc=USE_OCR)

            #print(line_gt)
            #for bb, word in zip(allign_bb_list, allign_bb_word):
            #    print(bb, word)
            if SAVE_OUT_IMG:
                save_bb_image(image, allign_bb_list, os.path.join(OUT_FOLDER, doc_folder_name, "all_"+line_name), word_list=allign_bb_word)

            line_alignments[line_name] = (allign_bb_list, allign_bb_word)

        all_alignment[doc_folder_name] = line_alignments

    # correct overlapped alignments
    handle_overlapped_box(all_alignment)
    
    # save aligments in file
    save_alignments(all_alignment, OUT_FILE)

    print("Done")
