import os
import cv2
from utils import load_aligments, save_alignments
import numpy as np
import time
from datetime import datetime
import configs

ALIGNMENT_FILE = os.path.join(configs.OUT_MIM_FOLDER, configs.OUT_MIM_FILENAME)
OUT_FOLDER = configs.OUT_WORDS_FOLDER
LINE_FOLDER = configs.LINE_FOLDER

H = 115

def correct_aligns(aligns, outfile="out"):

    for doc_folder, all_lines in aligns.items():
        for line_filename, (boxes, transcriptions) in all_lines.items():
            # leggi immagine
            line_img = cv2.imread(os.path.join(LINE_FOLDER, doc_folder, line_filename))
             
            for box, trans in zip(boxes, transcriptions):
                #Mouse CLick callback
                def click_event(event, x, y, flags, params):
                    if event == cv2.EVENT_LBUTTONDOWN:
                        # displaying the coordinates
                        new_start = x
                        print(f"[{new_start}")

                        cv2.line(curr_img, (new_start, 0), (new_start, H), (0,0,255), 2) 
                        cv2.imshow('image', curr_img)
                        
                        # write file!!
                        params[0] = new_start
                    
                    # checking for right mouse clicks    
                    if event==cv2.EVENT_RBUTTONDOWN:
                        new_end = x
                        print(f"[   ,{new_end}]")
                
                        cv2.line(curr_img, (new_end, 0), (new_end, H), (255,0,255), 2) 
                        cv2.imshow('image', curr_img)

                        params[1] = new_end
                
                print(box, trans)

                #curr_img = line_img.copy()
                curr_img = np.zeros((line_img.shape[0]+50, line_img.shape[1], line_img.shape[2]), dtype=np.uint8)
                curr_img[0:line_img.shape[0], 0:line_img.shape[1], 0:line_img.shape[2]] = line_img

                cv2.rectangle(curr_img,(box[0],0),(box[1],H),(0,255,0),1)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(curr_img, trans, (10,line_img.shape[0]+40), font, 1, (255, 200, 200), 2)
                
                cv2.imshow('image', curr_img)
                cv2.setMouseCallback('image', click_event, param=box)
                cv2.waitKey(0)

                #save modification
                save_alignments(aligns, outfile)


if __name__ == "__main__":
    aligns = load_aligments(ALIGNMENT_FILE)

    start_time = time.time()
    correct_aligns(aligns, outfile=ALIGNMENT_FILE)
    total_time = time.time() - start_time

    # save time file
    if not os.path.exists(configs.TIME_BASEFOLDER):
        os.mkdir(configs.TIME_BASEFOLDER)

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S__")

    time_filepath = os.path.join(configs.TIME_BASEFOLDER, dt_string+configs.TIME_WORDCORRECTION_FILENAME)
    
    with(open(time_filepath, "w") as timefile):
        timefile.write(f"Correction has required {total_time} seconds")


    print("Done!")


