import os
import cv2
from PIL import Image, ImageDraw,ImageFont
import copy
from utils import load_aligments, save_alignments
import numpy as np
import time
from datetime import datetime
import configs

ALIGNMENT_FILE = os.path.join(configs.OUT_MIM_FOLDER, configs.OUT_MIM_FILENAME)
OUT_FOLDER = configs.OUT_WORDS_FOLDER 
LINE_FOLDER = configs.LINE_FOLDER

H = configs.H


# all constants

TEST_PREV_LINE = -25
TEST_PREV_DOC  = -25

FONT_REGULAR_PIL = ImageFont.truetype('assets/font/AlteHaasGroteskRegular.ttf', 25)
FONT_BOLD_PIL = ImageFont.truetype('assets/font/AlteHaasGroteskBold.ttf', 30)
FONT_CV = cv2.FONT_HERSHEY_SIMPLEX


def correct_aligns(aligns, outfile="out"):
    count = 1
    all_aligns = _get_aligns_number(aligns)

    from_next_line = False
    from_next_doc = False

    curr_inddoc = 0
    keys_list_docs = list(aligns)

    while curr_inddoc < len(keys_list_docs):
        doc_folder = keys_list_docs[curr_inddoc]
        all_lines = aligns[doc_folder]

    #for doc_folder, all_lines in aligns.items():
        keys_list_lines = list(all_lines)
        curr_indline = 0
        if from_next_doc:
            curr_indline = len(keys_list_lines)-1
            from_next_doc = False
            from_next_line = True

        while 0 <= curr_indline < len(keys_list_lines):
            line_filename = keys_list_lines[curr_indline]
            (boxes, transcriptions) = all_lines[line_filename]

            # leggi immagine
            line_img = cv2.imread(os.path.join(LINE_FOLDER, doc_folder, line_filename))

            H = line_img.shape[0]

            curr_indword = 0
            if from_next_line:
                curr_indword = len(boxes)-1
                from_next_line = False

            while 0 <= curr_indword < len(boxes):
                box = boxes[curr_indword]
                trans = transcriptions[curr_indword]

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
                
                print(f"{box} {trans.ljust(15)}\t---| {count}/{all_aligns} |    \t..{os.path.join(doc_folder, line_filename)}")

                #curr_img = line_img.copy()
                curr_img = np.zeros((line_img.shape[0]+100, line_img.shape[1], line_img.shape[2]), dtype=np.uint8)
                curr_img[0:line_img.shape[0], 0:line_img.shape[1], 0:line_img.shape[2]] = line_img

                cv2.rectangle(curr_img,(box[0],1),(box[1],H),(0,255,0),1)

                #text under box
                #cv2.putText(curr_img, trans, (box[0],line_img.shape[0]+40), FONT_CV, 1, (128, 240, 128), 2) #box
                img = Image.new('RGB', (len(trans)*20, 40), color = (0, 0, 0))
                d = ImageDraw.Draw(img)
                d.text((0,0), trans, font=FONT_BOLD_PIL,  fill=(128, 240, 128))
                img = np.asarray(img)
                bottom_margin = 55
                y_start = curr_img.shape[0]-(40+bottom_margin)
                y_end = curr_img.shape[0]-bottom_margin
                x_start = box[0]
                x_end = box[0]+len(trans)*20
                a = (x_end-x_start)
                b = curr_img.shape[1]
                if x_end >curr_img.shape[1]:
                    x_end = curr_img.shape[1]
                    
                curr_img[y_start:y_end,x_start:x_end, : ] = img[:,:(x_end-x_start),:]
               
                # text bottom trans
                #cv2.putText(curr_img, trans, (10,line_img.shape[0]+90), font, 1, (255, 200, 200), 2) 
                img = Image.new('RGB', (len(trans)*25, 30), color = (0, 0, 0))
                d = ImageDraw.Draw(img)
                d.text((10,0), trans, font=FONT_REGULAR_PIL,  fill=(255,200,200))
                img = np.asarray(img)
                bottom_margin = 1
                curr_img[curr_img.shape[0]-(30+bottom_margin):curr_img.shape[0]-bottom_margin,0:len(trans)*25, : ] = img

                
                # state of progression
                str_curr_position =  f"{count}/{all_aligns}"
                cv2.putText(curr_img, str_curr_position, (line_img.shape[1]-15*len(str_curr_position),line_img.shape[0]+90), FONT_CV, 0.6, (255, 200, 200), 1)
                
                cv2.imshow('image', curr_img)
                cv2.setMouseCallback('image', click_event, param=box)
                key_pressed = cv2.waitKey(0)
                
                if key_pressed == 13:
                    #  ENTER
                    curr_indword += 1
                    count += 1
                elif key_pressed == 8:
                    # backspace
                    count -= 1
                    curr_indword -= 1
                    if curr_indword <0:
                        #curr_indword = 0
                        curr_indword = TEST_PREV_LINE
                        count += 1
                    
                elif key_pressed == 83:
                    # ALT+s
                    # VA RICHIAMATA LA FUNZIONE mesure_performnace()
                    print("-------- SAVE STATE -------")
                    save_alignments(aligns, outfile)
                elif key_pressed == 17:
                    #CTRL+q
                    #quit and save
                    print("-------- SAVE STATE -------")
                    save_alignments(aligns, outfile)
                    return
                    #exit()

                #save modification
                save_alignments(aligns, outfile)

            # new line
            if curr_indword > 0:
                #next line
                curr_indline += 1
            elif curr_indword == TEST_PREV_LINE and curr_indline>0:
                #prev line
                from_next_line = True
                curr_indline -= 1
                count -= 1
            elif curr_indword == TEST_PREV_LINE and curr_indline==0 and curr_inddoc>0:
                #prev doc
                curr_indline = TEST_PREV_DOC
                
        #("new_doc")
        if curr_indline > 0:
            curr_inddoc += 1
        elif curr_indline == TEST_PREV_DOC:
            from_next_doc = True
            curr_inddoc -= 1
            count -= 1

def _get_aligns_number(aligns):
    num_of_aligns = 0

    for doc_res in aligns:
        cur_doc_res = aligns[doc_res]
        for line_res in  cur_doc_res:
            num_of_aligns += len(cur_doc_res[line_res][0])

    return num_of_aligns

def mesure_performnace(original_aligns, correct_aligns):
    n_alignments = 0
    n_correct_aligns = 0

    correct_2 = 0
    correct_3 = 0
    correct_4 = 0
    correct_more = 0
    wrong_2 = 0
    wrong_3 = 0
    wrong_4 = 0
    wrong_more = 0


    for doc_id in original_aligns:
        original_lines_dic = original_aligns[doc_id]
        corrected_lines_dic = correct_aligns[doc_id]
        for line_id in original_lines_dic:
            original_boxes, transcrips = original_lines_dic[line_id]
            corrected_boxes, _ = corrected_lines_dic[line_id]

            for original_box, corrected_box, transcript in zip(original_boxes, corrected_boxes, transcrips):
                n_alignments += 1
                words_in_transcript = transcript.split(" ")
                if original_box[0]==corrected_box[0] and original_box[1]==corrected_box[1]:
                    # correct align
                    n_correct_aligns += 1

                    if len(words_in_transcript) == 1:
                        pass
                    elif len(words_in_transcript) == 2:
                        correct_2 += 1
                    elif len(words_in_transcript) == 3:
                        correct_3 += 1
                    elif len(words_in_transcript) == 4:
                        correct_4 += 1
                    else:
                        correct_more += 1
                    
                else:
                    # wrong align
                    if len(words_in_transcript) == 1:
                        pass
                    elif len(words_in_transcript) == 2:
                        wrong_2 += 1
                    elif len(words_in_transcript) == 3:
                        wrong_3 += 1
                    elif len(words_in_transcript) == 4:
                        wrong_4 += 1
                    else:
                        wrong_more += 1

    return n_alignments, n_correct_aligns, correct_2, correct_3, correct_4, correct_more, wrong_2, wrong_3, wrong_4, wrong_more


if __name__ == "__main__":
    aligns = load_aligments(ALIGNMENT_FILE)
    orig_aligns = copy.deepcopy(aligns)

    start_time = time.time()
    correct_aligns(aligns, outfile=ALIGNMENT_FILE)
    total_time = time.time() - start_time

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S__")
    # Compute performance of the allignment
    n_alignments, n_correct_aligns, correct_2, correct_3, correct_4, correct_more, wrong_2, wrong_3, wrong_4, wrong_more = mesure_performnace(orig_aligns, aligns)
    print(f"\n\nNumber of Alignments:\t{n_alignments}\nCorrect Alignments:\t{n_correct_aligns}\nPrecision:\t{n_correct_aligns/n_alignments}")
    print("Box with more than 1 word:")
    print(f"correct_2:{correct_2} correct_3:{correct_3} correct_4:{correct_4} correct_more:{correct_more}\nwrong_2:{wrong_2} wrong_3:{wrong_3} wrong_4:{wrong_4} wrong_morre:{wrong_more}")

    # save performance and time file
    if not os.path.exists(configs.PERFORMANCE_BASEFOLDER):
        os.mkdir(configs.PERFORMANCE_BASEFOLDER)
    performance_filepath = os.path.join(configs.PERFORMANCE_BASEFOLDER, dt_string+configs.PERFORMANCE_FILENAME)
    with(open(performance_filepath, "w") as performancefile):
        performancefile.write(f"Number of Alignments:\t{n_alignments}\nCorrect Alignments:\t{n_correct_aligns}\nPrecision:\t{n_correct_aligns/n_alignments}\n")
        performancefile.write("Box with more than 1 word:\n")
        performancefile.write(f"correct_2:{correct_2} correct_3:{correct_3} correct_4:{correct_4} correct_more:{correct_more}\nwrong_2:{wrong_2} wrong_2:{wrong_3} wrong_4:{wrong_4} wrong_morre:{wrong_more}\n")

    
    if not os.path.exists(configs.TIME_BASEFOLDER):
        os.mkdir(configs.TIME_BASEFOLDER)
    time_filepath = os.path.join(configs.TIME_BASEFOLDER, dt_string+configs.TIME_WORDCORRECTION_FILENAME)
    with(open(time_filepath, "w") as timefile):
        timefile.write(f"Correction has required {total_time} seconds")


    print("Done!")


