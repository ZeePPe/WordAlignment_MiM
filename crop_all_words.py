import os
import shutil
import cv2
from utils import load_aligments
import configs
from tqdm import tqdm

ALIGNMENT_FILE = os.path.join(configs.OUT_MIM_FOLDER, configs.OUT_MIM_FILENAME)
OUT_FOLDER = configs.OUT_WORDS_FOLDER
LINE_FOLDER = configs.LINE_FOLDER

def crop_all_words(aligns, dst_folder="out"):
    if os.path.exists(dst_folder):
        shutil.rmtree(dst_folder)
    os.mkdir(dst_folder)

    for doc_folder, all_lines in tqdm(aligns.items()):
        for line_filename, (boxes, transcriptions) in all_lines.items():
            # leggi immagine
            line_img = cv2.imread(os.path.join(LINE_FOLDER, doc_folder, line_filename), cv2.IMREAD_GRAYSCALE)
             
            inline_pos = 0
            for box, trans in zip(boxes, transcriptions):
                word_filename = doc_folder + "_" + line_filename.split(".")[0] + "_" + str(inline_pos).zfill(2) + "_" + trans + "." + line_filename.split(".")[-1]

                word_img = line_img[: , box[0]:box[1]]
                cv2.imwrite(os.path.join(dst_folder, word_filename), word_img)

                inline_pos += 1


if __name__ == "__main__":
    aligns = load_aligments(ALIGNMENT_FILE)

    crop_all_words(aligns, dst_folder=OUT_FOLDER)

    print("Done!")