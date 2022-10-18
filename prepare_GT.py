import os
from tqdm import tqdm

INPUT_PATH = "05_GT/006"
NAME_FOLDER = os.path.basename(INPUT_PATH)
GT_FILE = f"gt_{NAME_FOLDER}.txt"

index_line = 0
with open(os.path.join(INPUT_PATH, GT_FILE), "r", encoding="UTF8") as all_gt:
    lines = all_gt.readlines()
    index_char_n = len(str(len(lines)))
    for line in tqdm(lines):
        if line.strip() != "":
            index_str = str(index_line).zfill(index_char_n)
            line_file_name = f"gt_{index_str}_{NAME_FOLDER}.txt"
            index_line += 1

            with( open(os.path.join(INPUT_PATH, line_file_name), "w", encoding="utf8") as out_file):
                out_file.write(line)
