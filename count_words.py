import os, shutil
import re
import configs

N_OF_PAGES = 1
BLACLIST_CHARACTER = ".,:;!?/\"'"

GT_FOLDER = configs.GT_FOLDER

n_of_words = 0
n_of_lines = 0
# lines folder
doc_folder_contetn = os.listdir(GT_FOLDER)

for i in range(N_OF_PAGES):
    doc = doc_folder_contetn[i]
    doc_path = os.path.join(GT_FOLDER, doc)
    for gt_line in os.listdir(doc_path):
        with(open(os.path.join(doc_path,gt_line), "r") as gt_file):
            for line in gt_file.readlines():
                for ch in BLACLIST_CHARACTER: # remove blacklist characters
                    line = line.replace(ch, "")
                line = re.sub(' +', ' ', line) #remove multispace
                line = line.strip()

                n_of_words += len(line.split(" "))
                n_of_lines += 1
    
    i += 1

print(f"Documenti analizzati:{N_OF_PAGES}   -    righe:{n_of_lines}   parole:{n_of_words}")