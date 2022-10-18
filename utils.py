import pickle 

def save_alignments(all_alignments, file_name): 
    with open(file_name, 'wb') as handle:
        pickle.dump(all_alignments, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_aligments(file_name):
    with open(file_name, 'rb') as handle:
        all_alignments = pickle.load(handle)
    return all_alignments    