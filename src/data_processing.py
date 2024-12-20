# process_data.py
from prepare_data import file_processor
import os
from glob import glob
from natsort import natsorted

def load_prepared_data(args, folders):
    file_processor(folders, args)
    sorted_data_files = natsorted(glob(os.path.join(folders.prepared_data_folder, '*')))

    files = sorted_data_files[args.starting_index:args.ending_index]
    text_chunks = []
    chunk_count = 0
    for file in files:
        print('current file: ', file)
        with open(file, 'r') as f:
            text_chunk = f.read()
            text_chunks.append(text_chunk)
            chunk_count += 1
    return text_chunks, chunk_count
