# process_data.py
from prepare_data import file_processor
import os
from glob import glob
from natsort import natsorted

def load_prepared_data(args, folders):
    file_processor(folders, args)
    sorted_data_files = natsorted(glob(os.path.join(folders.prepared_data_folder, '*')))

    files = sorted_data_files[args.starting_index:args.ending_index]
    all_data = []
    file_contents = []
    chunk_count = 0
    for file in files:
        with open(file, 'r') as f:
            file_content = f.read()
            all_data.append(file_content)
            file_contents.append(file_content)
            chunk_count += 1
    return all_data, file_contents, chunk_count

