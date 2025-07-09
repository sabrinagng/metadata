import json
import argparse

import numpy as np

from load import NinaproLoader

def generate_subject_codelist(index_start: int, index_end: int) -> list:
    return [f's{i}' for i in range(index_start, index_end + 1)]

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compile Ninapro DB2/DB3 data into a JSON file.')
    parser.add_argument('--group', type=str, default='DB2', choices=['DB2', 'DB3'],
                        help='Specify the Ninapro dataset group to compile (DB2 or DB3).')
    parser.add_argument('--all-subjects', action='store_true', help='Compile data for all subjects from s1 to s40.')
    parser.add_argument('--index-start', type=int, default=1, help='Starting index for subject codes.')
    parser.add_argument('--index-end', type=int, default=2, help='Ending index for subject codes (exclusive).')
    args = parser.parse_args()
    
    loader = NinaproLoader(group=args.group)
    
    if args.all_subjects:
        if args.group == 'DB2':
            subject_codelist = generate_subject_codelist(1, 40)
            json_dump_file_path = 'data/Ninapro/DB2_emg_only_all_subjects.json'
        else:
            subject_codelist = generate_subject_codelist(1, 11)
            json_dump_file_path = 'data/Ninapro/DB3_emg_only_all_subjects.json'
    else:
        subject_codelist = generate_subject_codelist(args.index_start, args.index_end)
        if args.group == 'DB2':
            json_dump_file_path = f'data/Ninapro/DB2_emg_only_s{args.index_start}_to_s{args.index_end}.json'
        else:
            json_dump_file_path = f'data/Ninapro/DB3_emg_only_s{args.index_start}_to_s{args.index_end}.json'
    
    all_subjects_data = []
    for subject_code in subject_codelist:
        loader.inplace_unzip(subject_code)
        loader.load(subject_code)
        
        emg_sections, labels = loader.segment()
        
        subject_data = {
            'subject_code': subject_code,
            'data': []
        }
        
        for i, (section, label) in enumerate(zip(emg_sections, labels)):
            resampled_emg, _ = loader.align_section(section)
            subject_data['data'].append({
                'emg': resampled_emg,
                'label': label
            })
        all_subjects_data.append(subject_data)
    
    with open(json_dump_file_path, 'w') as f:
        json.dump(all_subjects_data, f, cls=NumpyEncoder, indent=4)