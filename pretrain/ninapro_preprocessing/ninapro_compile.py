import json
import numpy as np

from ninapro_load import NinaproLoader

def generate_subject_codelist():
    return [f's{i}' for i in range(1, 41)]

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
    loader = NinaproLoader()
    subject_codelist = generate_subject_codelist()
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
            
    with open('data/Ninapro/DB2_emg_only_all_subjects.json', 'w') as f:
        json.dump(all_subjects_data, f, cls=NumpyEncoder, indent=4)