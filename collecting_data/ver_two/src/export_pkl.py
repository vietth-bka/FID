import os
import pickle
import numpy as np

def export_pkl(pkl_file: str, img_dir: str) -> bool:
    """read images in img_dir, export to pkl_folder. return the actual filename"""

    list_person = []
    for person in os.listdir(img_dir):
        if 'noise' not in person:
            person_path = os.path.join(img_dir, person)
            for txt in os.listdir(person_path):
                if '.txt' in txt:
                    txt_path = os.path.join(person_path, txt)
                    with open(txt_path, 'r') as f:
                        emb = f.read()
                        emb = np.fromstring(
                            emb[1:-1], dtype=np.float32, sep=' ')
                    name, mail_code = person.split('_')
                    list_person.append({'emb': emb,
                                        'mail_code': mail_code,
                                        'full_name': name})

    print(len(list_person))
    with open(pkl_file, 'wb') as f:
        pickle.dump(list_person, f)
    return True

if __name__ == "__main__":
    pkl_folder = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'pkl'))
    if not os.path.exists(pkl_folder):
        os.makedirs(pkl_folder)
    base_pkl = os.path.join(pkl_folder, 'base.pkl')
    img_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'save_image'))
    export_pkl(base_pkl, img_dir)
