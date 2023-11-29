import pickle
import os
from glob import glob

train_path_1 = '/media/v100/DATA4/thviet/Pure_dataset/synthesis'
train_path_2 = '/media/v100/DATA4/thviet/Visiting_Data'
valid_path = '/media/v100/DATA4/thviet/Testing_data'

black_list = ['206881_Nguyen Van Nhan', '057755_Dinh Khanh Ha', '232508_Hoang Minh Tuan']
# black_list = ['057755_Dinh Khanh Ha']

staffs = [name for name in os.listdir(train_path_1) if name not in black_list]

for staff in os.listdir(train_path_2):
  if staff not in staffs and staff not in black_list:
    staffs.append(staff)

print('Length staffs:',len(staffs))

labels = {}

for idx, name in enumerate(sorted(staffs)):
  labels[name] = idx

samples = []
for staff in staffs:
  if staff in os.listdir(train_path_1):
    for img in glob(os.path.join(train_path_1, staff, '*')):
        if 'txt' not in img:
            samples.append({'img': img, 'label':labels[staff]})
  elif staff in os.listdir(train_path_2):
    for img in glob(os.path.join(train_path_2, staff, '*')):
        if 'txt' not in img:
            samples.append({'img': img, 'label':labels[staff]})
  
print('Length samples:', len(samples))

with open('/media/v100/DATA4/thviet/KD_research/FaceX-Zoo/training_mode/conventional_training/pkls/train_.pkl','wb') as f:
    pickle.dump(samples, f)
    
test_samples = []
print('Length test staffs:', len(os.listdir(valid_path)))

for staff in os.listdir(valid_path):
  if staff in staffs:
    for img in glob(os.path.join(valid_path, staff, '*')):
        if 'txt' not in img:
            test_samples.append({'img': img, 'label':labels[staff]})

print('Length test samples:', len(test_samples))

with open('/media/v100/DATA4/thviet/KD_research/FaceX-Zoo/training_mode/conventional_training/pkls/test_.pkl','wb') as f:
  pickle.dump(test_samples, f)