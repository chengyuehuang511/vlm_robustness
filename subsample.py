"/coc/testnvme/chuang475/projects/vlm_robustness/data/configs/defaults_vqa_raw.yaml"
"/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/vqa_train.json"
"/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/vqa_val.json"

"/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/vqa_val_eval.json"

"/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/v2_OpenEnded_mscoco_val2014_questions.json"
"/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/v2_mscoco_val2014_annotations.json"

import json

vqa_train = json.load(open("/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/vqa_train.json"))
vqa_val = json.load(open("/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/vqa_val.json"))
print(len(vqa_train))
print(len(vqa_val))
# shuffle and subsample 10% of the data
import random
random.seed(0)
vqa_train = random.sample(vqa_train, int(len(vqa_train) * 0.1))
print(len(vqa_train))
print(vqa_train[0].keys())

vqa_val_eval = json.load(open("/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/vqa_val_eval.json"))
print(len(vqa_val_eval))

vqa_val = random.sample(vqa_val, int(len(vqa_val) * 0.1))
print(len(vqa_val))

v2_OpenEnded_mscoco_val2014_questions = json.load(open("/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/v2_OpenEnded_mscoco_val2014_questions.json"))
print(len(v2_OpenEnded_mscoco_val2014_questions['questions']))
print(v2_OpenEnded_mscoco_val2014_questions['questions'][0])

# select the questions that are in the subsampled vqa_val using key=question_id
v2_OpenEnded_mscoco_val2014_questions['questions'] = [q for q in v2_OpenEnded_mscoco_val2014_questions['questions'] if q['question_id'] in [q_['question_id'] for q_ in vqa_val]]
assert len(v2_OpenEnded_mscoco_val2014_questions['questions']) == len(vqa_val)

v2_mscoco_val2014_annotations = json.load(open("/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/v2_mscoco_val2014_annotations.json"))
print(len(v2_mscoco_val2014_annotations['annotations']))

# select the annotations that are in the subsampled vqa_val using key=question_id
v2_mscoco_val2014_annotations['annotations'] = [a for a in v2_mscoco_val2014_annotations['annotations'] if a['question_id'] in [q_['question_id'] for q_ in vqa_val]]
assert len(v2_mscoco_val2014_annotations['annotations']) == len(vqa_val)

# json.dump(vqa_train, open("/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/vqa_train_10.json", "w"))
# json.dump(vqa_val, open("/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/vqa_val_10.json", "w"))
json.dump(v2_OpenEnded_mscoco_val2014_questions, open("/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/v2_OpenEnded_mscoco_val2014_questions_10.json", "w"))
json.dump(v2_mscoco_val2014_annotations, open("/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/v2_mscoco_val2014_annotations_10.json", "w"))

