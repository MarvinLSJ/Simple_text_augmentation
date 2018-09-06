import xlsxwriter
import argparse
import pandas as pd
from utils import remove_html, clean, aug_sent

"""

parameters with * is required to be set, others have default values. 

Generate more sentences: lower <aug_threshold> (raise <sub_threshold>)
Generate less sentences: raise <aug_threshold> (lower <sub_threshold>)

Input:
    * file_path: 
        Path of file that will be augmented.
        
    * data_type:
        ['comment' or 'csc']
        Related to label type, comment has 7 levels, csc has 5 levels.
        
    * aug_label:
        Numbers [0-6] for comment, [0-4] for csc.
        
    file_type: 
        Default read from file_path
        ['xlsx' or 'csv']
    
    save_type:
        Default xlsx
        ['xlsx' or 'csv']
        
    thesaurus_path:
        Default thesaurus dictionary path: 'data/thesaurus_dict.txt'
    
    aug_threshold:
        Default set to 1000. 
        Threshold for augmentation amount on a single sentence. 
        If augmentation amount is more than 1000, this sentence will be discard.
    
    sub_threshold: 
        Default set to 10. 
        If there are more than <sub_threshold> words can be substitute, 
        this sentence is too long to execute substitution. 
        These sentences will be saved into <redundant.csv> for future uses. 

"""

parser = argparse.ArgumentParser()

parser.add_argument('--file_path', type=str, required=True, help='Path of file that will be augmented.')
parser.add_argument('--data_type', type=str, required=True, help='comment or csc')
parser.add_argument('--aug_label', type=int, required=True, help='[0-6] for comment, [0-4] for csc.')

parser.add_argument('--file_type', type=str, help='csv or xlsx, default: read from filename')
parser.add_argument('--save_type', type=str, default='xlsx', help='csv or xlsx, default: xlsx')
parser.add_argument('--thesaurus_path', type=str, default='data/thesaurus_dict.txt', help='thesaurus dictionary path')
parser.add_argument('--aug_threshold', type=int, default=1000, help='Ignore sents generating more than this threshold')
parser.add_argument('--sub_threshold', type=int, default=10, help='Less sentences -[10,20]- More sentences')
args = parser.parse_args()

"""
Read Data
"""
# Required parameter
file_path = args.file_path
data_type = args.data_type
assert data_type == 'csc' or data_type == 'comment', "Data type need to be 'csc' or 'comment'"
aug_label = args.aug_label

thesaurus_path = args.thesaurus_path
aug_threshold = args.aug_threshold
sub_threshold = args.sub_threshold
save_type = args.save_type

if args.file_type is None:
    file_type = file_path.split('.')[-1]
else:
    file_type = args.file_type
assert file_type == 'xlsx' or file_type == 'csv', "File type need to be 'csv' or 'xlsx'"

csc_label_dict = {0: '愤怒', 1: '不满', 2: '中性', 3: '满意', 4: '表扬'}
comment_label_dict = {0: 'neg3', 1: 'neg2', 2: 'neg1', 3: 'neutral', 4: 'pos1', 5:'pos2', 6:'pos3'}
data_type_dict = {'comment': '评论', 'csc': '对话'}

print("\n========= 读取数据 =========")

if data_type == 'comment':
    label_dict = comment_label_dict
else:
    label_dict = csc_label_dict

if file_type == 'csv':
    data = pd.read_csv(file_path, header=None)
else:
    data = pd.read_excel(file_path, header=None)


data.columns = ['idx', 'content', 'label']

to_be_aug = data[data['label'] == aug_label]

print("读取<{}>数据共<{}>条, 开始增强<{}>数据:<{}>条".format(data_type_dict[data_type], data.shape[0], label_dict[aug_label], to_be_aug.shape[0]))

to_be_aug = to_be_aug.drop(['idx'], axis=1)
to_be_aug = to_be_aug.drop_duplicates()
to_be_aug.columns = ['content', 'label']

to_be_aug = to_be_aug.dropna()


to_be_aug['content'] = to_be_aug.content.map(lambda x: remove_html(x))
to_be_aug['content'] = to_be_aug.content.map(lambda x: clean(x))
# to_be_aug_idx = to_be_aug.reset_index()
print("去重清洗后有<{}>条".format(to_be_aug.shape[0]))

"""
Thesaurus Dictionary
"""

print("\n========= 读取同义词词典 =========")

with open(thesaurus_path, 'r', encoding='utf-8') as f:
    thesaurus_all = f.read().splitlines()

word_dict = {}
for i in range(len(thesaurus_all)):
    words_list = thesaurus_all[i]
    for word in words_list.split(','):
        word_dict[word] = i

print('共有{}个同义词组，包含{}个词'.format(len(thesaurus_all), len(word_dict)))


print("\n========= 数据增强 =========")

total_aug = pd.DataFrame()
aug_amount = []
redundant_sent = []
sent_list = to_be_aug['content'].tolist()
count = 0
for sent in sent_list:
    count += 1
#     print(sent)
    aug_sentences = aug_sent(sent, word_dict, thesaurus_all)
    if(count%1000==0):
        print("{}% Done".format(round(float(count) / len(sent_list), 2) * 100))
    if len(aug_sentences) == 0:
        continue
    if len(aug_sentences) > aug_threshold:
        redundant_sent.append(sent)
        # print(sent)
        continue
    aug_sentences_df = pd.DataFrame(aug_sentences).drop_duplicates()
    total_aug = total_aug.append(aug_sentences_df)
    aug_amount.append(aug_sentences_df.shape[0])

print('\n增强总数:', total_aug.shape[0])

label_col = pd.DataFrame([aug_label]*total_aug.shape[0])

if len(redundant_sent) != 0:
    redundant_sent = pd.DataFrame(redundant_sent)
    redundant_sent = redundant_sent.reset_index()
    redundant_sent = redundant_sent.drop(['index'], axis=1)
    redundant = pd.concat([redundant_sent, label_col], axis=1)
    redundant.columns = ['content', 'label']

    redundant_name = 'result/redundant' + data_type + "_" + label_dict[aug_label] + "_" + str(redundant.shape[0])
    redundant.to_csv(redundant_name + '.csv', header=None)


total_aug = total_aug.reset_index()
total_aug = total_aug.drop(['index'], axis=1)
res = pd.concat([total_aug,label_col], axis=1)
# res = res.drop(['level_0'], axis=1)
res.columns = ['content', 'label']

filename = "result/substitution_" + data_type + "_" + label_dict[aug_label] + "_" + str(res.shape[0])

if save_type == 'csv':
    filename += '.csv'
    res.to_csv(filename, header=None)
else:
    filename += '.xlsx'
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    res.to_excel(writer, sheet_name='Sheet1')
    writer.save()

print("\n结果写入{}".format(filename))
