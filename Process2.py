import json

json_path = '/data/yujie/Ebert/pre_dataset.json'
# 密码子-氨基酸对照表
codon_to_amino_acid = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'TAT':'Y','TAC':'Y','TAA':'W','TAG':'W','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
    'TGT':'C','TGC':'C','TGA':'W','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G'
}

complement_dict = {'a': 't', 't': 'a', 'c': 'g', 'g': 'c'}

# 翻译基因序列到氨基酸序列
def translate_gene_to_amino(gene_seq):
    amino_seq = ''
    for i in range(0, len(gene_seq), 3):
        codon = gene_seq[i:i+3].upper()
        amino_acid = codon_to_amino_acid.get(codon, '')
        if amino_acid:  # 忽略终止密码子
            amino_seq += amino_acid
    return amino_seq

# 将翻译后不符合标准的基因序列A-T,G-C对调
def complement_gene(gene):
    complement_sequence = ''.join(complement_dict[base] for base in gene)
    return complement_sequence

# 判断两个字符串不同的个数
def diff(str1, str2):
    # 判断字符串长度是否相等
    if len(str1) != len(str2):
        return 1000

    # 统计不同字符的个数
    diff_count = sum(1 for char1, char2 in zip(str1, str2) if char1 != char2)

    return diff_count

# 检查并修正序列
def check_and_correct_sequence(pdb_data, pdb_id, wrong_data):
    gene_seq = pdb_data['gene seq']
    translated_amino_seq = translate_gene_to_amino(gene_seq)
    original_amino_seq = pdb_data['amino seq']
    diff_num = diff(translated_amino_seq, original_amino_seq)

    if diff_num <= 5:
        return 0,0# 序列匹配，不需要修正
    else:
        # 尝试反译基因序列并重新翻译
        complement_gene_seq = complement_gene(gene_seq)
        complement_translated_amino_seq = translate_gene_to_amino(complement_gene_seq)
        diff_num = diff(complement_translated_amino_seq,original_amino_seq)
        if diff_num <= 10:
            pdb_data['gene seq'] = complement_gene_seq # 更新为翻转后的基因序列
            pdb_data['amino seq'] = complement_translated_amino_seq

            return complement_gene_seq, complement_translated_amino_seq
        else:
            wrong_data.append(pdb_id)  # 序列仍不匹配，记录PDB ID
            return 1,1

# 加载数据
with open(json_path, 'r') as file:
    data = json.load(file)

# 记录错误数据的PDB ID
wrong_data = []
# 检查并更新数据集
for pdb_id, pdb_data in data.items():
    gene_seq, amino_seq = check_and_correct_sequence(pdb_data, pdb_id, wrong_data)

for id in wrong_data:
    data.pop(id)

# 写入更新后的数据集
with open('pre_dataset_corrected.json', 'w') as file:
    json.dump(data, file, indent=3)

# 将错误数据的PDB ID写入文件
with open('wrong_data.txt', 'w') as file:
    for item in wrong_data:
        file.write("%s\n" % item)
