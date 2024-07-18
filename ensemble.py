import numpy as np
from collections import Counter

# npyファイルを読み込む
file1 = np.load('submissions/submission_PaliGemmaFT_vqav2-448-loraplus_epoch_2_5000.npy')
file2 = np.load('submissions/submission_PaliGemmaFT_vqav2-448-loraplus_epoch_1_r8.npy')
file3 = np.load('submissions/submission_PaliGemmaFT_vqav2-448_epoch_1.npy')
file4 = np.load('submissions/submission_PaliGemmaFT_vqav2_epoch_3.npy')
file5 = np.load('submissions/submission_PaliGemmaFT_vqav2-448_epoch_3.npy')
file6 = np.load('submissions/submission_PaliGemmaFT_vqav2_epoch_2.npy')
file7 = np.load('submissions/submission_PaliGemmaFT_vqav2_epoch_1.npy')
file8 = np.load('submissions/submission_PaliGemmaFT_okvqa_epoch_2.npy')


# アンサンブルリストを作成するための空のリスト
ensemble_list = []

# 各インデックスで多数決を行う
for i in range(len(file1)):
    votes = [file1[i], file2[i], file3[i], file4[i], file5[i],file6[i],file7[i],file8[i]]
    majority_vote = Counter(votes).most_common(1)[0][0]
    ensemble_list.append(majority_vote)

# 最終的なリストをnpyファイルとして保存
np.save('submissions/ensemble.npy', ensemble_list)

print("Ensemble list has been saved to 'ensemble.npy'")
