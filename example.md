# Vizwizデータセット
## 0. main_corpus.py
- コーパスの使用
- tokenize
- Resnet50 + LSTM

## 1. main_clip.py
- clipベースのモデル

https://github.com/yousefkotp/Visual-Question-Answering

Less Is More: Linear Layers on CLIP Features as Powerful VizWiz Model(2021)

## 2. main_openclip.py
- open clip clipのオープンソース

https://github.com/mlfoundations/open_clip?tab=readme-ov-file

https://github.com/mlfoundations/open_clip/blob/main/docs/openclip_results.csv

### 最上位モデル
https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378

## 3. main_paligemma-FT.py
Paligemmaをloraでファンチューニングする

メモリサイズ的に224pxが限界

https://note.com/npaka/n/n1afc8f64e447

- VQAv2```google/paligemma-3b-ft-vqav2-224```
0.3473, 0.5912, 0.7121, 0.7296
- OKVQA```google/paligemma-3b-ft-okvqa-224```
0.3053, 0.5519, 0.6142, 0.6044
- TextVQA```google/paligemma-3b-ft-textvqa-224```
0.3537, 0.5196, 0.5277, 0.5330, 0.5157 
- OCR-VQA```google/paligemma-3b-ft-ocrvqa-224```
0.2477, 0.2478, 0.4941, 

- GQA
- TallyQA
- DocVQA

### Vizwiz2024
https://eval.ai/web/challenges/challenge-page/2185/leaderboard/5394/overall
### チームSLCV
https://www.youtube.com/watch?v=Z3_QyH6zzJ8