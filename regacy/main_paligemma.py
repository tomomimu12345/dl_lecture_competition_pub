import re
import os
import random
import time
from statistics import mode
from tqdm import tqdm

from PIL import Image
import numpy as np
import pandas as pd
import pandas
from collections import Counter
import torch
import torch.nn as nn
import torchvision
import torchtext
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def process_text(text):
    # lowercase
    text = text.lower()
    # 数詞を数字に変換
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)
    # 小数点のピリオドを削除
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)
    # 冠詞の削除
    text = re.sub(r'\b(a|an|the)\b', '', text)
    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)
    text = re.sub(r"[^\w\s':]", ' ', text) # 句読点をスペースに変換
    text = re.sub(r'\s+,', ',', text) # 句読点をスペースに変換
    text = re.sub(r'\s+', ' ', text).strip()# 連続するスペースを1つに変換
    return text

# 1. データローダーの作成
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, answer=True):
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pandas.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
        self.answer = answer

        # answerの辞書を作成
        self.answer2idx = {}
        self.idx2answer = {}

        if self.answer:
            # コーパス読み込み(added)
            df_corpus = pd.read_csv("class_mapping.csv",sep=",")
            df_corpus["answer"] = df_corpus["answer"].apply(process_text)
            self.answer2idx = dict(zip(df_corpus["answer"], df_corpus["class_id"]))
            # 回答に含まれる単語を辞書に追加
            for answers in self.df["answers"]:
                for answer in answers:
                    word = answer["answer"]
                    word = process_text(word)
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # 逆変換用の辞書(answer)

    def update_dict(self, dataset):
        """
        検証用データ，テストデータの辞書を訓練データの辞書に更新する．

        Parameters
        ----------
        dataset : Dataset
            訓練データのDataset
        """
        self.answer2idx = dataset.answer2idx
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        """
        対応するidxのデータ（画像，質問，回答）を取得．

        Parameters
        ----------
        idx : int
            取得するデータのインデックス

        Returns
        -------
        image : torch.Tensor  (C, H, W)
            画像データ
        question_idx : torch.Tensor  (vocab_size)
            質問文をtokenizeしたもの
        answers : torch.Tensor  (n_answer)
            10人の回答者の回答のid
        mode_answer_idx : torch.Tensor  (1)
            10人の回答者の回答の中で最頻値の回答のid
        """
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        question = process_text(self.df["question"][idx])

        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            mode_answer_idx = mode(answers)  # 最頻値を取得（正解ラベル）
            return image, question, torch.Tensor(answers).to(torch.bfloat16), int(mode_answer_idx)

        else:
            return image, question

    def __len__(self):
        return len(self.df)


# 2. 評価指標の実装
# 簡単にするならBCEを利用する
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10

    return total_acc / len(batch_pred)


# 3. モデルのの実装
class VQAModel(nn.Module):
    def __init__(self, num_classes, hidden_size, model_name="google/paligemma-3b-pt-224",use_cls_token=True):
        super(VQAModel, self).__init__()
        dtype = torch.bfloat16
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(model_name,torch_dtype=dtype, revision="bfloat16")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.use_cls_token = use_cls_token

        for param in self.model.parameters():
            param.requires_grad = False
        # First linear layer
        self.linear_layer1 = nn.Sequential(
            nn.LayerNorm(2048,dtype=dtype),
            nn.Dropout(p=0.5),
            nn.Linear(2048, hidden_size, dtype=dtype)
        )
        # Second linear layer
        self.linear_layer2 = nn.Sequential(
            nn.LayerNorm(hidden_size,dtype=dtype),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, num_classes,dtype=dtype) 
        )
        self.answer_type_layer = nn.Linear(hidden_size, 4,dtype=dtype)
        self.answer_mask_layer = nn.Linear(4, num_classes,dtype=dtype)
        self.sigmoid = nn.Sigmoid()
    def forward(self, image, question):
        inputs = self.processor(text=question, images=image, return_tensors="pt", padding=True).to("cuda" if torch.cuda.is_available() else "cpu",torch.bfloat16)
        outputs = self.model(**inputs, output_hidden_states=True)

        last_hidden_state = outputs.hidden_states[-1]
        if self.use_cls_token:
            features = last_hidden_state[:, 0, :]  # Use [CLS] token
        else:
            features = last_hidden_state[:, -1, :]  # Use last token
        features = self.linear_layer1(features)
        answer_type = self.answer_type_layer(features)
        answer_mask = self.answer_mask_layer(answer_type)
        answer_mask = self.sigmoid(answer_mask)
        output = self.linear_layer2(features)

        output = output * answer_mask
        return output

# 4. 学習の実装
def train(model, dataloader, optimizer, criterion, device,epoch):
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for i, (image, question, answers, mode_answer) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch + 1}'):
        answer, mode_answer = answers.to(device), mode_answer.to(device)
        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy
    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def eval(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for i, (image, question, answers, mode_answer) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Valid'):
        answer, mode_answer = answers.to(device), mode_answer.to(device)
        with torch.no_grad():
            pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze())
        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy
    
    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def plot_metrics(metrics):
    epochs = range(len(metrics['train_loss']))
    plt.figure(figsize=(12, 12))
    # Plot loss
    plt.subplot(3, 1, 1)
    plt.plot(epochs, metrics['train_loss'], label='Train Loss')
    plt.plot(epochs, metrics['valid_loss'], label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train and Valid Loss')
    # Plot accuracy
    plt.subplot(3, 1, 2)
    plt.plot(epochs, metrics['train_acc'], label='Train Accuracy')
    plt.plot(epochs, metrics['valid_acc'], label='Valid Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Train and Valid Accuracy')
    # Plot simple accuracy
    plt.subplot(3, 1, 3)
    plt.plot(epochs, metrics['train_simple_acc'], label='Train Simple Accuracy')
    plt.plot(epochs, metrics['valid_simple_acc'], label='Valid Simple Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Simple Accuracy')
    plt.legend()
    plt.title('Train and Valid Simple Accuracy')
    
    plt.tight_layout()
    plt.savefig("learning.png", dpi=1000)
    plt.close()
    plt.clf()
    plt.cla()
def main():
    # !huggingface-cli login
    # deviceの設定
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"


    dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train")
    train_size = int(0.95 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", answer=False)
    test_dataset.update_dict(dataset)

    train_dataset.answer2idx = dataset.answer2idx
    train_dataset.idx2answer = dataset.idx2answer
    valid_dataset.answer2idx = dataset.answer2idx
    valid_dataset.idx2answer = dataset.idx2answer


    #######
    def collate_fn(batch):
        images, text_list, answers_list, mode_answers_list = [], [], [], []
        for item in batch:
            if len(item) == 4:  # `answer`がTrueの場合
                image, question, answers, mode_answer = item
                images.append(image)
                text_list.append(question)
                answers_list.append(answers.clone().detach())
                mode_answers_list.append(torch.tensor(mode_answer))
            else: 
                image, question = item
                images.append(image)
                text_list.append(question)

        if len(answers_list) > 0:
            return images, text_list, torch.stack(answers_list), torch.stack(mode_answers_list)
        else:
            return images, text_list
    #######
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn,drop_last=True,num_workers=4, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn,drop_last=True,num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn,num_workers=4, pin_memory=True)
    print(f"train:{len(train_loader)}, valid:{len(valid_loader)}, test:{len(test_loader)}")
    
    model = VQAModel(num_classes=len(train_dataset.answer2idx), hidden_size=512).to(device)

    # optimizer / criterion
    num_epoch = 50
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-7)

    os.makedirs('models', exist_ok=True)
    os.makedirs('submissions', exist_ok=True)
    
    metrics = {
        'train_loss': [],
        'valid_loss': [],
        'train_acc': [],
        'valid_acc': [],
        'train_simple_acc': [],
        'valid_simple_acc': []
    }

    # train model
    for epoch in range(num_epoch):
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device,epoch)
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['train_simple_acc'].append(train_simple_acc)
        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n"
              f"train acc: {train_acc:.4f}\n"
              f"train simple acc: {train_simple_acc:.4f}")
        valid_loss, valid_acc, valid_simple_acc, valid_time = eval(model, valid_loader, criterion, device)
        metrics['valid_loss'].append(valid_loss)
        metrics['valid_acc'].append(valid_acc)
        metrics['valid_simple_acc'].append(valid_simple_acc)
        print(f"valid time: {valid_time:.2f} [s]\n"
              f"valid loss: {valid_loss:.4f}\n"
              f"valid acc: {valid_acc:.4f}\n"
              f"valid simple acc: {valid_simple_acc:.4f}")
        plot_metrics(metrics)
        if (epoch + 1) % 4 == 1:
            # 提出用ファイルの作成
            model.eval()
            submission = []
            for image, question in test_loader:
                pred = model(image, question)
                pred = pred.argmax(1).cpu().item()
                submission.append(pred)

            submission = [train_dataset.idx2answer[id] for id in submission]
            submission = np.array(submission)
            torch.save(model.state_dict(), f"models/model_PaliGemma_epoch_{epoch + 1}.pth")
            np.save(f"submissions/submission_PaliGemma_epoch_{epoch + 1}.npy", submission)
            print(f"successfully saved model and submission\n")
            model.train() 


if __name__ == "__main__":
    main()
