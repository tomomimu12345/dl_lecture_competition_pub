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
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab
from torch.utils.tensorboard import SummaryWriter

# TensorBoardの初期化
writer = SummaryWriter(log_dir='./runs/vqa_1')

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
    # 句読点をスペースに変換
    text = re.sub(r"[^\w\s':]", ' ', text)
    # 句読点をスペースに変換
    text = re.sub(r'\s+,', ',', text)
    # 連続するスペースを1つに変換
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 1. データローダーの作成
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, tokenizer, vocab,transform=None, answer=True):
        self.transform = transform  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pandas.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
        self.answer = answer

        self.tokenizer=tokenizer
        self.vocab = vocab
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
        image = self.transform(image)
        question = process_text(self.df["question"][idx])
        # question_tokens = self.tokenizer(process_text(question))
        # question_idx = [self.vocab[token] for token in question_tokens]

        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            mode_answer_idx = mode(answers)  # 最頻値を取得（正解ラベル）

            return image, question, torch.Tensor(answers), int(mode_answer_idx)

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
    def __init__(self, vocab_size: int,embed_size:int, n_answer: int,hidden_size:int =512,weights=ResNet50_Weights.DEFAULT):
        super().__init__()
        # 事前学習済みのResNet50モデルをロード
        self.resnet = resnet50(weights=weights)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.resnet_fc = nn.Linear(2048, 512)

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.text_encoder = nn.LSTM(embed_size,hidden_size,num_layers=2, batch_first=True, bidirectional=True)
        self.fc0 = nn.Linear(2*hidden_size, 512)
        self.fc = nn.Sequential(
            nn.Linear(512 + 512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_answer)
        )

    def forward(self, image, question,len_seq):
        with torch.no_grad():  # 事前学習済みモデルのパラメータは更新しない
            image_feature = self.resnet(image)
        image_feature = image_feature.view(image_feature.size(0), -1)
        image_feature = self.resnet_fc(image_feature)

        question = self.embedding(question)
        packed_question = pack_padded_sequence(question, len_seq, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.text_encoder(packed_question)
        question_feature = torch.cat((hidden[-2], hidden[-1]), dim=1)
        question_feature = self.fc0(question_feature)

        x = torch.cat([image_feature, question_feature], dim=1)
        x = self.fc(x)
        return x


# 4. 学習の実装
def train(model, dataloader, optimizer, criterion, device,epoch):
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for i, (image, question, len_seq, answers, mode_answer) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch + 1}'):
        image, question,  answer, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)

        pred = model(image, question,len_seq)
        loss = criterion(pred, mode_answer.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy

        # TensorBoardに損失と精度を記録
        writer.add_scalar('Train/Loss', loss.item(), epoch * len(dataloader) + i)
        writer.add_scalar('Train/Accuracy', VQA_criterion(pred.argmax(1), answers), epoch * len(dataloader) + i)

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


def eval(model, dataloader, optimizer, criterion, device):
    model.eval()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, len_seq,answers, mode_answer in dataloader:
        image, question,  answer, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)
        pred = model(image, question,len_seq)
        loss = criterion(pred, mode_answer.squeeze())
        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


def main():
    # deviceの設定
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights=ResNet50_Weights.DEFAULT
    # dataloader / model
    transform = weights.transforms()

    #Lecture06より
    tokenizer = get_tokenizer("basic_english")
    df_train = pd.read_json("./data/train.json")
    df_train["question"]=df_train["question"].apply(process_text)
    counter = Counter()
    for question in df_train["question"]:
        counter.update(tokenizer(question))
    vocabulary = vocab(
        counter,
        min_freq=3,
        specials=('<unk>', '<PAD>', '<BOS>', '<EOS>')
    )
    # <unk>をデフォルトに設定することにより，min_freq回以上出てこない単語は<unk>になる
    vocabulary.set_default_index(vocabulary["<unk>"])
    print(f"単語種数: {len(vocabulary)}")
    print(*vocabulary.get_itos()[:100], sep=', ')
    #############################################3

    train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=transform,tokenizer=tokenizer, vocab=vocabulary)
    test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=transform, answer=False,tokenizer=tokenizer, vocab=vocabulary)
    test_dataset.update_dict(train_dataset)

    #######
    def text_transform(_text, max_length=256):
        # <BOS>と<EOS>の分 -2
        text = [vocabulary[token] for token in tokenizer(_text)][:max_length - 2]
        text = [vocabulary['<bos>']] + text + [vocabulary['<eos>']]
        return text, len(text)
    def collate_fn(batch):
        images, text_list, len_seq_list, answers_list, mode_answers_list = [], [], [], [], []
        for item in batch:
            if len(item) == 4:  # `answer`がTrueの場合
                image, question, answers, mode_answer = item
                processed_text, len_seq = text_transform(question)
                images.append(image)
                text_list.append(torch.tensor(processed_text))
                len_seq_list.append(len_seq)
                answers_list.append(answers.clone().detach())
                mode_answers_list.append(torch.tensor(mode_answer))
            else: 
                image, question = item
                processed_text, len_seq = text_transform(question)
                images.append(image)
                text_list.append(torch.tensor(processed_text))
                len_seq_list.append(len_seq)
        
        padded_questions = pad_sequence(text_list, batch_first=True, padding_value=vocabulary["<pad>"])
        lengths = torch.tensor(len_seq_list)

        if len(answers_list) > 0:
            return torch.stack(images), padded_questions, lengths, torch.stack(answers_list), torch.stack(mode_answers_list)
        else:
            return torch.stack(images), padded_questions, lengths
    #######

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True,collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,collate_fn=collate_fn)

    model = VQAModel(vocab_size=len(vocabulary), embed_size=200, n_answer=len(train_dataset.answer2idx),weights=weights).to(device)

    # optimizer / criterion
    num_epoch = 20
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

    os.makedirs('models', exist_ok=True)
    os.makedirs('submissions', exist_ok=True)

    # train model
    for epoch in range(num_epoch):
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device,epoch)
        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n"
              f"train acc: {train_acc:.4f}\n"
              f"train simple acc: {train_simple_acc:.4f}")
        if (epoch + 1) % 3 == 1:
            model.eval()
            submission = []
            for image, question, len_seq in test_loader:
                image, question = image.to(device), question.to(device)
                pred = model(image, question, len_seq)
                pred = pred.argmax(1).cpu().item()
                submission.append(pred)

            submission = [train_dataset.idx2answer[id] for id in submission]
            submission = np.array(submission)
            
            torch.save(model.state_dict(), f"models/model_corpus_epoch_{epoch + 1}.pth")
            np.save(f"submissions/submission_corpus_epoch_{epoch + 1}.npy", submission)
            print(f"successfully saved model and submission\n")
            model.train() 
    writer.close()

if __name__ == "__main__":
    main()
    #%tensorboard --logdir=./runs
