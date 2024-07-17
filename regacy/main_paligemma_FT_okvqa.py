import re
import os
import random
import time
from tqdm import tqdm

from PIL import Image
import numpy as np
import pandas
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import transforms

from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig, AdamW, get_scheduler
from peft import get_peft_model, LoraConfig, TaskType
import itertools
from collections import OrderedDict
import re

# https://github.com/huggingface/notebooks/blob/main/examples/paligemma/Fine_tune_PaliGemma.ipynb

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
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text) # 小数点のピリオドを削除
    text = re.sub(r'\b(a|an|the)\b', '', text) # 冠詞の削除
    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)
    text = re.sub(r"[^\w\s':]", ' ', text) # 句読点をスペースに変換
    text = re.sub(r'\s+,', ',', text) # 句読点をスペースに変換
    text = re.sub(r'\s+', ' ', text).strip() # 連続するスペースを1つに変換
    return text

# 1. データローダーの作成
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, answer=True):
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pandas.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
        self.answer = answer

    def __getitem__(self, idx):
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}").convert("RGB")
        # image = image.resize((224, 224), Image.BICUBIC)
        # transform = transforms.ToTensor()
        # image = transform(image)
        question = process_text(self.df["question"][idx])

        if self.answer:
            answers = [process_text(answer["answer"]) for answer in self.df["answers"][idx]]
            mode_answer = Counter(answers).most_common(1)[0][0] 
            return image, question, answers, mode_answer
        else:
            return image, question     
    def __len__(self):
        return len(self.df)

# 2. 評価指標の実装
def VQA_criterion(batch_pred, batch_answers):
    total_acc = 0
    for pred, answers in zip(batch_pred, batch_answers):
        pred = process_text(pred)
        match_counts = [sum([pred == answers[idx] for idx in comb]) for comb in itertools.combinations(range(10), 9)]
        acc = min(1.0, np.mean(match_counts) / 3.0)
        total_acc += acc
    return total_acc / len(batch_pred)

def collate_fn(batch):
    images, text_list, answers_list, mode_answers_list = [], [], [], []
    for item in batch:
        if len(item) == 4:  # `answer`がTrueの場合
            image, question, answers, mode_answer = item
            images.append(image)
            text_list.append("answer en "+question)
            answers_list.append(answers)
            mode_answers_list.append(mode_answer)
        else: 
            image, question = item
            images.append(image)
            text_list.append("answer en "+question)

    if len(answers_list) > 0:
        return images, text_list, answers_list, mode_answers_list
    else:
        return images, text_list
#4. Train
def train(model, processor, dataloader, optimizer, schedular, device, epoch):
    model.train()

    total_loss = 0

    start = time.time()
    with tqdm(dataloader, total=len(dataloader), desc=f'Epoch {epoch + 1}') as pbar:
        for i, (images, questions, answers, mode_answer) in enumerate(pbar):
            inputs = processor(text = questions, images = images, suffix = mode_answer,
                        return_tensors="pt", padding="longest", tokenize_newline_separately=False).to(torch.bfloat16).to(device)
            outputs = model(**inputs)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedular.step()
            total_loss += loss.item()

            pbar.set_postfix(OrderedDict(Loss=total_loss / (i + 1)))

    return total_loss / len(dataloader), time.time() - start
#5. Valid
def eval(model, processor, dataloader, device, epoch):
    model.eval()
    
    total_acc = 0

    start = time.time()
    with tqdm(dataloader, total=len(dataloader), desc=f'Valid: {epoch + 1}') as pbar:
        for i, (images, questions, answers, mode_answer) in enumerate(pbar):
            inputs = processor(text = questions, images = images,
                        return_tensors="pt", padding="longest",tokenize_newline_separately=False).to(torch.bfloat16).to(device)
            generate_ids = model.generate(**inputs, max_new_tokens=30)
            pred = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            pred = list(map(lambda x, y: x[len(y):].replace("\n", "").strip() if x.startswith(y) else x, pred,questions))
            acc =  VQA_criterion(pred, answers)
            total_acc += acc  # VQA accuracy
            pbar.set_postfix(OrderedDict(Acc = total_acc / (i + 1),pred = [", ".join(pred)],ans=[", ".join(mode_answer)]))

    return total_acc / len(dataloader), time.time() - start

def main():
    # !huggingface-cli login
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train")

    # tot_answer=[]
    # for idx in range(len(dataset.df["answers"])):
    #     count+=1
    #     answers = [process_text(answer["answer"]) for answer in dataset.df["answers"][idx]]
    #     tot_answer.extend(answers)
    # mode_answer = Counter(tot_answer).most_common(50)
    # print(mode_answer) 

    # [('unanswerable', 55613), ('no', 5225), ('yes', 4337), ('white', 2511), ('grey', 2097), ('black', 2032), ('blue', 1716), ('red', 1087), ('brown', 787), ('pink', 748), ('green', 703), ('keyboard', 672), ('purple', 566), ('nothing', 516), ('soup', 507), ('dog', 479), ('laptop', 476), ('yellow', 425), ('ph1', 392), ('food', 352), ('tan', 352), ('lotion', 339), ('orange', 330), ('cell ph1', 281), ('chicken', 279), ('pepsi', 279), ('corn', 279), ('coffee', 277), ('0', 274), ('remote', 267), ('coca cola', 262), ('shampoo', 253), ('beans', 245), ('wine', 245), ('remote control', 241), ('computer', 234), ('can', 227), ('soda', 227), ('bottle', 225), ('green beans', 225), ('tv', 224), ('chair', 211), ('beer', 208), ('table', 207), ('book', 205), ('beige', 202), ('computer screen', 198), ('pen', 194), ('black white', 184), ('cup', 184)]


    train_size = int(0.95 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", answer=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, drop_last=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, drop_last=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=4)
 
    print(f"train:{len(train_loader)}, valid:{len(valid_loader)}, test:{len(test_loader)}")

    # model_name="google/paligemma-3b-pt-224" # init_acc = 0.051(0.1896)
    # model_name="google/paligemma-3b-mix-224" # init_acc = 0.4506(0.8441) 
    # model_name="google/paligemma-3b-ft-vizwizvqa-224"  # init_acc = 0.4625(0.8501)
    model_name="google/paligemma-3b-ft-okvqa-224" # init_acc = 0.0921(0.3473), 0.5912, 0.7121, 0.7296

    match = re.search(r'ft-(.*?)-224', model_name)
    if match:
        extracted_string = match.group(1)
        print(f"Extracted string: {extracted_string}")
    else:
        print("The string between 'ft-' and '-224' was not found.")
        exit(1)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_type=torch.bfloat16
    )
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_name, quantization_config=bnb_config, device_map={"":0})

    processor = AutoProcessor.from_pretrained(model_name)

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8, 
	    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # optimizer / criterion/schedular
    num_epoch = 6
    optimizer = AdamW(model.parameters(), lr=2e-5,betas=(0.9, 0.999), weight_decay=1e-6)
    scheduler = get_scheduler(
        name = "linear", 
        optimizer = optimizer, 
        num_warmup_steps = 20,
        num_training_steps = num_epoch*len(train_loader)
    )

    os.makedirs('models', exist_ok=True)
    os.makedirs('submissions', exist_ok=True)

    valid_acc, valid_time = eval(model, processor, valid_loader, device, -1)
    print(f"【{0}/{num_epoch}】\n"
            f"valid time: {valid_time:.2f} [s]\n"
            f"valid acc: {valid_acc:.4f}\n")

    with open('Zeroshot.txt', 'a') as f:
        f.write(f"{extracted_string},{valid_acc}\n")

    best_acc = 0
    best_submission=[]
    # train model
    for epoch in range(num_epoch):
        train_loss, train_time = train(model, processor, train_loader, optimizer, scheduler, device, epoch)
        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n")
        valid_acc, valid_time = eval(model, processor, valid_loader, device, epoch)
        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"valid time: {valid_time:.2f} [s]\n"
              f"valid acc: {valid_acc:.4f}\n")
        if best_acc < valid_acc:
            best_acc = valid_acc
            # 提出用ファイルの作成
            model.eval()
            submission = []
            with tqdm(test_loader, total=len(test_loader), desc=f'Test {epoch + 1}') as pbar:
                for images, questions in pbar:
                    inputs = processor(text=questions, images=images, return_tensors="pt", padding="longest").to(device, torch.bfloat16)
                    generate_ids = model.generate(**inputs, max_new_tokens=30)
                    pred = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0][len(questions[0]):].replace("\n","")
                    pred = process_text(pred)
                    pbar.set_postfix(OrderedDict(prediction = pred))
                    submission.append(pred)

            submission = np.array(submission)
            # torch.save(model.state_dict(), f"models/model_PaliGemmaFT_epoch_{epoch + 1}.pth")
            np.save(f"submissions/submission_PaliGemmaFT_{extracted_string}_epoch_{epoch + 1}.npy", submission)
            print(f"successfully saved model and submission\n")
            best_submission = submission
        else:
            break
    np.save(f"submissions/submission_PaliGemmaFT_{extracted_string}_best.npy", best_submission)

if __name__ == "__main__":
    main()
