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

from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
from lora_plus import create_loraplus_optimizer

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
        image = f"{self.image_dir}/{self.df['image'][idx]}"
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

def collate_fn(batch, vl_chat_processor):
    inputs_list=[]
    answers_list, mode_answers_list=[],[]
    for item in batch:
        if len(item) == 4:  # `answer`がTrueの場合
            image, question, answers, mode_answer = item
            answers_list.append(answers)
            mode_answers_list.append(mode_answer)
        else: 
            image, question = item
        conversation = [
            {
                "role": "User",
                "content": "<image_placeholder>"+question,
                "images": [image]
            },
            {
                "role": "Assistant",
                "content": ""
            }
        ]
        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations = conversation,
            images = pil_images,
            force_batchify=True
        )
        inputs_list.append(prepare_inputs)
    if len(answers_list) > 0:
        return inputs_list, answers_list, mode_answers_list
    else:
        return inputs_list
#4. Train
def train(model, tokenizer, dataloader,valid_loader, optimizer, schedular, device, epoch):
    model.train()

    total_loss = 0

    start = time.time()
    with tqdm(dataloader, total=len(dataloader), desc=f'Epoch {epoch + 1}') as pbar:
        for i, (inputs, answers, mode_answer) in enumerate(pbar):
            for id in range(len(inputs)):
                prepare_input = inputs[id].to(device=device,dtype=torch.float16)
                inputs_embeds = model.prepare_inputs_embeds(**prepare_input)
                # input_ids = prepare_input.input_ids
                attention_mask =prepare_input.attention_mask
                target_ids = tokenizer(mode_answer[id], return_tensors='pt',max_length=inputs_embeds.shape[1], padding='max_length').input_ids.to(device)
                target_ids[target_ids == tokenizer.pad_token_id] = -100
                outputs = model.language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask = attention_mask,
                    labels = target_ids
                )
                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                schedular.step()
                total_loss += loss.item()
            # if (i+1)%1000==0:
            #     valid_acc, valid_time = eval(model, tokenizer, valid_loader, device, epoch)
            #     print(f"【{epoch + 1}】\n"
            #         f"valid time: {valid_time:.2f} [s]\n"
            #         f"valid acc: {valid_acc:.4f}\n")
            #     model.train()
            pbar.set_postfix(OrderedDict(Loss=total_loss / (i + 1)))

    return total_loss / len(dataloader), time.time() - start
#5. Valid
def eval(model, tokenizer, dataloader, device, epoch):
    model.eval()
    
    total_acc = 0

    start = time.time()
    with tqdm(dataloader, total=len(dataloader), desc=f'Valid: {epoch + 1}') as pbar:
        for i, (inputs, answers, mode_answer) in enumerate(pbar):
            pred =[]
            for id in range(len(inputs)):
                prepare_input = inputs[id].to(device=device,dtype=torch.float16)
                inputs_embeds = model.prepare_inputs_embeds(**prepare_input)
                outputs = model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_input.attention_mask,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=10,
                    do_sample=False,
                    use_cache=False
                )
                answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
                pred.append(answer)
            acc =  VQA_criterion(pred, answers)
            total_acc += acc  # VQA accuracy
            pbar.set_postfix(OrderedDict(Acc = total_acc / (i + 1),pred = [", ".join(pred)],ans=[", ".join(mode_answer)]))

    return total_acc / len(dataloader), time.time() - start

def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train")
    train_size = int(0.95 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", answer=False)

    model_path = "deepseek-ai/deepseek-vl-7b-base" #or "c:/local/path/to/deepseek-vl-7b-chat"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    pad_token = "<PAD>"
    if pad_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'pad_token': pad_token})
    pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
    tokenizer.pad_token_id = pad_token_id
    print("before:"+vl_chat_processor.system_prompt)
    vl_chat_processor.system_prompt = (
        "You are a visual question answering assistant. "
        "Provide short, direct answers. Few words, no full sentences, no articles, very concise. "
        "If the question is unanswerable, respond with 'unanswerable'. "
        "For yes/no questions, respond with 'yes' or 'no'."
    )# 0.2135
    print("after:"+vl_chat_processor.system_prompt)


    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, low_cpu_mem_usage=True, quantization_config=bnb_config, device_map="cuda")

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=256, 
	    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x:collate_fn(x,vl_chat_processor), drop_last=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=True, collate_fn=lambda x:collate_fn(x,vl_chat_processor), drop_last=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x:collate_fn(x,vl_chat_processor), num_workers=4)
 
    print(f"train:{len(train_loader)}, valid:{len(valid_loader)}, test:{len(test_loader)}")

    # optimizer / criterion/schedular
    num_epoch = 6
    optimizer_kwargs = {'lr': 2e-5, 'betas': (0.9, 0.999), 'weight_decay': 1e-6}
    loraplus_ratio = 16.0
    optimizer = create_loraplus_optimizer(model, AdamW, optimizer_kwargs, loraplus_ratio)
    scheduler = get_scheduler(
        name = "linear", 
        optimizer = optimizer, 
        num_warmup_steps = 20,
        num_training_steps = num_epoch*len(train_loader)
    )

    os.makedirs('models', exist_ok=True)
    os.makedirs('submissions', exist_ok=True)

    valid_acc, valid_time = eval(model, tokenizer, valid_loader, device, -1)
    print(f"【{0}/{num_epoch}】\n"
            f"valid time: {valid_time:.2f} [s]\n"
            f"valid acc: {valid_acc:.4f}\n")

    best_acc = 0
    best_submission=[]
    # train model
    for epoch in range(num_epoch):
        train_loss, train_time = train(model, tokenizer, train_loader,valid_loader, optimizer, scheduler, device, epoch)
        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n")
        valid_acc, valid_time = eval(model, tokenizer, valid_loader, device, epoch)
        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"valid time: {valid_time:.2f} [s]\n"
              f"valid acc: {valid_acc:.4f}\n")
        if best_acc < valid_acc:
            best_acc = valid_acc
            # 提出用ファイルの作成
            model.eval()
            submission = []
            with tqdm(test_loader, total=len(test_loader), desc=f'Test {epoch + 1}') as pbar:
                for inputs in pbar:
                    prepare_input = inputs[0].to(device=device,dtype=torch.float16)
                    inputs_embeds = model.prepare_inputs_embeds(**prepare_input)
                    outputs = model.language_model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=prepare_input.attention_mask,
                        pad_token_id=tokenizer.pad_token_id,
                        bos_token_id=tokenizer.bos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=10,
                        do_sample=False,
                        use_cache=False
                    )
                    pred = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
                    pred = process_text(pred)
                    pbar.set_postfix(OrderedDict(prediction = pred))
                    submission.append(pred)

            submission = np.array(submission)
            # torch.save(model.state_dict(), f"models/model_PaliGemmaFT_epoch_{epoch + 1}.pth")
            np.save(f"submissions/submission_DeepSeek_epoch_{epoch + 1}.npy", submission)
            print(f"successfully saved model and submission\n")
            best_submission = submission
    np.save(f"submissions/submission_DeepSeek_best.npy", best_submission)

if __name__ == "__main__":
    main()
