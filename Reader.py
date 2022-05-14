# !pip install -q transformers

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

def simpleReader(alltext, allquery, topk):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # tokenizer = AutoTokenizer.from_pretrained("HankyStyle/Multi-ling-BERT")
    # model = AutoModelForQuestionAnswering.from_pretrained("HankyStyle/Multi-ling-BERT")
    tokenizer = AutoTokenizer.from_pretrained("nyust-eb210/braslab-bert-drcd-384")
    model = AutoModelForQuestionAnswering.from_pretrained("nyust-eb210/braslab-bert-drcd-384").to(device)
    encoded_input = tokenizer(alltext, allquery, return_tensors="pt", padding=True, truncation=True, max_length=512, stride=256).to(device)
    qa_outputs = model(**encoded_input)

    starts = torch.argmax(qa_outputs.start_logits, dim=-1)
    starts_pos_topk = torch.topk(qa_outputs.start_logits, dim=-1, k=topk)
    starts_pos_topk_indice = starts_pos_topk.indices.tolist()

    ends = torch.argmax(qa_outputs.end_logits, dim=-1)
    ends_pos_topk = torch.topk(qa_outputs.end_logits, dim=-1, k=topk)
    ends_pos_topk_indice = ends_pos_topk.indices.tolist()

    # verbose mode
    # print("verbose mode") # top-k
    number_of_questiosn = len(starts_pos_topk_indice)
    startends = []
    for i in range(number_of_questiosn):
      startends.append(list(zip(starts_pos_topk_indice[i], ends_pos_topk_indice[i])))
    
    # print(startends)
    
    answers = []
    for i, ses in enumerate(startends):
      for se in ses:
        ans = encoded_input.input_ids.tolist()[i][se[0] : se[1] + 1]
        ans = "".join(tokenizer.decode(ans).split())
        # print(answer)
        answers.append(ans)

    return answers