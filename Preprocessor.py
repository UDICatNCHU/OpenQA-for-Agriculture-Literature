#!pip install pyserini
#!pip install faiss-cpu


def DocumentUnitProcessor(inputfile, sentence_level=True):
    pyserini_format_json_list = []
    if sentence_level==True:
        id_counter=0
        for id in inputfile:
            if input[id]["全文"]!= None:
                for sentence in input[id]["全文"].split("。"):
                    if len(sentence) < 512:                        #(移掉超過512長度的句子)
                        id_counter = id_counter + 1  
                        pyserini_format_json_list.append({"id":id_counter, "contents":sentence})
    else:
        for id in input:
            if input[id]["全文"]!= None:
                pyserini_format_json_list.append({"id":id, "contents":input[id]["全文"]})# 使用內文為單位進行index construction

    return pyserini_format_json_list
