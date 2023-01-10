from transformers import GPT2Model
import jittor as jt
import pickle
import argparse


parser = argparse.ArgumentParser(description='from model directory to jittor readable parameters')
parser.add_argument('--name', type=str, default='gpt2-large', help='model_name, default: gpt2-large')
parser.add_argument('--cache_dir', type=str, default="./gpt2-model", help='cache dir')# 这里要链接到正确的gpt2model地址
args = parser.parse_args()

def state_dict_convert(indict:dict)->dict:
    res_dict = {}
    for key in indict.keys():
        val = indict[key]
        if(isinstance(val,jt.Var)):
            res_dict["transformer."+key]=val
        else:
            res_dict["transformer."+key]=jt.from_torch(val)
    return res_dict


name = args.name
model = GPT2Model.from_pretrained(name, cache_dir=args.cache_dir+f"/{name}-s3")
modified_dict = state_dict_convert(model.state_dict())
print("Para List")
for each in modified_dict:
    print(each,modified_dict[each].shape)
jt.save(modified_dict,"./gpt-state-dict/{}-paras.jt".format(name))