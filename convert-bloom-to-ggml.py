# Convert Hugging Face fine-tuned bloom-like models to ggml format
#
# Usage:
#
#   python models/convert-bloomz-to-ggml.py
#
# This script is similar to "convert-pt-to-ggml.py"


import os
import sys
import struct
import torch
import numpy as np

from transformers import AutoConfig
from transformers import BloomForCausalLM, BloomTokenizerFast

conv_map = {
    'word_embeddings': 'tok_embeddings',
    "word_embeddings_layernorm": 'norm',
    'input_layernorm': 'attention_norm',
    'self_attention.query_key_value': 'attention.query_key_value',
    'self_attention.dense':          'attention.wo',
    'post_attention_layernorm': 'ffn_norm',
    'mlp.dense_h_to_4h': 'feed_forward.w1',
    'mlp.dense_4h_to_h': 'feed_forward.w2',
    'ln_f': 'output_norm',
    'lm_head': 'output',
    }


if len(sys.argv) < 3:
    print("Usage: python convert-hf-to-ggml.py model_name dir-output [use-f32]")
    print("  model_name: name of the model to convert. Example: 'bigscience/bloomz-560m'")
    print("  dir-output: directory where the output file will be written")
    print("  use-f32:    if present, use float32 instead of float16")
    sys.exit(1)

model_name = sys.argv[1]
dir_out = sys.argv[2]

# make sure the output directory exists
os.makedirs(dir_out, exist_ok=True)

# possible data types
#   ftype == 0 -> float32
#   ftype == 1 -> float16
#
# map from ftype to string
ftype_str = ["f32", "f16"]
ftype = 1
if len(sys.argv) > 3:
    ftype = 0

tokenizer = BloomTokenizerFast.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
hparams = config.to_dict()
print("Loading model: ", model_name)


def get_all_chinese():
    """_summary_
        U+4E00~U+9FA5： 是最常用的范围，即名为：CJK Unified Ideographs 的区块
        U+9FA6~U+9FFF： 之间的字符还属于空码，暂时还未定义，但不能保证以后不会被定义。
    Returns:
        _type_: _description_
    """
    words = []
    for i in range(0x4e00, 0x9fff, 1):
        words.append(chr(i))
    return words


def is_chinese(string):
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False


# 取得vobcab词表，并把词表中的token转化为word
vocabs = tokenizer.get_vocab()
vocab_words = []
for key in vocabs:
    ids = vocabs[key]
    text = tokenizer.decode(ids)
    vocab_words.append(text)


# 取得所有在vocab词表里不存在的汉字
lack_words = []
words = get_all_chinese()
for word in words:
    if word not in vocab_words:
        tks = tokenizer.tokenize(word)
        if (len(tks) == 2):
            lack_words.append(" ".join(tks))

hparams["lack_words_size"] = len(lack_words)

# 输出ggml模型文件
# struct.pack，支持的fmt格式：
# FORMAT	C TYPE	PYTHON TYPE	STANDARD SIZE	NOTES
# x	pad byte	no value
# c	char	string of length 1	1
# b	signed char	integer	1	(3)
# B	unsigned char	integer	1	(3)
# ?	_Bool	bool	1	(1)
# h	short	integer	2	(3)
# H	unsigned short	integer	2	(3)
# i	int	integer	4	(3)
# I	unsigned int	integer	4	(3)
# l	long	integer	4	(3)
# L	unsigned long	integer	4	(3)
# q	long long	integer	8	(2), (3)
# Q	unsigned long long	integer	8	(2), (3)
# f	float	float	4	(4)
# d	double	float	8	(4)
# s	char[]	string
# p	char[]	string
# P	void *	integer	 	(5), (3)
fname_out = dir_out + f"/ggml-model-{model_name.split('/')[-1]}-{ftype_str[ftype]}.bin"
fout = open(fname_out, "wb")

hparams["multiple_of"] = 1
fout.write(struct.pack("i", 0x67676d6c))  # magic: ggml in hex
fout.write(struct.pack("i", hparams["vocab_size"]))
fout.write(struct.pack("i", hparams["lack_words_size"]))
fout.write(struct.pack("i", hparams["hidden_size"]))
fout.write(struct.pack("i", hparams["multiple_of"]))
fout.write(struct.pack("i", hparams["n_head"]))
fout.write(struct.pack("i", hparams["n_layer"]))
fout.write(struct.pack("i", ftype))

for i in range(hparams["vocab_size"]):
    id = struct.pack("I", i)
    fout.write(id)
    text = tokenizer.decode([i]).encode('utf-8')
    fout.write(struct.pack("i", len(text)))
    fout.write(text)

for lack_word in lack_words:
    tokens = lack_word.split(" ")   
    ids = tokenizer.convert_tokens_to_ids(tokens)
    id = struct.pack('2H', *ids)
    fout.write(id)
    text = tokenizer.decode(ids).encode('utf-8')
    fout.write(struct.pack("i", len(text)))
    fout.write(text)

model = BloomForCausalLM.from_pretrained(
    model_name,
    config=config,
    torch_dtype=torch.float16 if ftype == 1 else torch.float32,
    low_cpu_mem_usage=True)
print("Model loaded: ", model_name)

list_vars = model.state_dict()
for name in list_vars.keys():
    src = name
    nn = name
    if name != "lm_head.weight":
        nn = nn.split(".")[1:]
    else:
        nn = nn.split(".")

    if nn[0] == "h":
        nn[0] = "layers"
        mapped = conv_map[".".join(nn[2:-1])]
        name = ".".join(nn[:2] + [mapped] + nn[-1:])
    else:
        mapped = conv_map[".".join(nn[:-1])]
        name = ".".join([mapped] + nn[-1:])

    if "query_key_value" in src:
        q, k, v = list_vars[src].reshape(config.n_head, 3, -1).unbind(1)
        list_vars[src] = torch.cat([q, k, v], dim=0).reshape_as(list_vars[src])

    print(src, ' -> ', name)
    data = list_vars[src].squeeze().numpy()
    data = data.astype(np.float32)

    n_dims = len(data.shape)
    print(name, n_dims, data.shape)

    # default type is fp32
    ftype_cur = 0
    if ftype == 1 and n_dims > 1:
        print("  Converting to float16")
        data = data.astype(np.float16)
        ftype_cur = 1

    # header
    str = name.encode('utf-8')
    fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
    for i in range(n_dims):
        fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
    fout.write(str)

    # data
    data.tofile(fout)

fout.close()

print("Done. Output file: " + fname_out)
print("")
