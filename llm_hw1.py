import torch as t
from transformers import GPT2LMHeadModel, GPT2Tokenizer  # pip install transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
import re  # regular expressions, useful for decoding the output

def num_to_words(input_s):
    ones = ['zero','one','two','three','four','five','six','seven','eight','nine']
    tens = ['','ten','twenty','thirty','forty','fifty']

    operands = input_s.split("+")
    back = operands[1].split("=")
    operands[1] = back[0]
    operator = "plus"
    equals = "equals"

    operand_words = []
    for operand in operands:
        operand = int(operand)
        if operand <= 50:
            if operand < 10:
                operand_word = ones[operand]
            elif operand < 20:
                operand_word = " ".join([tens[1], ones[operand-10]])
            else:
                tens_digit = operand // 10
                ones_digit = operand % 10
                if ones_digit == 0:
                    operand_word = tens[tens_digit]
                else:
                    operand_word = " ".join([tens[tens_digit], ones[ones_digit]])
            operand_words.append(operand_word)
    phrase = operand_words[0] + " " + operator + " " + operand_words[1] + " " + equals
    return phrase

def words_to_num(word_string):
    words = word_string.split()
    ones = {'zero':0, 'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9, 'eleven':11, 'twelve':12}
    tens = {'ten':10, 'twenty':20, 'thirty':30, 'forty':40, 'fifty':50}
    total = 0
    for i in range(len(words)):
        if words[i] in ones:
            total += ones[words[i]]
        elif words[i] in tens:
            total += tens[words[i]]
        elif words[i].endswith('teen') and words[i][:-4] in ones:
            total += ones[words[i][:-4]] + 10
        else:
            return -1
    return total


def create_dataset(i_start=0, i_end=50, operation=t.add):
    """(1 pt) Create a dataset of pairs of numbers to calculate an operation on.
    DO NOT USE A FOR LOOP. Use pytorch functions, possibilities include meshgrid, stack, reshape, repeat, tile.
    (Note you'll have to use for loops on string stuff in other functions)

    The dataset should be a tuple of two tensors, X and y, where X is a Nx2 tensor of numbers to add,
    and y is a N tensor of the correct answers.
    E.g., if i_start=0, i_end=2, then X should be tensor([[0,0,1,1],[0,1,0,1]]).T and y should be tensor([0,1,1,2]).
    I recommend doing all pairs of sums involving 0-49, but you may modify this.
    """
    # TODO
    i = t.arange(i_start, i_end)
    r, c = t.meshgrid(i, i)
    X = t.stack([r.flatten(), c.flatten()], dim=0)
    y = t.sum(X, dim=0)
    return X, y

def load_LLM(default="EleutherAI/gpt-neo-1.3B", device='cpu'):
    """(1 pt) Load a pretrained LLM and put on device. Default choice is a large-ish GPT-neo-2.7B model on Huggingface.
    Could also consider the "open GPT" from facebook: "facebook/opt-2.7b", or others
    here: https://huggingface.co/models?pipeline_tag=text-generation
    Explicitly load model and tokenizer, don't use the huggingface "pipeline" which hides details of the model
    (and it also has no batch processing, which we need here)
    """
    # TODO
    tokenizer = AutoTokenizer.from_pretrained(default)
    model = AutoModelForCausalLM.from_pretrained(default).to(device)
    return model, tokenizer

def encode_problems(X, strategy='baseline'):
    """(1 pts) Encode the problems as strings. For example, if X is [[0,0,1,1],[0,1,0,1]],
    then the baseline output should be ["0+0=", "0+1=", "1+0=", "1+1="]"""
    output_strings = []
    lst = X.tolist()
    X_string = [[str(elem) for elem in sublst] for sublst in lst]
    #for xi in X:
    for xi in range(X.shape[1]):
        e_string = str(X_string[0][xi]) + "+" + str(X_string[1][xi]) + "="
        if strategy == 'baseline':
            # TODO: encode_string =
            encode_string = e_string
        else:
            # TODO: encode_string =
            encode_string = num_to_words(e_string)
        output_strings.append(encode_string)
    return output_strings

def generate_text(model, tokenizer, prompts, verbose=True, device='cpu'):
    """(3 pts) Complete the prompt using the LLM.
    1. Tokenize the prompts: https://huggingface.co/docs/transformers/preprocessing
        Put data and model on device to speed up computations
        (Note that in real life, you'd use a dataloader to do this efficiently in the background during training.)

    2. Generate text using the model.
        Turn off gradient tracking to save memory.
        Determine the sampling hyper-parameters.
        You may need to do it in batches, depending on memory constraints

    3. Use the tokenizer to decode the output.
    You will need to optionally print out the tokenization of the input and output strings for use in the write-up.
    """
    t0 = time.time()
    # TODO: tokenize
    # TODO: generate text, turn off gradient tracking
    # TODO: decode output, output_strings = ...
    output_strings = []
    for input_string in prompts:
        input_ids = tokenizer.encode(input_string, return_tensors="pt").to(device)
        attention_mask = t.ones(input_ids.shape, dtype=t.long, device=input_ids.device)
        with t.no_grad():
            out_tokens = model.generate(input_ids, attention_mask=attention_mask, max_length=input_ids.size(1)+1, do_sample=True, temperature=0.0001)
        next_word = tokenizer.decode(out_tokens[0][input_ids.size(1):], skip_special_tokens=True)
        output_strings.append(next_word)

    if verbose:
        # TODO: print example tokenization for write-up
        print("predicted: ", output_strings)
    print("Time to generate text: ", time.time() - t0)  # It took 4 minutes to do 25000 prompts on an NVIDIA 1080Ti.
    return output_strings

def decode_output(output_strings, strategy='baseline', verbose=True):
    """(1 pt) Decode the output strings into a list of integers. Use "t.nan" for failed responses.
    One suggestion is to split on non-numeric characters, then convert to int. And use try/except to catch errors.
    """
    print(output_strings)
    y_hat = []
    for s in output_strings:
        # TODO: y = f(s)
        r = None
        if s.isdigit():
            r = int(s)
        else:
            r = words_to_num(s)
            if r == -1:
                r = t.nan
        y_hat.append(r)
    return y_hat

def analyze_results(X, y, y_hats, strategies):
    """(3 pts) Analyze the results.
    Output the accuracy of each strategy.
    Plot a scatter plot of the problems “x1+x2” with x1,x2 on each axis,
    and different plot markers to indicate whether the answer from your LLM was correct.
    (See write-up instructions for requirements on plots)
    Train a classifier to predict whether the LLM gave the correct response (using scikit-learn, for example)
    and plot the classifier boundary over the scatter plot with “contour”. (Use whatever classifier looks appropriate)"""
    # TODO
    yhat_split = np.split(np.array(y_hats), 2)
    y_nump = np.asarray(y).ravel()
    x1 = X[0].numpy()
    x2 = X[1].numpy() 
    i = 0

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    for strategy in strategies:
        yhat = yhat_split[i].ravel()

        result = np.where(yhat == y_nump, 1, 0)
        acc = (np.count_nonzero(result == 1) / len(result)) * 100
        
        clf = LogisticRegression(random_state=0, max_iter=2000, tol=1e-4).fit(np.column_stack((x1, x2)), y_nump)
        tit = "{} accuracy: {}".format(strategy, acc)

        xx, yy = np.meshgrid(np.linspace(-1, np.max(x1) + 1, 100), np.linspace(-1, np.max(x1) + 1, 100))
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        axs[i].contourf(xx, yy, Z, cmap=plt.cm.RdBu_r, alpha=.8)
        axs[i].scatter(x1[result==0], x2[result==0], c='r', marker='x', label='Incorrect')
        axs[i].scatter(x1[result==1], x2[result==1], c='g', marker='o', label='Correct')
        axs[i].set_xlabel('x1')
        axs[i].set_ylabel('x2')
        axs[i].set_title(tit)
        axs[i].legend()
        i = i + 1
    plt.show()

if __name__ == "__main__":
    device = t.device("cuda" if t.cuda.is_available() else "cpu")  # Use GPU if available
    device = t.device('mps') if t.backends.mps.is_available() else device  # Use Apple's Metal backend if available

    X, y = create_dataset(0, 50)
    model, tokenizer = load_LLM(device=device)

    y_hats = []  # list of lists of predicted answers, y_hat, for each strategy
    strategies = ['baseline', 'new']
    for strategy in strategies:
        input_strings = encode_problems(X, strategy=strategy)
        output_strings = generate_text(model, tokenizer, input_strings, device=device)
        #output_strings = [out_s[len(in_s):] for in_s, out_s in zip(input_strings, output_strings)]  # Remove the input string from generated answer
        y_hats.append(decode_output(output_strings, strategy=strategy))

    analyze_results(X, y, y_hats, strategies)

