import sys
import re
import pdb

def remove_multi_space(s):
    s_len = len(s)
    s = s.replace('  ', ' ')
    if s_len > len(s):
        remove_multi_space(s)
    return s

def fix_bracket_spaces(s):
    s = s.replace('(', ' ( ')
    s = s.replace(')', ' ) ')
    s = remove_multi_space(s)
    s = s.replace(' ( ', '( ')
    return s

def fix_op(s):
    s = re.sub("Op\( (.*?) \)", r"Op(\1)", s)
    return s

def fix_const(s):
    s = re.sub("Const\( ([0-9]*) \)", r"Const(\1)", s)
    return s


def remove_hash(s):
    return re.sub("SyntaxTree.*: ", "", s)

def remove_newlines(s):
    return re.sub("\n", "<end>", s)

def clean_string(s):    
     return remove_hash(
            remove_newlines(
            fix_op( 
            fix_const(
            fix_bracket_spaces(
                s)))))

def to_tokens(string_list, vocab=None):
    use_supplied_vocab = True
    if vocab is None:
        use_supplied_vocab = False
        vocab = {}
        vocab['<pad>'] = len(vocab)
        vocab['<unk>'] = len(vocab)
        vocab['<start>'] = len(vocab)
        vocab['<end>'] = len(vocab)

    strings_as_int_lists = []
    for s in string_list:
        s_as_int_list = [vocab['<start>']]
        for t in s.split(' '):
            tt = t
            if t not in vocab:
                if use_supplied_vocab:
                    tt = '<unk>'
                else:
                    vocab[t] = len(vocab)
            s_as_int_list.append(vocab[tt])
        strings_as_int_lists.append(s_as_int_list)

    return strings_as_int_lists, vocab

def save_vocab(vocab, vocab_filename):
    sorted_vocab = [None]*len(vocab)
    for k,v in vocab.iteritems():
        sorted_vocab[v] = k

    with open(vocab_filename, 'w') as f:
        for i,v in enumerate(sorted_vocab):
            print >> f, "%i %s" % (i, v)

    print "saved vocab: %s" % vocab_filename

def load_vocab(vocab_filename):
    with open(vocab_filename, 'r') as f:
        vocab_lines = map(lambda x: x.split(' ')[1].replace('\n',''), f.readlines())
    vocab = {}
    for i,v in enumerate(vocab_lines):
        vocab[v] = i
    return vocab

def invert_vocab(vocab):
    inv_vocab = [None]*len(vocab)
    for k,v in vocab.iteritems():
        inv_vocab[v] = k
    return inv_vocab

def save_data(as_int_lists, data_filename):
    with open(data_filename, 'w') as f:
        for L in as_int_lists:
            print >> f, " ".join(map(str, L))

    print "saved data: %s" % data_filename

def main():
    data_file = sys.argv[1]

    with open(data_file) as f:
        programs_as_strings = f.readlines()

    programs_as_clean_strings = map(lambda x: clean_string(x), programs_as_strings)
    input_vocab = None if len(sys.argv) < 3 else load_vocab(sys.argv[2])
    programs_as_int_lists, vocab = to_tokens(programs_as_clean_strings, vocab = input_vocab)

    if input_vocab is None:
        save_vocab(vocab, "vocab.txt")

    save_data(programs_as_int_lists, 'tokens_%s' % data_file)


if __name__ == '__main__':
    main()