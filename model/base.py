"""
Contains the base Tokenizer class and a few common helper functions.
The base class also contains the (common) save/load functionality.
It would be possible to be a lot more strict about the interface and
e.g. isolating all regex/pattern parts to the RegexTokenizer, but
some concessions are made for simplicity.
"""
import unicodedata

# -----------------------------------------------------------------------------
# a few helper functions useful for both BytePairTokenizer and RegexTokenizer



"""
    Given a list of integers, return a dictionary of counts of consecutive pairs.
    
    Args:
        ids (list of int): A list of integers representing token IDs.
        counts (dict, optional): An existing dictionary of counts. If provided,
                                 the counts will be updated.
    
    Returns:
        dict: A dictionary where keys are consecutive pairs of integers (as tuples) 
              from `ids`, and values are their counts.
    
    Example:
        >>> ids = [1, 2, 3, 1, 2]
        >>> get_stats(ids)
        {(1, 2): 2, (2, 3): 1, (3, 1): 1}
"""
def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts


"""
    In the list of integers `ids`, replace all consecutive occurrences of `pair` 
    with a new integer token `idx`.
    
    Args:
        ids (list of int): A list of integers representing token IDs.
        pair (tuple of int): A tuple containing two integers to be merged.
        idx (int): The new integer token that replaces the pair.
    
    Returns:
        list: A new list of integers with the pair replaced by `idx`.

    Example:
        >>> ids = [1, 2, 3, 1, 2]
        >>> merge(ids, (1, 2), 4)
        [4, 3, 4]
"""
def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids




"""
    Replaces control characters in a string with their Unicode escape sequences.

    Control characters are characters that may affect the output display (e.g., newlines, 
    null characters, etc.). This function replaces them with their Unicode escape sequences 
    to avoid output distortion.

    Control characters are identified using their Unicode category (starting with 'C'),
    and are escaped to avoid distortion in the output.

    Parameters:
    s (str): The input string.

    Returns:
    str: A string with control characters replaced by their Unicode escape sequences.

    Example:
    --------
    >>> replace_control_characters("Hello\nWorld")
    'Hello\\u000aWorld'

    >>> replace_control_characters("Tab\tCharacter")
    'Tab\\u0009Character'

\0 -Null,\n -newline ,\r-caraige return

"""


# first two helper functions...
def replace_control_characters(s: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table

    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes) -> str:

    """
    Decodes a byte token into a UTF-8 string, replacing control characters 
    with their Unicode escape sequences.

    This function is useful for rendering byte tokens that may contain control 
    characters, which can cause display issues. The control characters are 
    escaped using `replace_control_characters`.

    Parameters:
    t (bytes): The input byte token.

    Returns:
    str: The decoded string with control characters replaced.

    Example:
    --------
    >>> render_token(b'Hello\\nWorld')
    'Hello\\u000aWorld'

    >>> render_token(b'Example\\x00Token')
    'Example\\u0000Token'
    """


    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

# -----------------------------------------------------------------------------
# the base Tokenizer class

"""
        Build a vocabulary from merges and special tokens.

        The vocabulary is initialized with 256 single-byte tokens (0-255). 
        Then, the merge pairs (tuples of two tokens) are added, followed 
        by any special tokens such as <PAD> or <UNK>.
        # Example Usage
                merges = {
                    (65, 66): 256,  # A + B -> AB
                    (67, 68): 257   # C + D -> CD
                }

                special_tokens = {
                    "<PAD>": 258,
                    "<UNK>": 259
                }

        Returns:
            dict: A dictionary mapping token indices to byte sequences (or special tokens)."""

class Tokenizer:
    """Base class for Tokenizers"""

    def __init__(self):
        # default: vocab size of 256 (all bytes), no merges, no patterns
        self.merges = {} # (int, int) -> int
        self.pattern = "" # str
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab() # int -> bytes

    def train(self, text, vocab_size, verbose=False):
        # Tokenizer can train a vocabulary of size vocab_size from text
        raise NotImplementedError

    def encode(self, text):
        # Tokenizer can encode a string into a list of integers
        raise NotImplementedError

    def decode(self, ids):
        # Tokenizer can decode a list of integers into a string
        raise NotImplementedError

    def _build_vocab(self):

        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    """
    Saves two files: file_prefix.vocab and file_prefix.model

    - model file: Contains the critical data needed to load the model later.
    - vocab file: Provides a human-readable version of the vocabulary, intended for inspection only.

    The process is inspired by (but not equivalent to) SentencePiece's model saving mechanism.

    Parameters:
    -----------
    file_prefix : str
        The prefix used to generate filenames for saving the model and vocab files. 
        For example, 'my_bpe' will create 'my_bpe.model' and 'my_bpe.vocab'.
    
    What gets saved:
    ----------------
    1. `.model` file:
        - Version: The version of this format.
        - Pattern: A regex pattern associated with tokenization.
        - Special tokens: Number of special tokens followed by each token and its index.
        - Merges: The token pairs merged together during training.

    2. `.vocab` file:
        - Vocab: A human-readable list of tokens, with descriptions of merges where applicable.
        - Tokens: Rendered in a form suitable for inspection, but not usable for loading due to potential decoding issues.

    Example:
    --------
    >>> class MyTokenizer:
    >>>     def __init__(self):
    >>>         self.pattern = "\\w+|\\S"  # Example pattern for tokenization
    >>>         self.special_tokens = {"<pad>": 0, "<unk>": 1}
    >>>         self.merges = {(100, 101): 200}
    >>>         self.vocab = {0: b"<pad>", 1: b"<unk>", 100: b"a", 101: b"b", 200: b"ab"}

    >>>     def save(self, file_prefix):
    >>>         # [Your save implementation goes here]
    >>>         pass

    >>> tokenizer = MyTokenizer()
    >>> tokenizer.save('my_bpe')

    This will generate two files:
    - `my_bpe.model`: Contains the version, pattern, special tokens, and merges.
    - `my_bpe.vocab`: Provides a readable version of the vocabulary for inspection.
    """
    """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
    """


    def save(self, file_prefix):

        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("model v1\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # the merges dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        # write the vocab: for the human to look at
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")


    """
    Load a model from a specified file.

    This method reads a model file that follows the format specified by the 
    model tokenizer. It extracts the version, pattern, special tokens, 
    and merges from the file and populates the corresponding attributes 
    in the current instance.

    Args:
        model_file (str): The path to the model file to load. 
                          Must end with ".model".

    Raises:
        AssertionError: If the model file does not end with ".model" 
                        or if the version in the file is not "minbpe v1".

    Example:
        # Create an instance of the class that contains the load method
        tokenizer = MyTokenizer()  # Assuming MyTokenizer is the class name
        
        # Load a model from a file
        tokenizer.load("my_model.model")

        # Access the loaded merges and special tokens
        print(tokenizer.merges)           # Should print the loaded merges
        print(tokenizer.special_tokens)    # Should print the loaded special tokens

    Notes:
        The model file must follow this structure:
        - The first line should contain the version: "minbpe v1".
        - The second line should contain the pattern used for tokenization.
        - The third line should specify the number of special tokens.
        - Each subsequent line should define a special token followed by its index.
        - After the special tokens, each line should define a pair of indices representing 
          merges, which will be read until the end of the file.
    """



    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "minbpe v1"
            # read the pattern = self.pattern = "\w+"

            self.pattern = f.readline().strip()
            # read the no of the special tokens = 10
            num_special = int(f.readline().strip())
            # this loop will iterate till te num_special
            # <sls>:0 , <sep>:1
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            # 0 1 --> idx1=0,idx2=1, ---> merges[(0,1)] =256 --> idx is incremented to the 257

            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()






# readline() - this method rads the nextline from th efile na dreturn tas steirng. including newline char(\n)
#strip this = d is mecalled on the string returned by readline(. it removes anywhitespace from the begineing and the end of the string)

# split( )- this methid split th he string into a list of subsbasd on whitepsace(spaces,tabs,etc)