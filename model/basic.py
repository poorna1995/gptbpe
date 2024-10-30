
"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

This tokenizer is a simplified implementation of Byte Pair Encoding (BPE), a compression algorithm that iteratively merges the most common character pairs into new tokens. It follows the GPT-2 tokenizer's approach but with a few differences:
- It operates directly at the byte level (raw text bytes).
- It does not handle regular expression splitting patterns.
- It does not support any special tokens (e.g., padding, end-of-sentence).

## Attributes:
    merges (dict): A dictionary storing the merged pairs of byte tokens and their associated new token ID.
    vocab (dict): A dictionary mapping token IDs to the corresponding byte sequences.

## Methods:
    train(text, vocab_size, verbose=False): Trains the tokenizer on a given text to build a vocabulary of the specified size using byte pair encoding.
    encode(text): Encodes the input text string into a list of token IDs.
    decode(ids): Decodes a list of token IDs back into a text string.

## Example Usage:

1. Training the Tokenizer:

```python
# Create a tokenizer instance
tokenizer = BytePairTokenizer()

# Sample input text
text = "Hello, world! This is a test for Byte Pair Encoding."

# Train the tokenizer with a desired vocabulary size
vocab_size = 300
tokenizer.train(text, vocab_size, verbose=True)

# The tokenizer is now trained and can encode and decode text.
"""


from .base import Tokenizer, get_stats, merge

"""
    A byte-pair encoding (BPE) tokenizer that encodes and decodes text based on 
    byte-level tokenization and subsequent merging of the most frequent token pairs.

    The tokenizer starts by converting text into bytes (integers in the range 0-255). 
    Then, it iteratively merges the most common byte pairs to form new tokens, 
    allowing for an extended vocabulary.

    Attributes:
        merges (dict): A dictionary that maps pairs of byte tokens to new token ids.
        vocab (dict): A dictionary that maps token ids to their byte representations
            
"""


"""
        Trains the tokenizer by learning vocab_size - 256 merges. The vocab starts 
        with all single-byte tokens (0 to 255), and then merges the most frequent 
        byte pairs iteratively until the desired vocabulary size is reached.

        Args:
            text (str): The input text to train on.
            vocab_size (int): The desired size of the vocabulary. Must be at least 256.
            verbose (bool): If True, prints information about each merge.

        Example:
            tokenizer = BytePairTokenizer()
            tokenizer.train("hello world", vocab_size=300, verbose=True)
            
            # Sample output (with verbose=True):
            # merge 1/44: (108, 108) -> 256 (b'll') had 1 occurrences
            # merge 2/44: (101, 108) -> 257 (b'el') had 1 occurrences

        Raises:
            AssertionError: If vocab_size is less than 256.
"""

class BytePairTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
    def train(self, text, vocab_size, verbose=False):
        # vocab_size=300
        assert vocab_size >= 256
        #num_merges =300-256 = 44
        num_merges = vocab_size - 256
        # text = "i ampoorna praneesha, woking as a senior software engineer at hivepath an startup company"
        text_bytes = text.encode('utf-8')
        # ids =[101,103,52,124,90,35,13,101,103,235]
        ids = list(text_bytes)
        merges={}
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        for i in range(num_merges):
                
            #{(101,103):2,(103,52):1,(52,124):1}
            stats = get_stats(ids)
            # pair = #(101,103)
            pair = max(stats, key=stats.get)
            #idx = 256+0 =256
            idx = 256 + i
            # ids = [256,52,124,90,35,13,256,235]
            ids = merge(ids,pair,idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]]+vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()


        """
        Decodes a list of token ids back into a string.

        Args:
            ids (list): A list of token ids to decode.

        Returns:
            str: The decoded string.

        Example:
            tokenizer = BytePairTokenizer()
            tokenizer.train("hello world", vocab_size=300)
            
            encoded = tokenizer.encode("hello")
            print("Encoded:", encoded)  # Output: [104, 101, 108, 108, 111]

            decoded = tokenizer.decode(encoded)
            print("Decoded:", decoded)  # Output: hello
        """


    def decode(self,ids):
        # given ids (list of integers), return Python string
        text_bytes= b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode('utf-8',errors ='replace')
        return text


  
    def encode(self, text):



        """
        Encodes a string into a list of token ids.

        Args:
            text (str): The input text to encode.

        Returns:
            list: A list of token ids representing the encoded text.

        Example:
            tokenizer = BytePairTokenizer()
            tokenizer.train("hello world", vocab_size=300)
            
            token_ids = tokenizer.encode("hello")
            print(token_ids)  # Output: [104, 101, 108, 108, 111]
        """


        # given a string text, return the token ids
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            merges = {(1,2):10,(2,4):8}
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids




# Initializing and training the tokenizer
tokenizer = BytePairTokenizer()
tokenizer.train("hello world how are you doing how", vocab_size=258, verbose=True)

# Encoding a string
encoded = tokenizer.encode("hello")
print("Encoded:", encoded)  

# Decoding a list of token ids
decoded = tokenizer.decode(encoded)
print("Decoded:", decoded)  # Output: Decoded: hello