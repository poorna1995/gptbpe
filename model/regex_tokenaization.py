import regex as re
from .base import Tokenizer, get_freq,merge



# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):
    """
    A tokenizer that uses regular expressions to split text into tokens 
    and encodes those tokens using a Byte Pair Encoding (BPE) approach.

    Parameters:
    - pattern (str, optional): The regex pattern to use for tokenization. 
      If None, GPT-4's split pattern is used by default.

    Attributes:
    - special_tokens (dict): A dictionary of special tokens.
    - inverse_special_tokens (dict): A dictionary mapping token IDs to their corresponding special tokens.
    """

    def __init__(self, pattern=None):
        """
        Initializes the RegexTokenizer with a specified or default regex pattern.
        
        Parameters:
        - pattern (str, optional): The regex pattern for tokenization.
        
        Example:
        Input:
        >>> tokenizer = RegexTokenizer()  # Default pattern used

        Output:
        (None, an instance of RegexTokenizer is created)
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        # Compiles the regex pattern
        self.compiled_pattern = re.compile(self.pattern)
        # Initializes special token mappings
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        """
        Trains the tokenizer on the provided text using the specified vocabulary size.
        
        Parameters:
        - text (str): The input text to train on.
        - vocab_size (int): The desired vocabulary size.
        - verbose (bool, optional): If True, prints details of each merge operation.

        Example:
        Input:
        >>> tokenizer = RegexTokenizer()
        >>> tokenizer.train("Hello, world! This is a test.", vocab_size=300)

        Expected Output:
        (None, but prints merge operations if verbose=True)
        """
        assert vocab_size >= 256  # Ensures vocabulary size is at least 256
        # Calculates the number of merges needed
        num_merges = vocab_size - 256  # Input: 300 => Output: 44

        # Split the text into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)
        # Input: "Hello, world! This is a test sentence. <|endoftext|> Let's see how it works." 
        # Output: ['Hello', ',', 'world', '!', 'This', 'is', 'a', 'test', 'sentence', '.', '<|endoftext|>', 'Let', "'s", 'see', 'how', 'it', 'works', '.']

        # Convert text chunks to UTF-8 encoded byte arrays
        ids = list((ch.encode("utf-8")) for ch in text_chunks)
        # Input: ['Hello', ',', 'world', '!', 'This', 'is', 'a', 'test', 'sentence', '.', '<|endoftext|>', 'Let', "'s", 'see', 'how', 'it', 'works', '.']
        # Output:[b'Hello', b',', b'world', b'!', b'This', b'is', b'a', b'test', b'sentence', b'.', b'<|endoftext|>', b'Let', b"'s", b'see', b'how', b'it', b'works', b'.']

        merges = {}  # Merges to create new tokens
        vocab = {idx: bytes([idx]) for idx in range(256)}  # Initial vocabulary
        # Output: {0: b'\x00', 1: b'\x01', ..., 255: b'\xff'}

        for i in range(num_merges):
            # Count the occurrences of every consecutive pair
            stats = {}
            for chunks_ids in ids:
                get_freq(chunks_ids, stats)
            # Input: ids may look like [b'Hello', b',', b'world', ...]
            # Expected Output: stats = [(72, 101): 1, (101, 108): 1, ...]

            # Find the most common pair
            pair = max(stats, key=stats.get)
            # Example Output: pair = (116, 101) # 'e', 'l' from "Hello"

            idx = 256 + i  # Assigning new index for the merged token
            # Example: idx = 256 for the first merge

            # Merge the pair in the token lists
            ids = [merge_ids(chunk_ids, pair, idx) for chunk_ids in ids]
            # Output: [[72, 101, 108, 108, 111], [44], [119, 111, 114, 108, 100], [33], [84, 104, 105, 115], [105, 115], [97], [256, 115, 116], [115, 101, 110, 256, 110, 99, 101], [46], [60, 124, 101, 110, 100, 111, 102, 256, 120, 116, 124, 62], [76, 101, 116], [39, 115], [115, 101, 101], [104, 111, 119], [105, 116], [119, 111, 114, 107, 115], [46]]

            merges[pair] = idx  # Store the merge
            # Output: merges = {(101, 108): 256, ...}

            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]  # Update vocabulary
            # Output: vocab = {256: b'el', ...}

            if verbose:
                print(f"merge_ids {i + 1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
                # Example Output: merge 1/44: (101, 108) -> 256 (b'el') had 5 occurrences

        # Save class variables
        self.merges = merges  # Used in encode()
        self.vocab = vocab    # Used in decode()

    def register_special_token(self, special_tokens):
        """
        Registers special tokens and their corresponding IDs for encoding/decoding.

        Parameters:
        - special_tokens (dict): A dictionary mapping special token strings to unique IDs.

        Example:
        Input:
        >>> tokenizer = RegexTokenizer()
        >>> tokenizer.register_special_token({'[PAD]': 256, '[UNK]': 257})

        Expected Output:
        (None, but the internal mapping for special tokens is updated)
        """
        self.special_tokens = special_tokens
        # Example: special_tokens = {'[PAD]': 256, '[UNK]': 257}
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}
        # Output: inverse_special_tokens = {256: '[PAD]', 257: '[UNK]'}

    def decode(self, ids):
        """
        Decodes a list of token IDs back into the original text.

        Parameters:
        - ids (list of int): A list of token IDs to decode.

        Returns:
        - str: The decoded text.

        Example:
        Input:
        >>> tokenizer = RegexTokenizer()
        >>> tokenizer.train("Hello, world! This is a test.", vocab_size=300)
        >>> tokenizer.encode_ordinary("Hello")
        >>> ids = [72, 101, 256]  # Example token IDs
        >>> decoded_text = tokenizer.decode(ids)

        Expected Output:
        'Hello'  # Assuming [256] corresponds to a valid token
        """
        parts_bytes = []

        for idx in ids:
            if idx in self.vocab:
                parts_bytes.append(self.vocab[idx])
                # Example Output: For idx = 72, parts_bytes becomes [b'H']
            elif idx in self.inverse_special_tokens:
                parts_bytes.append(self.inverse_special_tokens[idx])
                # Example Output: For idx = 256, parts_bytes becomes ['[PAD]']
            else:
                raise ValueError(f'Invalid token id: {idx}')
        
        text_bytes = b''.join(parts_bytes)  # Join all byte parts
        # Output: b'Hello'

        text = text_bytes.decode("utf-8", errors='replace')  # Convert bytes to string
        # Expected Output: 'Hello'
        return text

    def _encode_chunk(self, text_bytes):
        """
        Encodes a chunk of text into token IDs.

        Parameters:
        - text_bytes (bytes): The chunk of text to encode, represented as bytes.

        Returns:
        - list of int: The list of token IDs.

        Example:
        Input:
        >>> tokenizer = RegexTokenizer()
        >>> chunk_ids = tokenizer._encode_chunk(b'Hello')

        Expected Output:
        [72, 101, 108, 108, 111]  # Example output based on encoding (actual values may vary)
        """
        ids = list(text_bytes)  # Convert bytes to list of integers
        # Input: b'Hello' => ids = [72, 101, 108, 108, 111]

        while len(ids) >= 2:
            # Find the pair with the lowest merge index
            stats = get_freq(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # Example Output: pair = (72, 101) for 'H', 'e'

            if pair not in self.merges:
                break  # No more merges available
            
            idx = self.merges[pair]  # Get the index of the merge
            # Example Output: idx = 256
            ids = merge_ids(ids, pair, idx)  # Merge the best pair
            # Output: ids updated with merged tokens

        return ids  # Return the final token IDs

    def encode_ordinary(self, text):
        """
        Encodes the given text into token IDs, ignoring special tokens.

        Parameters:
        - text (str): The input text to encode.

        Returns:
        - list of int: The list of token IDs corresponding to the input text.

        Example:
        Input:
        >>> tokenizer = RegexTokenizer()
        >>> tokenizer.train("Hello, world! This is a test.", vocab_size=300)
        >>> ids = tokenizer.encode_ordinary("Hello, world!")

        Expected Output:
        [72, 101, 108, 108, 111, 44, 32, 119, 111, 114, 108, 100, 33]  # Example output based on encoding
        """
        text_chunks = re.findall(self.compiled_pattern, text)
        # Input: "Hello, world!" 
        # Output: ['Hello', ',', 'world', '!']

        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids
    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        # decode the user desire w.r.t. handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)
        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids



# Initialize and train the tokenizer
if __name__ == "__main__":
    # Create an instance of RegexTokenizer
    reg_tokenizer = RegexTokenizer()

    # Train the tokenizer with a sample text and a specified vocabulary size
    reg_tokenizer.train("hello world how are you doing how", vocab_size=290, verbose=True)