


### Let's build a Byte Pair Encoding Algorithm (Tokenization method) :

The project explains the process or workflow of the Byte pair Encoding(BPE) algorithm which is used in the LLMs Tokenization (GPT model) to convert the sequence of raw text into tokens (numerical representation). The BPE algorithm is "byte-level" because it runs on UTF-8 encoded strings.


<img width="771" alt="Screenshot 2024-10-30 at 5 28 54 PM" src="https://github.com/user-attachments/assets/7ac1a230-1628-4c14-a84f-a072d4f17716">


### Why do LLMs require Tokenisation?
Large language models (LLMs) require tokenization because, at their core, they do not inherently understand raw text. For LLMs to process and "understand" language, text must first be converted into a structured, numerical format that they can analyze. Tokenization serves this purpose by breaking down raw text into manageable units, called tokens, which represent meaningful segments of the text.

### Are there any methods to tokeniser?
Yes, there are many other methods such as:
- Character-based - ['1','q','f', 'h', 'k',.....]
- Word based - ['hello',' how',' there', 'enjoy',' holiday',.....]
- Subword-based (BPE - used IN GPT, Wordpiece - used in BERT )  


### Now, let's understand what it is Byte Pair Encoding?
Byte Pair Encoding (BPE) tokenizer is a subword tokenization algorithm that splits words into smaller units and maps text data to integer sequences. It seems a quite simple right, but in terms of definition it is as simple. but the magic involves in it it working process.

 ### Let's see how the BPE Works.
 BPE iteratively merges the most frequent pairs of characters in a vocabulary. This process results in a vocabulary of variable-length character sequences that can represent an open vocabulary

 #### Workflow of BPE:

 ![bpe](https://github.com/user-attachments/assets/51d69f84-a777-4748-b578-84824561c7f4)


Key Benefits:
- **Computational Efficiency:** By optimizing tokenization, the model processes text with fewer, more meaningful tokens, reducing both training and inference time.
- **Memory Optimization:** A smaller, more efficient vocabulary reduces memory footprint, allowing for faster and more scalable model deployment.
- **Context Length Optimization:** Longer input sequences can be processed within the same context window, improving the model's ability to maintain context over extended text.




