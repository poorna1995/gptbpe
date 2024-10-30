

### Let's build a Byte Pair Encoding Algorithm (Tokenization method) :

The project explains the process or workflow of the Byte pair Encoding(BPE) algorithm which is used in the LLMs Tokenization (GPT model) to convert the sequence of raw text into tokens (numerical representation). The BPE algorithm is "byte-level" because it runs on UTF-8 encoded strings.

                 
### Why do LLMs require Tokenisation?
Large language models (LLMs) require tokenization because, at their core, they do not inherently understand raw text. For LLMs to process and "understand" language, text must first be converted into a structured, numerical format that they can analyze. Tokenization serves this purpose by breaking down raw text into manageable units, called tokens, which represent meaningful segments of the text.

### Are there any methods to tokeniser?
Yes, there are many other methods such as:
- Character-based - ['1','q','f', 'h', 'k',.....]
- Word based - ['hello',' how',' there', 'enjoy',' holiday',.....]
- Subword-based (BPE - used in GPT, Wordpiece - used in BERT )  


### Now, let's understand what it is Byte Pair Encoding?
Byte Pair Encoding (BPE) tokenizer is a subword tokenization algorithm that splits words into smaller units and maps text data to integer sequences. It seems a quite simple right, but in terms of definition, it is as simple. but the magic involves in it it working process.




<img width="770" alt="Screenshot 2024-10-30 at 5 28 54 PM" src="https://github.com/user-attachments/assets/7ac1a230-1628-4c14-a84f-a072d4f17716">

                 
                 
  #### Converting the sequence of raw into tokens - (numerical representation)

 How are we achieving this? Let's dig into it is the working process :

 ### Let's see how the BPE Works.
 BPE iteratively merges the most frequent pairs of characters in a vocabulary. This process results in a vocabulary of variable-length character sequences that can represent an open vocabulary

 #### Workflow of BPE:
 
 1. **Initialize Vocabulary**: Start with a character-level vocabulary of unique words.
2. **Count Pair Frequencies**: Identify the most frequent adjacent character pairs in the vocabulary.
3. **Merge the Most Frequent Pair**: Replace instances of the most frequent pair with a new symbol.
4. **Repeat the Process**: Continue merging pairs until the vocabulary size meets requirements.
5. **Generate Tokens**: The resulting vocabulary then maps text to token sequences, which are converted to numerical representations.

This iterative process allows LLMs to represent language with fewer tokens while preserving efficiency.


![bpe](https://github.com/user-attachments/assets/37b86fe5-b4de-4e0a-b14c-57ea3c66b20f)



#### Key Benefits:
- **Computational Efficiency:** By optimizing tokenization, the model processes text with fewer, more meaningful tokens, reducing both training and inference time.
- **Memory Optimization:** A smaller, more efficient vocabulary reduces memory footprint, allowing for faster and more scalable model deployment.
- **Context Length Optimization:** Longer input sequences can be processed within the same context window, improving the model's ability to maintain context over extended text.



### Building and Running the BPE Algorithm

Now that we understand BPE, let’s implement it.

1. **Clone the Repository**
   ```bash
   git clone <repository_url>
   cd <repository_name>

2. **Create a Virtual Environment**
   ```bash
   python3 -m venv bpe_env
   source bpe_env/bin/activate  # Linux/Mac
   bpe_env\Scripts\activate  # Windows
  
3. **Install Dependencies Install all required packages:**
   ```bash
   pip install -r requirements.txt

  
4. **Understand the Code Structure**
   ```bash
   model/basic.py: Contains the main BytePairTokenizer() class implementing BPE logic.
   model/base.py: Holds helper functions supporting BytePairTokenizer() operations.

5. **Train the Model Run the training script on a sample corpus:**
   ```bash
   python train.py

7. **Test the Model Validate tokenization on sample input:**
   ```bash
   python test.py 


#### References 
Inspired by Andrej Karpathy's work on tokenization in LLMs, this implementation integrates foundational insights into token efficiency and model optimization. His teachings on BPE have shaped the approach used in this project.




