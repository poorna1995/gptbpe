Skip to content
Navigation Menu
poorna1995
/
gptbpe

Type / to search
Code
Issues
Pull requests
Actions
Projects
Wiki
Security
Insights
Settings
Editing README.md in gptbpe
Breadcrumbsgptbpe
/
README.md
in
main

Edit

Preview
Indent mode

Spaces
Indent size

2
Line wrap mode

Soft wrap
Editing README.md file contents
Selection deleted


1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96


### Let's build a Byte Pair Encoding Algorithm (Tokenization method) :

The project explains the process or workflow of the Byte pair Encoding(BPE) algorithm which is used in the LLMs Tokenization (GPT model) to convert the sequence of raw text into tokens (numerical representation). The BPE algorithm is "byte-level" because it runs on UTF-8 encoded strings.

                 
### Why do LLMs require Tokenisation?
Large language models (LLMs) require tokenization because, at their core, they do not inherently understand raw text. For LLMs to process and "understand" language, text must first be converted into a structured, numerical format that they can analyze. Tokenization serves this purpose by breaking down raw text into manageable units, called tokens, which represent meaningful segments of the text.

### Are there any methods to tokeniser?
Yes, there are many other methods such as:
- Character-based - ['1','q','f', 'h', 'k',.....]
- Word based - ['hello',' how',' there', 'enjoy',' holiday',.....]
- Subword-based (BPE - used IN GPT, Wordpiece - used in BERT )  


### Now, let's understand what it is Byte Pair Encoding?
Byte Pair Encoding (BPE) tokenizer is a subword tokenization algorithm that splits words into smaller units and maps text data to integer sequences. It seems a quite simple right, but in terms of definition, it is as simple. but the magic involves in it it working process.




<img width="771" alt="Screenshot 2024-10-30 at 5 28 54 PM" src="https://github.com/user-attachments/assets/7ac1a230-1628-4c14-a84f-a072d4f17716">

                 
                 
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


 ![bpe](https://github.com/user-attachments/assets/51d69f84-a777-4748-b578-84824561c7f4)


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

  python3 -m venv bpe_env
  source bpe_env/bin/activate  # Linux/Mac
  bpe_env\Scripts\activate  # Windows
  
3. **Install Dependencies Install all required packages:**
   
  pip install -r requirements.txt
4. **Understand the Code Structure**

  model/basic.py: Contains the main BytePairTokenizer() class implementing BPE logic.
  model/base.py: Holds helper functions supporting BytePairTokenizer() operations.

5. **Train the Model Run the training script on a sample corpus:**
    python train.py

6. **Test the Model Validate tokenization on sample input:**
    python test.py --input "Hello, world!"


#### References & Acknowledgment
Inspired by Andrej Karpathy's work on tokenization in LLMs, this implementation integrates foundational insights into token efficiency and model optimization. His teachings on BPE have shaped the approach used in this project.




