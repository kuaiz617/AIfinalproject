import re
import jieba.posseg as posseg
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer # Convert text collection to TF-IDF feature matrix
from sklearn.metrics.pairwise import cosine_similarity # Compute cosine similarity between two vectors
import os
###################################################
# TextRank Implementation
###################################################


# Stopwords file path
stopwords_path = 'stopwords.txt'
# POS tags to exclude
stopPOS = []

# Load stopwords safely
if not os.path.exists(stopwords_path):
    print(f" Stopwords file not found at: {stopwords_path}")
    stopwords = []  # fallback: use empty list if not found
else:
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f.readlines()]
    print(f" Loaded {len(stopwords)} stopwords from {stopwords_path}")


def segment_text_to_sentence(text):
    # Split text into sentences
    sentences = re.split(r'[。！？!?]', text)
    sentences = [sentence.strip().replace(" ", "").replace('\n', '') for sentence in sentences if sentence.strip()]
    return sentences


def segment_text_to_words(text, use_stopwords):
    # Segment text into words and remove stopwords
    global stopPOS, stopwords
    stopPOS = [item.lower() for item in stopPOS]
    words = posseg.cut(text)
    if use_stopwords:
        words = [word for word, flag in words if flag[0].lower() not in stopPOS and word not in stopwords]
    else:
        words = [word for word, flag in words if flag[0].lower() not in stopPOS]
    words = set(words)

    return words


def original_similarity_matrix(sentences, use_stopwords):
    # Calculate original similarity matrix
    sentence_words = [set(segment_text_to_words(item, use_stopwords)) for item in sentences]
    size = len(sentences)
    similarity_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(i + 1, size):
            if len(sentence_words[i]) == 0 or len(sentence_words[j]) == 0:
                similarity = 0
            else:
                # Compute Similarity
                similarity = len(sentence_words[i] & sentence_words[j]) / (
                            np.log(len(sentence_words[i])) + np.log(len(sentence_words[i])) + 1e-10)
            similarity_matrix[i][j] = similarity_matrix[j][i] = similarity

    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # # Assume we already have a similarity matrix named similarity_matrix
    # # similarity_matrix = ...
    #
    # # Plot a heatmap
    # plt.figure(figsize=(10, 8))  # Set the figure size
    # sns.heatmap(similarity_matrix, cmap='coolwarm', linewidths=0.5, annot=True, fmt=".2f")
    # plt.title('Similarity Matrix Heatmap')  # set the title
    # plt.xlabel('Sentences')  # Set the x-axis label
    # plt.ylabel('Sentences')  # Set the y-axis label
    # # Save the image to the current directory
    # plt.savefig('similarity_matrix_heatmap.png', dpi=300)
    # plt.show()

    return similarity_matrix


def cosine_tfidf_similarity_matrix(sentences, use_stopwords):
    # Calculate cosine similarity matrix based on TF-IDF
    sentence_words = [' '.join(segment_text_to_words(item, use_stopwords)) for item in sentences]
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentence_words)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Set the diagonal elements to 0 to avoid interference from self-similarity.
    np.fill_diagonal(similarity_matrix, 0)

    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # # Assume that we already have a similarity matrix named similarity_matrix
    # # similarity_matrix = ...
    # # Plot a heatmap
    # plt.figure(figsize=(10, 8))  # Set the figure size
    # sns.heatmap(similarity_matrix, cmap='coolwarm', linewidths=0.5, annot=True, fmt=".2f")
    # plt.title('Cosine_tfidf Similarity Matrix Heatmap')  # set the title
    # plt.xlabel('Sentences')  # Set the x-axis label
    # plt.ylabel('Sentences')  # Set the y-axis label
    # # Save the image to the current directory
    # plt.savefig('TF-IDF Cosine similarity matrix heatmap.png', dpi=300)
    # plt.show()
    return similarity_matrix


def summarize_text_rank(text, d=0.85, iter_num=200, top=3, method='Default Metric', use_stopwords=True):
    sentences = segment_text_to_sentence(text)

    print('---------Start----------------------------------------')
    if method == 'Default Metric':
        edge_weight = original_similarity_matrix(sentences, use_stopwords)
    elif method == 'TF-IDF':
        edge_weight = cosine_tfidf_similarity_matrix(sentences, use_stopwords)

    node_weight = np.ones((len(sentences)))

    for num in range(iter_num):
        # TextRank Iterative formula
        node_weight_new = (1 - d) + d * node_weight @ (edge_weight / (edge_weight.sum(axis=-1) + 1e-10)).T
        if ((node_weight_new - node_weight) ** 2).sum() < 1e-10:
            break
        node_weight = node_weight_new

    if num < iter_num:
        print('Converged after {} iterations'.format(num))
    else:
        print('Did not converge after {} iterations'.format(num))

    sorted_indices = np.argsort(node_weight)[::-1]

    # Get the top values and their corresponding indices
    top_indices = sorted(sorted_indices[:top])
    top_values = node_weight[top_indices]

    print('Top {} values:'.format(top), top_values)
    print('Corresponding indices:', top_indices)
    print('Result:')
    result = ''
    for idx in top_indices:
        result += sentences[idx] + '。\n'
    print(result)

    return result

# Example
# text = 'Enter your text here'
# summarize_text_rank(text)


###################################################
# MT5 Implementation
###################################################

flag = True

# Try to import the necessary libraries and models
try:
    import re
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    # Define a function to handle spaces and line breaks
    WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

    # Define the name of the MT5 model
    model_name = "./mt5-base"
    # model_name = "google/mt5-base"

    # Use AutoTokenizer to load the pre-trained MT5 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Use AutoModelForSeq2SeqLM to load the pre-trained MT5 model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
except:
    # If the import fails, set the flag to False
    flag = False


def summary_mt5(text):
    global flag
    # Check whether the MT5 model was imported successfully
    if not flag:
        return 'The MT5 model was not imported'

    try:
        # Use the MT5 tokenizer to process the input text and generate input token IDs
        input_ids = tokenizer(
            [WHITESPACE_HANDLER(text)],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )["input_ids"]

        # Use the MT5 model to generate a summary
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=84,
            no_repeat_ngram_size=2,
            num_beams=4
        )[0]

        # Decode the generated token IDs to obtain the summary
        summary = tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
    except:
        # If an exception occurs, prompt the user to check the version of Transformers
        return 'Please check the version of Transformers'

    return summary


import tkinter as tk
from tkinter import ttk, scrolledtext
import jieba


###################################################
# UI implementation
###################################################

def summarize_text():
    # Summary generation via GUI
    input_text = input_text_widget.get("1.0", "end-1c")
    d = float(d_entry.get()) if d_entry.get() else 0.85
    top = int(top_entry.get()) if top_entry.get() else 3
    processing_method = processing_method_var.get()
    use_stopwords = use_stopwords_var.get()
    summary = summarize_text_rank(input_text, d=d, top=top, method=processing_method, use_stopwords=use_stopwords)
    output_text_widget.delete(1.0, tk.END)
    output_text_widget.insert(tk.END, summary)


def summarize_text_mt5():
    input_text = input_text_widget.get("1.0", "end-1c")
    summary_result = summary_mt5(input_text)
    output_text_widget_mt5.delete(1.0, tk.END)
    output_text_widget_mt5.insert(tk.END, summary_result)


# Create the main window
root = tk.Tk()
root.title("Chinese Text Automatic Summarization Tool")

# Adjust the style using the ttk module
style = ttk.Style()
style.configure('TFrame', padding=10)
style.configure('TButton', padding=(10, 5), font=('Helvetica', 10))
style.configure('TLabel', font=('Helvetica', 10))

# Create an input text box
input_label_frame = ttk.LabelFrame(root, text="Input Text")
input_label_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew", columnspan=2)  # 设置columnspan为2，使其横跨两列
input_text_widget = scrolledtext.ScrolledText(input_label_frame, wrap=tk.WORD, width=70, height=10)
input_text_widget.pack(pady=10, fill='both', expand=True)

# Create a summary length input box and set the default value to 100
frame1 = ttk.LabelFrame(root, text="TextRank Parameters")
frame1.grid(row=1, column=0, padx=10, pady=10, sticky="nsew", columnspan=2)  # Set columnspan to 2 to make it span across two columns

# Create a stop words checkbox
use_stopwords_var = tk.BooleanVar(root)
use_stopwords_var.set(True)  # Use stop words by default
use_stopwords_checkbutton = ttk.Checkbutton(frame1, text="Use Stop Words", variable=use_stopwords_var)
use_stopwords_checkbutton.grid(row=0, column=0, pady=5)

default_d = 0.85
d_label = ttk.Label(frame1, text=f"Damping Factor:")
d_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
d_entry = ttk.Entry(frame1, width=10)
d_entry.insert(0, str(default_d))
d_entry.grid(row=1, column=1, padx=2, pady=5)

default_top = 3
top_label = ttk.Label(frame1, text=f"Number of Summary Sentences:")
top_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
top_entry = ttk.Entry(frame1, width=10)
top_entry.insert(0, str(default_top))
top_entry.grid(row=2, column=1, padx=2, pady=5)

processing_method_var = tk.StringVar(root)
processing_method_var.set("Default Metric")  # Set the default option
processing_method_label = ttk.Label(frame1, text="Similarity Measure:")
processing_method_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")
processing_method_menu = ttk.Combobox(frame1, textvariable=processing_method_var, values=["Default Metric", "TF-IDF"], width=10)
processing_method_menu.grid(row=3, column=1, padx=2, pady=5)

# Create a button to trigger text summarization
summarize_button = ttk.Button(root, text="Generate TextRank Summary", command=summarize_text, style='TButton')
summarize_button.grid(row=2, column=0, padx=(10, 5), pady=10)  # Add horizontal and vertical padding

summarize_button_mt5 = ttk.Button(root, text="Generate MT5 Summary", command=summarize_text_mt5, style='TButton')
summarize_button_mt5.grid(row=2, column=1, padx=(5, 10), pady=10)  # Add horizontal and vertical padding

# Create an output text box
output_label_frame = ttk.LabelFrame(root, text="TextRank Output")
output_label_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nsew", columnspan=2)  # Set columnspan to 2 so that it spans two columns
output_text_widget = scrolledtext.ScrolledText(output_label_frame, wrap=tk.WORD, width=50, height=10)
output_text_widget.pack(pady=10, fill='both', expand=True)

output_label_frame_mt5 = ttk.LabelFrame(root, text="MT5 Output")
output_label_frame_mt5.grid(row=4, column=0, padx=10, pady=10, sticky="nsew", columnspan=2)  #  Set columnspan to 2 to span across two columns
output_text_widget_mt5 = scrolledtext.ScrolledText(output_label_frame_mt5, wrap=tk.WORD, width=50, height=10)
output_text_widget_mt5.pack(pady=10, fill='both', expand=True)

#  Set row and column weights so that the text box and label box can expand when the window is resized
for i in range(4):  #  Set the weight of all rows to 1
    root.grid_rowconfigure(i, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# Run the main loop
root.mainloop()
