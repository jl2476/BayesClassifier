import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List, Tuple

import os
from pathlib import Path
from typing import List, Tuple, Union


# The NLP(natural language processing) module

#
#  Comment this out when you've ran main.py at least once
#
""
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet') 
""

def load_data(data_dir: Union[str, os.PathLike], subset: str = 'training') -> Tuple[List[str], List[int]]:
    data_path = Path(data_dir) / subset
    if not data_path.is_dir():
        raise NotADirectoryError(f"'{data_path}' is not a valid directory")

    emails = []
    labels = []
    
    for label, class_label in [('spam', 1), ('ham', 0)]:
        class_path = data_path / label
        if not class_path.is_dir():
            raise NotADirectoryError(f"'{class_path}' is not a valid directory")
        
        email_files = [f for f in class_path.iterdir() if f.is_file()]
        if not email_files:
            print(f"Warning: No files found in '{class_path}'")
            continue
        
        for email_file in email_files:
            try:
                with email_file.open('r', encoding='utf-8', errors='ignore') as f:
                    email_content = f.read()
                processed_email = preprocess_email(email_content)
                emails.append(processed_email)
                labels.append(class_label)
            except Exception as e:
                print(f"Error processing file {email_file}: {str(e)}")
    
    if not emails:
        raise ValueError(f"No valid email files found in '{data_path}'")
    
    return emails, labels




def preprocess_email(email_content: str) -> str:
    # Convert to lowercase
    email_content = email_content.lower()
    
    # Remove special characters and digits
    email_content = re.sub(r'[^a-zA-Z\s]', '', email_content)
    
    # Tokenize
    tokens = nltk.word_tokenize(email_content)
    
    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    processed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    # Join tokens back into a string
    return ' '.join(processed_tokens)

