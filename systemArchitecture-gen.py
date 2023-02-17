import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Define the set of requirements
requirements = [
    "The system shall provide data on atmospheric conditions.",
    "The system shall include a camera to capture images of Earth.",
    "The system shall have a ground station to communicate with the satellite.",
    "The system shall provide real-time data to end-users.",
    "The system shall have a data processing center to process the received data."
]

# Define the list of keywords for the system architecture diagram
keywords = ["system", "payload", "ground station", "communication", "data processing center", "user community"]

# Define the stop words for text processing
stop_words = set(stopwords.words("english"))

# Define the lemmatizer for text processing
lemmatizer = WordNetLemmatizer()

# Initialize the dictionary for the system architecture diagram
architecture = {keyword: [] for keyword in keywords}

# Process each requirement
for requirement in requirements:
    # Tokenize the requirement into words
    words = word_tokenize(requirement.lower())
    
    # Remove the stop words from the list of words
    words = [word for word in words if word not in stop_words]
    
    # Lemmatize the words
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Extract the noun phrases from the words
    noun_phrases = []
    for chunk in nltk.ne_chunk(nltk.pos_tag(words)):
        if hasattr(chunk, "label") and chunk.label() == "PERSON":
            continue
        subtree_words = [word for word, pos in chunk.leaves()]
        if len(subtree_words) > 1:
            noun_phrase = " ".join(subtree_words)
            noun_phrases.append(noun_phrase)
    
    # Add the noun phrases to the appropriate keyword in the system architecture diagram
    for noun_phrase in noun_phrases:
        for keyword in keywords:
            if keyword in noun_phrase:
                architecture[keyword].append(noun_phrase)

# Print the system architecture diagram
for keyword in keywords:
    print(keyword.upper())
    for item in set(architecture[keyword]):
        print("-", item)
    print()
