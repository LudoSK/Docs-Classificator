import os
from bs4 import BeautifulSoup

data = []

# For each file
for file in os.listdir('reuters_test'):
    if file.endswith(".sgm"):
        try:
            with open(os.path.join('reuters_test', file), 'r', encoding='utf-8') as f:
                content = f.read()
        except:
            continue
        # Use BeautifulSoup to parse the file's content
        soup = BeautifulSoup(content, 'html.parser')

        # For each document in the file
        for doc in soup.find_all('reuters'):
            # Check if the document is for training or testing
            lewis_split = doc.get('lewissplit')

            # Get the first class of the document
            topics = doc.find('topics').find_all('d')
            if topics:
                first_class = topics[0].get_text()

                # Add the document's information to the list
                data.append({
                    'text': doc.find('text').get_text(),
                    'class': first_class,
                    'split': lewis_split
                })

# At this point, 'data' is a list of dictionaries, each dictionary containing
# a document's text, its class and whether it's meant for training or testing.

