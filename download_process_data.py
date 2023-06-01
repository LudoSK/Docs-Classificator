import requests
from bs4 import BeautifulSoup

# Reuters corpus link
base_url = "http://kdd.ics.uci.edu/databases/reuters21578/"

# You can choose how many files you want to use.
# For this example, I will use the first 22 files.
files = ['reut2-{:03d}.sgm'.format(i) for i in range(22)]

data = []

# For each file
for file in files:
    # Download the file from the website
    response = requests.get(base_url + file)

    # Use BeautifulSoup to parse the file's content
    soup = BeautifulSoup(response.content, 'html.parser')

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

training_data = [doc for doc in data if doc['split'] == 'TRAIN']
