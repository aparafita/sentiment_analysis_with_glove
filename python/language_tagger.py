# Author: Álvaro Parafita (parafita.alvaro@gmail.com)

# Usage: 
# python3 python/language_tagger.py data/reviews.json.gz data/language_tagging.json
# Type in single keystrokes, the ones enclosed between []
# Type b (back) to go back in you previous assignments
# Type q to exit the process

import sys
import json
import gzip

from random import shuffle
from getch import getch

if __name__ == '__main__':
    reviews_filename = sys.argv[1] # compulsory
    output_filename = sys.argv[2] # compulsory

    with gzip.open(reviews_filename, 'rt') as f:
        reviews = json.load(f)

    reviews = {
        '|'.join((eatery_id, review['review_id'])): review['review_text']

        for eatery_id, d in filter(lambda pair: bool(pair[1]), reviews.items())
        for review in d.get('reviews') or []
        if review.get('review_id') and review.get('review_text').strip()
    }

    try:
        with open(output_filename) as f:
            tagged = json.load(f)
    except FileNotFoundError:
        tagged = {}

    op_dict = {
        'e': 'english',
        's': 'spanish',
        'f': 'french',
        'i': 'italian',
        'o': 'other',
        'n': 'null',
        'b': 'back',
        'q': 'quit',
    }


    to_be_tagged = list(set(reviews) - set(tagged))
    shuffle(to_be_tagged)

    back = []
    while to_be_tagged:
        k = to_be_tagged.pop(0)

        print('%.6d/%.6d' % (len(tagged), len(reviews)), k)
        print(reviews[k])
        
        op = getch(
            '-' * 20 + 
            '\n' + 
            '[e]nglish/[s]panish/[f]rench/[i]talian/' +
            '[o]ther/[n]ull/[b]ack/[q]uit > '
        ).strip().lower()

        if op in op_dict:
            op = op_dict[op]

            if op == 'quit':
                break

            elif op == 'back':
                to_be_tagged.insert(0, k) # repeat actual k
                
                if back:
                    k = back.pop()
                    del tagged[k]

                    to_be_tagged.insert(0, k) # insert previous k

                    continue
                else:
                    print('No more entries back')
                    print()
                    continue
                    
            else:
                tagged[k] = op
        else:
            print('Incorrect option; entry considered as NULL')
            tagged[k] = None

        back.append(k)

        print()

    # Save
    with open(output_filename, 'w') as f:
        json.dump(tagged, f, indent=2)