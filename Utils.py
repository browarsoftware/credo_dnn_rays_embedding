"""
Author: Tomasz Hachaj, 2021
Department of Signal Processing and Pattern Recognition
Institute of Computer Science in Pedagogical University of Krakow, Poland
https://sppr.up.krakow.pl/hachaj/
Data source:
https://credo.nkg-mn.com/hits.html
"""

import os

def create_dir(my_path):
    try:
        os.mkdir(my_path)
    except OSError:
        a = 0
        a = a + 1
    else:
        a = 0
        a = a + 1