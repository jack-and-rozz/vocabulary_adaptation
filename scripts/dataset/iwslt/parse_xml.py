import xml.etree.ElementTree as ET
import os, sys, argparse

def main(args):
    # print(args.input_file, file=sys.stderr)
    tree = ET.parse(args.input_file)
    root = tree.getroot()

    for doc in list(root)[0]:
        for seg in doc:
            if seg.tag == 'seg' and seg.text.strip():
                print(seg.text.strip())

if __name__ == "__main__":
    desc = ''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('input_file', type=str)
    args = parser.parse_args()
    main(args)
