#!/usr/bin/env python3

import sys
import os
import argparse
import random


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--input", help="Input file")
  parser.add_argument("-o", "--outdir", default="./data", help="Output dir")
  parser.add_argument("-r", "--retention", type=float, default=0.5, help="Space retention rate of the training data")
  cmd_args = parser.parse_args(sys.argv[1:])

  src_path = os.path.join(cmd_args.outdir, "data.src")
  tgt_path = os.path.join(cmd_args.outdir, "data.tgt")
  with open(cmd_args.input) as infile:
    with open(src_path, "wt", encoding="utf8") as src_file:
      with open(tgt_path, "wt", encoding="utf8") as tgt_file:
        for line_no, line in enumerate(infile):
          # Generate train file
          omit = False
          for c in line:
            if c == " " and random.random() > cmd_args.retention:
              omit = True
            else:
              src_file.write(c)
              tgt_file.write(c if c == "\n" else "0" if omit else "1")
              omit = False


if __name__ == "__main__":
  main()

