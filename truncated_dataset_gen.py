#!/usr/bin/env python
# coding: utf-8

# # Dataset Generation
# Allie Stanton
# 
# In this module I will read in the fasta files of HA sequences, truncate them to 150bp, and convert them to one-hot encoding readable by TensorFlow.
# 
# The input data consists of 18 fasta files each containing the complete and unique HA segments from the NCBI Influenza Virus Database. There is one file for each of the 18 subtypes. The files were retrieved from NCBI on April 10, 2019. The search parameters were as follows:
# 
# Basic filters:
# - Sequence type: nucleotide
# - Type: A
# - Host: any
# - Country/Region: any
# - Segment: 4 (HA)
# - Subtype: 
# 	 - H: (1-18, different for each file)
# 	 - N: any     
# - Full-length only
# 
# Additional filters:
# - Required segments: full-length only, 4 (HA)
# - Include pandemic (H1N1) viruses
# - Exclude lab strains
# - Include lineage defining strains
# - Include the FLU project
# - Include vaccine strains
# - Include mixed subtype
# - Collapse identical sequences

from Bio import SeqIO
import numpy as np
import os
import random
import h5py

# load in files
seqdir = os.path.join(os.getcwd(), "flu_seqs")
files = []
for file in os.listdir(seqdir):
    if file.endswith('.fa'):
        filename = os.fsdecode(file)
        filepath = os.path.join(seqdir, file)
        files.append(filepath)

def truncate_sequence(sequence, length):
############################################################################                                                   
#                                                                                                                              
# Truncates a DNA sequence.                                                                                                             
#                                                                                                                              
# Inputs:                                                                                                                      
#   -sequence: a DNA sequence of type str.                                                                                     
#   -length: desired length of truncated sequence
# Outputs:                                                                                                                     
#   -truncated_seq: a DNA sequence of type str and specified length                                                   
#                                                                                                                              
############################################################################    

    start = random.randint(0, len(sequence) - length)
    truncated_seq = sequence[start : start + length]

    return truncated_seq

def sequence_to_one_hot(sequence, chars = 'ACGTNRYSWKMBDHV'):
############################################################################
#
# Converts a DNA sequence to one-hot encoding, setting all ambiguous bases
#   to zeros.
#
# Inputs:
#   -sequence: a DNA sequence of type str.
#   -chars: a string containing all possible bases, with ACGT as the first 
#      four.
#
# Outputs:
#   -onehot_encoded: a list the same length as the input sequence where each
#      item is a list of length 4 corresponding to the one-hot encoding of 
#      the base at that position.
#
############################################################################
    
    seqlen = len(sequence)
    char_to_int = dict((c, i) for i, c in enumerate(chars))

    onehot_encoded = []
        
    integer_encoded = [char_to_int[base] for base in sequence]
    for value in integer_encoded:
        letter = [0 for _ in range(4)]
        
        # if the base is ambiguous, leave the one-hot as zeros
        if value < 4:
            letter[value] = 1
        onehot_encoded.append(letter)
    
    return onehot_encoded


def parse_and_encode(seqdir, num_subtypes):
############################################################################
#
# Reads in each record in each fasta file and converts to one-hot encoding,
#   then pairs the sequence and label (subtype) and shuffles the entire list.
#
# Inputs:
#   -seqdir: the directory that the fasta files are in
#   -num_subtypes: the number of fasta files
#
# Outputs:
#   -pairs_list: a list of tuples where each tuple has the one-hot encoded
#      sequence as its first element and the one-hot encoded label as its
#      second element.
#
############################################################################
    
    pairs_list_nested = []
    for i in range(1, num_subtypes + 1):
        print("Reading in subtype", i)
        file = os.path.join(seqdir, 'H' + str(i) + '.fa')
        
        label_onehot = [0 for _ in range(num_subtypes)]
        label_onehot[i - 1] = 1
    
        subtype_pairs_list = []
        for record in SeqIO.parse(file, 'fasta'):
            seq = str(record.seq)
            
            j = 0
            while j < 20:
                truncated_seq = truncate_sequence(seq, 150)
                seq_onehot = sequence_to_one_hot(truncated_seq)
                pair = (seq_onehot, label_onehot)
                subtype_pairs_list.append(pair)
                j += 1
        pairs_list_nested.append(subtype_pairs_list)
    
    pairs_list = [seq for subtype in pairs_list_nested for seq in subtype]
    random.shuffle(pairs_list)
    
    return pairs_list


seqdir = os.path.join(os.getcwd(), "flu_seqs")

seqs_by_subtype = parse_and_encode(seqdir, 16)


def pad_seqs(paired_seqs_and_labels, width):
############################################################################
# 
# Pads the sequences with zeros until they are all the same length, and then
#   converts the padded sequences to an array of shape (num_seqs, length, 4)
#   and converts the labels to an array of shape (num_seqs, num_subtypes).
#
# Inputs:
#   -paired_seqs_and_labels: a list of length (num_seqs) where each element
#      is a list or tuple where the first item is the one-hot encoded  
#      sequence (type: list) and the second item is the one-hot encoded
#      label.
#   -width: the length of the longest sequence. All sequences will be padded
#      to this length.
#
# Outputs:
#   -padded_seqs: an array of shape (num_seqs, length, 4) containing the
#      one-hot encoded sequences.
#   -labels: an array of shape (num_seqs, num_subtypes) containing the one-
#      hot encoded labels.
#
############################################################################

    padded_seqs_list = []
    labels_list = []
    for pair in paired_seqs_and_labels:
        if len(pair[0]) < width:
            seq = pair[0] + [[0, 0, 0, 0]] * (width - len(pair[0]))
        else: 
            seq = pair[0]
        padded_seqs_list.append([seq])
    
        labels_list.append(pair[1])
        
    padded_seqs = np.array(padded_seqs_list)
    labels = np.array(labels_list)
    
    return padded_seqs, labels


padded, labels = pad_seqs(seqs_by_subtype, 150)


print(np.shape(padded))
print(np.shape(labels))


# save arrays
train = h5py.File('truncated_flu_train.h5', 'w')
train.create_dataset('features', data = padded[0:745000])
train.create_dataset('labels', data = labels[0:745000])
train.close()

val = h5py.File('truncated_flu_val.h5', 'w')
val.create_dataset('features', data = padded[745000:992220])
val.create_dataset('labels', data = labels[745000:992220])
val.close()

test = h5py.File('truncated_flu_test.h5', 'w')
test.create_dataset('features', data = padded[992220:])
test.create_dataset('labels', data = labels[992220:])
test.close()




