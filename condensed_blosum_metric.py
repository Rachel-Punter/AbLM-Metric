import numpy as np

'''This file stores all values in the BLOSUM scoring matrix in the form of a dictionary with all values being between 0 and 1. 
These new values were calculated using the linear equation new_score = ((1 / (max_possible_value - min-POSSIBLE_VALUE)) * (blosum_score - max_possible_value)) + 1
Each max and min possible value is relative to the actual amino acid in a given sequence. The blosum_score should be the value in BLOSUM
for the actual and inferred amino acids.

BLOSUM was created by Heinkoff & Heinkoff :) '''

new_blosum62 = {
    "A": {
        "A": 1.0000,
        "R": 0.2857,
        "N": 0.1429,
        "D": 0.1429,
        "C": 0.4286,
        "Q": 0.2857,
        "E": 0.2857,
        "G": 0.4286,
        "H": 0.1429,
        "I": 0.2857,
        "L": 0.2857,
        "K": 0.2857,
        "M": 0.2857,
        "F": 0.1429,
        "P": 0.2857,
        "S": 0.5714,
        "T": 0.4286,
        "W": 0.0000,
        "Y": 0.1429,
        "V": 0.4286
    },
    "R": {
        "A": 0.2500,
        "R": 1.0000,
        "N": 0.3750,
        "D": 0.1250,
        "C": 0.0000,
        "Q": 0.5000,
        "E": 0.3750,
        "G": 0.1250,
        "H": 0.3750,
        "I": 0.0000,
        "L": 0.1250,
        "K": 0.6250,
        "M": 0.2500,
        "F": 0.0000,
        "P": 0.1250,
        "S": 0.2500,
        "T": 0.2500,
        "W": 0.0000,
        "Y": 0.1250,
        "V": 0.0000
    },
    "N": {
        "A": 0.2000,
        "R": 0.4000,
        "N": 1.0000,
        "D": 0.5000,
        "C": 0.1000,
        "Q": 0.4000,
        "E": 0.4000,
        "G": 0.4000,
        "H": 0.5000,
        "I": 0.1000,
        "L": 0.1000,
        "K": 0.4000,
        "M": 0.2000,
        "F": 0.1000,
        "P": 0.2000,
        "S": 0.5000,
        "T": 0.4000,
        "W": 0.0000,
        "Y": 0.2000,
        "V": 0.1000
    },
    "D": {
        "A": 0.2000,
        "R": 0.2000,
        "N": 0.5000,
        "D": 1.0000,
        "C": 0.1000,
        "Q": 0.4000,
        "E": 0.6000,
        "G": 0.3000,
        "H": 0.3000,
        "I": 0.1000,
        "L": 0.0000,
        "K": 0.3000,
        "M": 0.1000,
        "F": 0.1000,
        "P": 0.3000,
        "S": 0.4000,
        "T": 0.3000,
        "W": 0.0000,
        "Y": 0.1000,
        "V": 0.1000
    },
    "C": {
        "A": 0.3077,
        "R": 0.0769,
        "N": 0.0769,
        "D": 0.0769,
        "C": 1.0000,
        "Q": 0.0769,
        "E": 0.0000,
        "G": 0.0769,
        "H": 0.0769,
        "I": 0.2308,
        "L": 0.2308,
        "K": 0.0769,
        "M": 0.2308,
        "F": 0.1538,
        "P": 0.0769,
        "S": 0.2308,
        "T": 0.2308,
        "W": 0.1538,
        "Y": 0.1538,
        "V": 0.2308
    },
    "Q": {
        "A": 0.2500,
        "R": 0.5000,
        "N": 0.3750,
        "D": 0.3750,
        "C": 0.0000,
        "Q": 1.0000,
        "E": 0.6250,
        "G": 0.1250,
        "H": 0.3750,
        "I": 0.0000,
        "L": 0.1250,
        "K": 0.5000,
        "M": 0.3750,
        "F": 0.0000,
        "P": 0.2500,
        "S": 0.3750,
        "T": 0.2500,
        "W": 0.1250,
        "Y": 0.2500,
        "V": 0.1250
    },
    "E": {
        "A": 0.3333,
        "R": 0.4444,
        "N": 0.4444,
        "D": 0.6667,
        "C": 0.0000,
        "Q": 0.6667,
        "E": 1.0000,
        "G": 0.2222,
        "H": 0.4444,
        "I": 0.1111,
        "L": 0.1111,
        "K": 0.5556,
        "M": 0.2222,
        "F": 0.1111,
        "P": 0.3333,
        "S": 0.4444,
        "T": 0.3333,
        "W": 0.1111,
        "Y": 0.2222,
        "V": 0.2222
    },
    "G": {
        "A": 0.4000,
        "R": 0.2000,
        "N": 0.4000,
        "D": 0.3000,
        "C": 0.1000,
        "Q": 0.2000,
        "E": 0.2000,
        "G": 1.0000,
        "H": 0.2000,
        "I": 0.0000,
        "L": 0.0000,
        "K": 0.2000,
        "M": 0.1000,
        "F": 0.1000,
        "P": 0.2000,
        "S": 0.4000,
        "T": 0.2000,
        "W": 0.2000,
        "Y": 0.1000,
        "V": 0.1000
    },
    "H": {
        "A": 0.0909,
        "R": 0.2727,
        "N": 0.3636,
        "D": 0.1818,
        "C": 0.0000,
        "Q": 0.2727,
        "E": 0.2727,
        "G": 0.0909,
        "H": 1.0000,
        "I": 0.0000,
        "L": 0.0000,
        "K": 0.1818,
        "M": 0.0909,
        "F": 0.1818,
        "P": 0.0909,
        "S": 0.1818,
        "T": 0.0909,
        "W": 0.0909,
        "Y": 0.4545,
        "V": 0.0000
    },
    "I": {
        "A": 0.3750,
        "R": 0.1250,
        "N": 0.1250,
        "D": 0.1250,
        "C": 0.3750,
        "Q": 0.1250,
        "E": 0.1250,
        "G": 0.0000,
        "H": 0.1250,
        "I": 1.0000,
        "L": 0.7500,
        "K": 0.1250,
        "M": 0.6250,
        "F": 0.5000,
        "P": 0.1250,
        "S": 0.2500,
        "T": 0.3750,
        "W": 0.1250,
        "Y": 0.3750,
        "V": 0.8750
    },
    "L": {
        "A": 0.3750,
        "R": 0.2500,
        "N": 0.1250,
        "D": 0.0000,
        "C": 0.3750,
        "Q": 0.2500,
        "E": 0.1250,
        "G": 0.0000,
        "H": 0.1250,
        "I": 0.7500,
        "L": 1.0000,
        "K": 0.2500,
        "M": 0.7500,
        "F": 0.5000,
        "P": 0.1250,
        "S": 0.2500,
        "T": 0.3750,
        "W": 0.2500,
        "Y": 0.3750,
        "V": 0.6250
    },
    "K": {
        "A": 0.2500,
        "R": 0.6250,
        "N": 0.3750,
        "D": 0.2500,
        "C": 0.0000,
        "Q": 0.5000,
        "E": 0.5000,
        "G": 0.1250,
        "H": 0.2500,
        "I": 0.0000,
        "L": 0.1250,
        "K": 1.0000,
        "M": 0.2500,
        "F": 0.0000,
        "P": 0.2500,
        "S": 0.3750,
        "T": 0.2500,
        "W": 0.0000,
        "Y": 0.1250,
        "V": 0.1250
    },
    "M": {
        "A": 0.2500,
        "R": 0.2500,
        "N": 0.1250,
        "D": 0.0000,
        "C": 0.2500,
        "Q": 0.3750,
        "E": 0.1250,
        "G": 0.0000,
        "H": 0.1250,
        "I": 0.5000,
        "L": 0.6250,
        "K": 0.2500,
        "M": 1.0000,
        "F": 0.3750,
        "P": 0.1250,
        "S": 0.2500,
        "T": 0.2500,
        "W": 0.2500,
        "Y": 0.2500,
        "V": 0.5000
    },
    "F": {
        "A": 0.2000,
        "R": 0.1000,
        "N": 0.1000,
        "D": 0.1000,
        "C": 0.2000,
        "Q": 0.1000,
        "E": 0.1000,
        "G": 0.1000,
        "H": 0.3000,
        "I": 0.4000,
        "L": 0.4000,
        "K": 0.1000,
        "M": 0.4000,
        "F": 1.0000,
        "P": 0.0000,
        "S": 0.2000,
        "T": 0.2000,
        "W": 0.5000,
        "Y": 0.7000,
        "V": 0.3000
    },
    "P": {
        "A": 0.2727,
        "R": 0.1818,
        "N": 0.1818,
        "D": 0.2727,
        "C": 0.0909,
        "Q": 0.2727,
        "E": 0.2727,
        "G": 0.1818,
        "H": 0.1818,
        "I": 0.0909,
        "L": 0.0909,
        "K": 0.2727,
        "M": 0.1818,
        "F": 0.0000,
        "P": 1.0000,
        "S": 0.2727,
        "T": 0.2727,
        "W": 0.0000,
        "Y": 0.0909,
        "V": 0.1818
    },
    "S": {
        "A": 0.5714,
        "R": 0.2857,
        "N": 0.5714,
        "D": 0.4286,
        "C": 0.2857,
        "Q": 0.4286,
        "E": 0.4286,
        "G": 0.4286,
        "H": 0.2857,
        "I": 0.1429,
        "L": 0.1429,
        "K": 0.4286,
        "M": 0.2857,
        "F": 0.1429,
        "P": 0.2857,
        "S": 1.0000,
        "T": 0.5714,
        "W": 0.0000,
        "Y": 0.1429,
        "V": 0.1429
    },
    "T": {
        "A": 0.2857,
        "R": 0.1429,
        "N": 0.2857,
        "D": 0.1429,
        "C": 0.1429,
        "Q": 0.1429,
        "E": 0.1429,
        "G": 0.0000,
        "H": 0.0000,
        "I": 0.1429,
        "L": 0.1429,
        "K": 0.1429,
        "M": 0.1429,
        "F": 0.0000,
        "P": 0.1429,
        "S": 0.4286,
        "T": 1.0000,
        "W": 0.0000,
        "Y": 0.0000,
        "V": 0.2857
    },
    "W": {
        "A": 0.0667,
        "R": 0.0667,
        "N": 0.0000,
        "D": 0.0000,
        "C": 0.1333,
        "Q": 0.1333,
        "E": 0.0667,
        "G": 0.1333,
        "H": 0.1333,
        "I": 0.0667,
        "L": 0.1333,
        "K": 0.0667,
        "M": 0.2000,
        "F": 0.3333,
        "P": 0.0000,
        "S": 0.0667,
        "T": 0.1333,
        "W": 1.0000,
        "Y": 0.4000,
        "V": 0.0667
    },
    "Y": {
        "A": 0.1000,
        "R": 0.1000,
        "N": 0.1000,
        "D": 0.0000,
        "C": 0.1000,
        "Q": 0.2000,
        "E": 0.1000,
        "G": 0.0000,
        "H": 0.5000,
        "I": 0.2000,
        "L": 0.2000,
        "K": 0.1000,
        "M": 0.2000,
        "F": 0.6000,
        "P": 0.0000,
        "S": 0.1000,
        "T": 0.1000,
        "W": 0.5000,
        "Y": 1.0000,
        "V": 0.2000
    },
    "V": {
        "A": 0.4286,
        "R": 0.0000,
        "N": 0.0000,
        "D": 0.0000,
        "C": 0.2857,
        "Q": 0.1429,
        "E": 0.1429,
        "G": 0.0000,
        "H": 0.0000,
        "I": 0.8571,
        "L": 0.5714,
        "K": 0.1429,
        "M": 0.5714,
        "F": 0.2857,
        "P": 0.1429,
        "S": 0.1429,
        "T": 0.4286,
        "W": 0.0000,
        "Y": 0.2857,
        "V": 1.0000
    }
}


    
#finds the modified score for the inference
def find_all_scores(actual_seq, inferred_seq, masked_index):
    bloss = new_blosum62[actual_seq[masked_index]][inferred_seq[masked_index]] 
    return bloss

def new_scores(actual_seq, inferred_seq, region):

    seq_scores = []
    frw_scores = []
    cdr1_scores = []
    cdr2_scores = []
    cdr3_scores = []

    
    for i in range (0, len(actual_seq)):

        all_scores = find_all_scores(actual_seq, inferred_seq, i)
        
        seq_scores.append(all_scores)

        if (region[i] == "0"):
            frw_scores.append(all_scores) 

        elif (region[i] == "1"):
            cdr1_scores.append(all_scores)

        elif (region[i] == "2"):
            cdr2_scores.append(all_scores)

        elif (region[i] == "3"):
            cdr3_scores.append(all_scores)

        else: 
            print("The region here is incorrect", region[i])

    # finds the average and medians for the entire sequence 
    average_score = np.mean(seq_scores)
    average_frw = np.mean(frw_scores)
    average_cdr1 = np.mean(cdr1_scores)
    average_cdr2 = np.mean(cdr2_scores)
    average_cdr3 = np.mean(cdr3_scores)

    median_total = np.median(seq_scores)
    median_frw = np.median(frw_scores)
    median_cdr1 = np.median(cdr1_scores)
    median_cdr2 = np.median(cdr2_scores)
    median_cdr3 = np.median(cdr3_scores)

    return average_score, average_frw, average_cdr1, average_cdr2, average_cdr3, median_total, median_frw, median_cdr1, median_cdr2, median_cdr3

'''
The main function is used as a test to see if the code is working. 
The output should be:

The following are the averages for sequence:  QQQADKKLLLKKKQWRA
The average score for the entire run is:  0.27311176470588233
The average score for the framework region of the entire run is:  0.2366125
The average score for CDR1 of the entire run is:  0.4083333333333334
The average score for CDR2 of the entire run is:  0.25
The average score for CDR3 of the entire run is:  0.25833333333333336
'''
def main():

    sequencelist = ["QQQADKKLLLKKKQWRA"]
    inferredlist = ["RYTACGLKKKAAYYYDD"]
    # inferredlist = ["QQQADKKLLLKKKQWRA"]

    region = ["00011102220033300"]

    
    sum_run_avg = 0
    sum_run_frw_avg = 0
    sum_run_cdr1_avg = 0
    sum_run_cdr2_avg = 0
    sum_run_cdr3_avg = 0

    for i in range (0, len(sequencelist)):
        find = new_scores(sequencelist[i], inferredlist[i], region[i])

        sum_run_avg += find[0]
        sum_run_frw_avg += find[1]
        sum_run_cdr1_avg += find[2]
        sum_run_cdr2_avg += find[3]
        sum_run_cdr3_avg += find[4]
    
    run_avg = sum_run_avg/len(sequencelist)
    run_frw_avg = sum_run_frw_avg/len(sequencelist)
    run_cdr1_avg = sum_run_cdr1_avg/len(sequencelist)
    run_cdr2_avg = sum_run_cdr2_avg/len(sequencelist)
    run_cdr3_avg = sum_run_cdr3_avg/len(sequencelist)

    print("The following are the averages for sequence: ", sequencelist[0])
    print("The average score for the entire run is: ", run_avg)
    print("The average score for the framework region of the entire run is: ", run_frw_avg)
    print("The average score for CDR1 of the entire run is: ", run_cdr1_avg)
    print("The average score for CDR2 of the entire run is: ", run_cdr2_avg)
    print("The average score for CDR3 of the entire run is: ", run_cdr3_avg)





if __name__ == "__main__":
    main()