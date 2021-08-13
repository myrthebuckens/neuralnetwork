import pandas as pd
import argparse

def read_file(path):
    """
    Function to read in a path to data file and return it as a pandas dataframe
    :param path: path to data file
    :return: pandas dataframe with loaded data
    """
    with open(path) as infile:

        # for some reason, the conll files I downloaded are saved with space as delimiter instead of '\t', but reading
        # in the data as pandas data frame and saving later with delimiter '\t' does resolve this

        data = pd.read_csv(infile, delimiter = ' ', names=["Token", "POS", "Phrase", "NER"])
    print(data)
    return data

def simplify_labels(data):
    """
    Function to convert labels with BIO scheme annotation to labels without BIO scheme annotation
    :param path: path to data file
    :return: pandas dataframe with simplified labels
    """
    # indicating which column of the dataframe contains the labels for NER
    labels = data.iloc[: , -1]

    # list for the converted labels
    converted_labels = []

    print(set(labels))

    for label in labels:

        # the empty lines are read as nan so only check the lines with labels
        if type(label) == str and (label.startswith('B-') or label.startswith('I-')):

            # only take the PER, ORG, LOC or MISC after first two characters of B- or I-
            new_label = label[2::]
            converted_labels.append(new_label)

        else:
            # for O label
            new_label = 'O'
            converted_labels.append(new_label)

    # create pandas df from the converted labels
    new_labels = pd.DataFrame(converted_labels)

    # initialize original dataframe but without the labels
    df = data.iloc[: , 0:-1]

    # concatenate both dataframes, last column will be the converted labels
    new_df = pd.concat([df, new_labels], axis =1)

    return new_df


def main():
    # adding arguments to functions
    parser = argparse.ArgumentParser(description='This script preprocesses conll files by removing the BIO scheme and saving as conll files with "\t" delimiter')

    parser.add_argument('path',
                        help='file path to conll file with data. Recommended path: "../data/conll2003/{filename}"')

    args = parser.parse_args()

    # running functions
    data = read_file(args.path)
    new_df = simplify_labels(data)

    # saving new dataframes
    filename = args.path.replace(".conll", "_converted.conll")
    new_df.to_csv(filename, sep='\t', index=False)


if __name__ == '__main__':
    main()
