# Problem Set 4A
# Name: <your name here>
# Collaborators:
# Time Spent: x:xx

def get_permutations(sequence):
    '''
    Enumerate all permutations of a given string

    sequence (string): an arbitrary string to permute. Assume that it is a
    non-empty string.  

    You MUST use recursion for this part. Non-recursive solutions will not be
    accepted.

    Returns: a list of all permutations of sequence

    Example:
    >>> get_permutations('abc')
    ['abc', 'acb', 'bac', 'bca', 'cab', 'cba']

    Note: depending on your implementation, you may return the permutations in
    a different order than what is listed here.
    '''
    
    #first, turn the string into a list with a delimiter of spaces,
    lists = sequence.split(sep= ' ')
    #The first word has the unparsed letters, the rest of the words are the various permutations
    #this checks to see how long the first word is, if it is equal to one, that means that there is either 1 letter left to iterate over, or 
    #that the word only has 1 letter in it 
    parseletter = sequence[0]
    templist = []
    returnedsequence = ''
    
    if len(lists[0]) == 1: 
        #if it is just one letter long, it just returns the list. 
        if len(lists) == 1:
            return lists
        for word in lists [1:]:
            count = 0
            #for each word, this adds the parsing letter into each unique place it could be and 
            #adds it to the end of the tempwords list. 
            while count < len(word) + 1:
                templist.append(word[0:count] + parseletter + word[count:])
                count += 1
        return templist
    else:
        #the following code evaluates all of the sequences in the given list, and inserts parsing letter in all possible positions. 
        templist.append (lists [0][1:])
        if len(lists) == 1: 
            templist.append (lists[0][0])
        for word in lists [1:]:
            count = 0
            #for each word, this adds the parsing letter into each unique place it could be and 
            #adds it to the end of the tempwords list. 
            while count < len(word) + 1:
                templist.append(word[0:count] + parseletter + word[count:])
                count += 1
        #then we need to turn tempwords back into a string so I can feed it back into the function again
        print(f'templist: {templist}')
        for segments in templist: 
            returnedsequence += segments + ' '
        x = returnedsequence.strip()
        print(f'returned sequence: {returnedsequence}')
#        print ('+'+ x + '+')
        return get_permutations(x)
        


if __name__ == '__main__':
#    #EXAMPLE
#    example_input = 'abc'
#    print('Input:', example_input)
#    print('Expected Output:', ['abc', 'acb', 'bac', 'bca', 'cab', 'cba'])
#    print('Actual Output:', get_permutations(example_input))
    
#    # Put three example test cases here (for your sanity, limit your inputs
#    to be three characters or fewer as you will have n! permutations for a
#    sequence of length n)

#    print('Input:', 'abc')
#    print('Expected Output:', ['abc', 'acb', 'bac', 'bca', 'cab', 'cba'])
    input = 'abc'
    print(f'Actual Output:{get_permutations(input)}')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
