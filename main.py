from shutil import copy2

import json
import os
import random
import sys
import tarfile
import urllib.request


random.seed=(72019)
DIALOGUE_ACT_FILE = "acts.json"

def getDialogueActTags(filename):
    # Load a dictionary of speech acts from the designated json file
    
    with open(filename, "r") as j:
        d = json.load(j)
    return d


def getListOfFiles(dirName):
    # create a lit of file and sub directories
    # names in the given directory
    listOfFiles = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the enteries
    for entry in listOfFiles:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory, then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles


def makeTrainAndTest():
    # Take the contents of the swbd folder and randomly assign
    # 90% to the training set and 10% to the test set
    
    try:
        os.mkdir("./train")
        os.mkdir("./test")
    except:
        pass
    for f in getListOfFiles("swbd"):
        roll = random.randrange(0,10)
        directory = "./train"
        if roll < 1:
            directory = "./test"
        copy2(f, directory)

def processFile(f):
    # Take an input file (Conversation from the SWBD-DAMSL corpus)
    # And parse it into a dictionairy of the form:
    # output[uttnum][speaker,dialogue_act,pair_part,words]
    output = {}
    file_object = open(f, "r")
    tag_dict = getDialogueActTags(DIALOGUE_ACT_FILE)
    prev_tag = None
    for line in file_object.readlines():
        if " utt" in line:
            try:
                metadata, words = line.split(':')
                dialogue_act, speaker_uttnum, pair_part = metadata.split()
                speaker, uttnum_str = speaker_uttnum.split('.')
                uttnum = int(uttnum_str)
                swbd_dialogue_act = update_tag(dialogue_act, tag_dict, prev_tag)
                prev_tag = swbd_dialogue_act
                output[uttnum] = {}
                output[uttnum]["dialogue_act"] = swbd_dialogue_act
                output[uttnum]["speaker"] = speaker
                output[uttnum]["pair_part"] = pair_part
                output[uttnum]["words"] = words
            except ValueError:
                pass
    return output


def update_tag(act, tag_dict, prev_tag):
    # Update tags from the DAMSL set to the SWBD-DAMSL set
    # See https://web.stanford.edu/~jurafsky/ws97/manual.august1.html
    # Section 1d for details

    ret_tag = None
    if act in tag_dict.keys():
        ret_tag = tag_dict[act]
    elif "+" in act:
        ret_tag = prev_tag
    elif "^e" in act:
        ret_tag = prev_tag

    # TODO: Need to figure out what these represent
    # I think some of them represent differing opinions from
    # The anotaters, which means there could be several interpretations
    error_chars = ["(","^","@","*",",",";"]
    for char in error_chars:
        if act.split(char)[0] in tag_dict:
            ret_tag = tag_dict[act.split(char)[0]]

    if "^" in act:
        try:
            act.split("^")[2]
            ret_tag = tag_dict["^" + act.split("^")[1]]
        except IndexError:
            pass
        except KeyError:
            return update_tag("^" + act.split("^")[1], tag_dict, prev_tag)

    if ret_tag == None:
        try:
            ret_tag = tag_dict[act[:2]]
        except KeyError:
            print(act)
    return ret_tag


def processConversations(conversationSet):
    # Get the list of conversations, and turn it into a list
    # Containing the info of each conversation per entry

    processed_conversations = []
    c = 0
    u = 0
    for conversation in conversationSet:
        processed_conversation = processFile(conversation)
        processed_conversations.append(processed_conversation)
        c += 1
        u += len(processed_conversation)
    return processed_conversations


def getBasicStats(convos):
    # Take in conversations and return basic chances
    # of each utterance type

    tag_dict = getDialogueActTags(DIALOGUE_ACT_FILE)
    tags = {}
    for key in tag_dict.keys():
        tags[tag_dict[key]] = 0

    for convo in convos:                # Per Conversation
        for utt in convo.keys():        # Per Utterance
            tags[convo[utt]["dialogue_act"]] += 1

    total_utterances = 0
    for tag in tags:
        total_utterances += tags[tag]

    for tag in tags:
        tags[tag] = tags[tag] / total_utterances
        
    return tags


def outputStats(tags):
    for tag in tags:
        print("\"{0}\": {1},".format(tag, tags[tag]))

def addLevels(n, tags=[]):

    model = {}
    
    if len(tags) == 0:
        tag_dict = getDialogueActTags(DIALOGUE_ACT_FILE)
        for tag in tag_dict.keys():
            tags.append(tag_dict[tag])
            
    if n == 0:
        for tag in tags:
            model[tag] = 0
    else:
        for tag in tags:
            model[tag] = addLevels(n - 1, tags)

    return model


def makeOneGramModel(convos):
    # model[i-n][i-n+1]...[i] = P(i)
    errors = 0
    model = addLevels(1)
    
    # Recall the form convo[uttnum][speaker,dialogue_act,pair_part,words]
    for convo in convos:
        for utt in convo.keys():
            if utt < 2:
                pass
            else:
                a = convo[utt]["dialogue_act"]
                try:
                    b = convo[utt-1]["dialogue_act"]
                    model[b][a] += 1
                except KeyError:
                    errors += 1


    tag_dict = getDialogueActTags(DIALOGUE_ACT_FILE)
    tags = []
    for key in tag_dict.keys():
        tags.append(tag_dict[key])

    predictions = addLevels(0)
        
    for lessOne in tags:
        max_occurrences = 0
        prediction = None
        for tag in tags:
            if model[lessOne][tag] > max_occurrences:
                prediction = tag
                max_occurrences = model[lessOne][tag]
        predictions[lessOne] = prediction

    return predictions


def makeBiGramModel(convos):
    # model[i-n][i-n+1]...[i] = P(i)
    errors = 0
    model = addLevels(2)

    # Recall the form convo[uttnum][speaker,dialogue_act,pair_part,words]
    for convo in convos:
        for utt in convo.keys():
            if utt < 3:
                pass
            else:
                a = convo[utt]["dialogue_act"]
                try:
                    b = convo[utt-1]["dialogue_act"]
                    c = convo[utt-2]["dialogue_act"]
                    model[c][b][a] += 1
                except KeyError:
                    errors += 1

    tag_dict = getDialogueActTags(DIALOGUE_ACT_FILE)
    tags = []
    for key in tag_dict.keys():
        tags.append(tag_dict[key])

    predictions = addLevels(1)
        
    for lessTwo in tags:
        for lessOne in tags:
            max_occurrences = 0
            prediction = None
            for tag in tags:
                if model[lessTwo][lessOne][tag] > max_occurrences:
                    prediction = tag
                    max_occurrences = model[lessTwo][lessOne][tag]
            predictions[lessTwo][lessOne] = prediction

    return predictions
    
                    
def makeTriGramModel(convos):
    # model[i-n][i-n+1]...[i] = P(i)
    errors = 0
    model = addLevels(3)

    # Recall the form convo[uttnum][speaker,dialogue_act,pair_part,words]
    for convo in convos:
        for utt in convo.keys():
            if utt < 4:
                pass
            else:
                a = convo[utt]["dialogue_act"]
                try:
                    b = convo[utt-1]["dialogue_act"]
                    c = convo[utt-2]["dialogue_act"]
                    d = convo[utt-3]["dialogue_act"]
                    model[d][c][b][a] += 1
                except KeyError:
                    errors += 1

    tag_dict = getDialogueActTags(DIALOGUE_ACT_FILE)
    tags = []
    for key in tag_dict.keys():
        tags.append(tag_dict[key])

    predictions = addLevels(2)

    for lessThree in tags:
        for lessTwo in tags:
            for lessOne in tags:
                max_occurrences = 0
                prediction = None
                for tag in tags:
                    if model[lessThree][lessTwo][lessOne][tag] > max_occurrences:
                        prediction = tag
                        max_occurrences = model[lessThree][lessTwo][lessOne][tag]
                predictions[lessThree][lessTwo][lessOne] = prediction

    return predictions


def makeForwardBiModel(convos):
    # model[i-n][i-n+1]...[i] = P(i)
    errors = 0
    model = addLevels(3)

    # Recall the form convo[uttnum][speaker,dialogue_act,pair_part,words]
    for convo in convos:
        for utt in convo.keys():
            if utt < 3:
                pass
            else:
                target = convo[utt]["dialogue_act"]
                try:
                    uptake = convo[utt+1]["dialogue_act"]
                    oneBack = convo[utt-1]["dialogue_act"]
                    twoBack = convo[utt-2]["dialogue_act"]
                    model[uptake][twoBack][oneBack][target] += 1
                except KeyError:
                    errors += 1

    tag_dict = getDialogueActTags(DIALOGUE_ACT_FILE)
    tags = []
    for key in tag_dict.keys():
        tags.append(tag_dict[key])

    predictions = addLevels(2)

    for uptake in tags:
        for lessTwo in tags:
            for lessOne in tags:
                max_occurrences = 0
                prediction = None
                for tag in tags:
                    if model[uptake][lessTwo][lessOne][tag] > max_occurrences:
                        prediction = tag
                        max_occurrences = model[uptake][lessTwo][lessOne][tag]
                predictions[uptake][lessTwo][lessOne] = prediction

    # i = 0
    # j = 0
    # k = 0
    # for uptake in tags:
    #     for lessTwo in tags:
    #         for lessOne in tags:
    #             j += 1
    #             if predictions[uptake][lessTwo][lessOne]:
    #                 i += 1
    #                 if predictions[uptake][lessTwo][lessOne] != "Statement-non-opinion":
    #                     k += 1

    # print("forward model finished with {0} errors".format(errors - len(convos)))
    # print(i / j)
    # print(k / i)
    
    return predictions


def makeForwardTriModel(convos):
    # model[i-n][i-n+1]...[i] = P(i)
    errors = 0
    model = addLevels(4)

    # Recall the form convo[uttnum][speaker,dialogue_act,pair_part,words]
    for convo in convos:
        for utt in convo.keys():
            if utt < 3:
                pass
            else:
                target = convo[utt]["dialogue_act"]
                try:
                    uptake = convo[utt+1]["dialogue_act"]
                    oneBack = convo[utt-1]["dialogue_act"]
                    twoBack = convo[utt-2]["dialogue_act"]
                    threeBack = convo[utt-3]["dialogue_act"]
                    model[uptake][threeBack][twoBack][oneBack][target] += 1
                except KeyError:
                    errors += 1

    tag_dict = getDialogueActTags(DIALOGUE_ACT_FILE)
    tags = []
    for key in tag_dict.keys():
        tags.append(tag_dict[key])

    predictions = addLevels(3)

    for uptake in tags:
        for lessThree in tags:
            for lessTwo in tags:
                for lessOne in tags:
                    max_occurrences = 0
                    prediction = None
                    for tag in tags:
                        if model[uptake][lessThree][lessTwo][lessOne][tag] > max_occurrences:
                            prediction = tag
                            max_occurrences = model[uptake][lessThree][lessTwo][lessOne][tag]
                    predictions[uptake][lessThree][lessTwo][lessOne] = prediction
    
    return predictions


def testModels():

    trainSet = getListOfFiles("./train")
    convos = processConversations(trainSet)
    oneGram = makeOneGramModel(convos)
    twoGram = makeBiGramModel(convos)
    threeGram = makeTriGramModel(convos)
    uptakeBiGram = makeForwardBiModel(convos)
    uptakeTriGram = makeForwardTriModel(convos)
        
    testSet = getListOfFiles("./test")
    convos = processConversations(testSet)

    total = 0
    uniGramCorrect = 0
    biGramCorrect = 0
    triGramCorrect = 0
    forwardBiGramCorrect = 0
    forwardTriGramCorrect = 0
    errors = 0

    for convo in convos:
        for utt in convo.keys():
            if utt < 4:
                pass
            elif utt > (max(list(convo.keys())) - 1):
                pass
            else:
                true_act = convo[utt]["dialogue_act"]

                try:
                    uptake = convo[utt+1]["dialogue_act"]
                    oneBack = convo[utt-1]["dialogue_act"]
                    twoBack = convo[utt-2]["dialogue_act"]
                    threeBack = convo[utt-3]["dialogue_act"]


                    uniGramPrediction = oneGram[oneBack]
                    if not uniGramPrediction:
                        uniGramPrediction = "Statement-non-opinion"

                    biGramPrediction = twoGram[twoBack][oneBack]
                    if not biGramPrediction:
                        biGramPrediction = oneGram[oneBack]
                    if not biGramPrediction:
                        biGramPrediction = "Statement-non-opinion"

                    triGramPrediction = threeGram[threeBack][twoBack][oneBack]
                    if not triGramPrediction:
                        triGramPrediction = twoGram[twoBack][oneBack]
                    if not triGramPrediction:
                        triGramPrediction = oneGram[oneBack]
                    if not triGramPrediction:
                        triGramPrediction = "Statement-non-opinion"

                    forwardBiGramPrediction = uptakeBiGram[uptake][twoBack][oneBack]
                    if not forwardBiGramPrediction:
                        forwardBiGramPrediction = threeGram[threeBack][twoBack][oneBack]
                    if not forwardBiGramPrediction:
                        forwardBiGramPrediction = twoGram[twoBack][oneBack]
                    if not forwardBiGramPrediction:
                        forwardBiGramPrediction = oneGram[oneBack]
                    if not forwardBiGramPrediction:
                        forwardBiGramPrediction = "Statement-non-opinion"

                    forwardTriGramPrediction = uptakeTriGram[uptake][threeBack][twoBack][oneBack]
                    if not forwardTriGramPrediction:
                        forwardTriGramPrediction = uptakeBiGram[uptake][twoBack][oneBack]
                    if not forwardTriGramPrediction:
                        forwardTriGramPrediction = threeGram[threeBack][twoBack][oneBack]
                    if not forwardTriGramPrediction:
                        forwardTriGramPrediction = twoGram[twoBack][oneBack]
                    if not forwardTriGramPrediction:
                        forwardTriGramPrediction = oneGram[oneBack]
                    if not forwardTriGramPrediction:
                        forwardTriGramPrediction = "Statement-non-opinion"

                    total += 1
                    if uniGramPrediction == true_act:
                        uniGramCorrect += 1
                    if biGramPrediction == true_act:
                        biGramCorrect += 1
                    if triGramPrediction == true_act:
                        triGramCorrect += 1
                    if forwardBiGramPrediction == true_act:
                        forwardBiGramCorrect += 1
                    if forwardTriGramPrediction == true_act:
                        forwardTriGramCorrect += 1
                except KeyError: # Not sure what's going on here
                    errors += 1
                    
    print("finished with {0} errors".format(errors))
    print("\nuniGram: {0} of {1} correct".format(uniGramCorrect, total))
    print(uniGramCorrect / total)
    print("\nbiGram: {0} of {1} correct".format(biGramCorrect, total))
    print(biGramCorrect / total)
    print("\ntriGram: {0} of {1} correct".format(triGramCorrect, total))
    print(triGramCorrect / total)
    print("\nforwardBiGram: {0} of {1} correct".format(forwardBiGramCorrect, total))
    print(forwardBiGramCorrect / total)
    print("\nforwardTriGram: {0} of {1} correct".format(forwardTriGramCorrect, total))
    print(forwardTriGramCorrect / total)


makeTrainAndTest()
testModels()
