from systemtools.location import *
from systemtools.basics import *
from systemtools.file import *
from systemtools.printer import *
from systemtools.number import *
from datatools.jsonutils import *
from datatools.jsonutils import *
from nlptools.basics import *
from nlptools.tokenizer import *


def makeBuckets\
(
    idTokensCount,
    tokensPerBucket=None,
    maxLabelsPerBucket=None,
    maxVarianceRatio=0.2,
    defaultCountDivider=10,
    maxConsecutiveBadBucketCount=2,
    maxConsecutiveNoChangeCount=2,
    returnEliminatedIds=False,
    logger=None, verbose=False,
):
    worstLabels = []
    eliminatedLabels = set()
    eliminatedIds = dict()
    totalTokens = countTokens(idTokensCount)
    tickedTokensAmount = 0
    pbar = ProgressBar(totalTokens, logger=logger, verbose=verbose, printRatio=0.01)
    if tokensPerBucket is None:
        logWarning("Please provide a tokensPerBucket to prevent the function to choose a default one (usualy 1e6).", logger, verbose=verbose)
        tokensPerBucket = int(totalTokens / defaultCountDivider)
        logWarning("The default tokensPerBucket will be " + str(tokensPerBucket) + " (" + str(totalTokens) + " tokens / " + str(defaultCountDivider) + ")", logger, verbose=verbose)
    buckets = []
    remaining = copy.deepcopy(idTokensCount)
    allocated = dict()
    badBucketCount = 0
    noChangeCount = 0
    previousRemainingCount = None
    firstRemainingTokensCount = countTokens(remaining)
    while len(remaining) > 0:
        remainingTokensCount = countTokens(remaining) if verbose else 1
        log("-" * 20 + "\nRemaining labels: " + str(len(remaining)), logger, verbose=verbose)
        log("Remaining tokens: " + str(remainingTokensCount) + " (" + str(truncateFloat(remainingTokensCount / firstRemainingTokensCount * 100, 2)) + "%)", logger, verbose=verbose)
        log("Allocated labels: " + str(len(allocated)), logger, verbose=verbose)
        bucket = makeBucket(remaining, allocated, tokensPerBucket, maxLabelsPerBucket=maxLabelsPerBucket, logger=logger, verbose=verbose)
        assert bucket is not None and len(bucket) > 0
        isValidBucket = isValidBucketFunct(bucket, maxVarianceRatio=maxVarianceRatio)
        # Finally we transfer choosen ids from remaining to allocated:
        newRemaining = copy.deepcopy(remaining)
        newAllocated = copy.deepcopy(allocated)
        for key in bucket:
            for currentId in bucket[key]:
                if key in newRemaining and currentId in newRemaining[key]:
                    if key not in newAllocated:
                        newAllocated[key] = dict()
                    newAllocated[key][currentId] = newRemaining[key][currentId]
                    del newRemaining[key][currentId]
                    if len(newRemaining[key]) == 0:
                        del newRemaining[key]
        # We check if remaining doesn't change:
        newRemainingCount = countTokens(newRemaining)
        if previousRemainingCount is None:
            noChange = False
        else:
            noChange = newRemainingCount == previousRemainingCount
        if not isValidBucket:
            badBucketCount += 1
            logError("This bucket is not a valid bucket...", logger, verbose=verbose)
        if noChange:
            noChangeCount += 1
        if isValidBucket and not noChange:
            badBucketCount = 0
            noChangeCount = 0
            previousRemainingCount = newRemainingCount
            remaining = newRemaining
            allocated = newAllocated
            buckets.append(bucket)
        elif verbose:
            logError("\n\nThis bucket is not ADDED\n\n", logger)
        # Prunning of remaining for the algo to converge:
        if badBucketCount > maxConsecutiveBadBucketCount or noChangeCount > maxConsecutiveNoChangeCount:
            bucketSums = dict()
            for label in bucket:
                bucketSums[label] = sum([e[1] for e in bucket[label].items()])
            bucketMean = sum([e[1] for e in bucketSums.items()]) / len(bucketSums)
            # The absolute difference with the mean:
            for label in bucketSums:
                bucketSums[label] = abs(bucketSums[label] - bucketMean)
            blackBucketLabels = sorted(bucketSums.items(), key=lambda x: x[1], reverse=True)
            blackBucketLabels = [e[0] for e in blackBucketLabels]
            eliminatedAtLeastOne = False
            # We delete min and max documents from remaining (the worst label):
            for blackBucketLabel in blackBucketLabels:
                if blackBucketLabel in remaining:
                    worstLabels.append(blackBucketLabel)
                    recentWorsts = None
                    # We remove this label:
                    if recentWorsts is not None and set(recentWorsts) == 1:
                        del remaining[blackBucketLabel]
                        eliminatedLabels.add(blackBucketLabel)
                        log("We eliminated " + blackBucketLabel + " from remaining", logger, verbose=verbose)
                    else:
                        minId = None
                        minCount = None
                        maxId = None
                        maxCount = None
                        for currentId, currentCount in remaining[blackBucketLabel].items():
                            if minCount is None or currentCount < minCount:
                                minId = currentId
                                minCount = currentCount
                            if maxCount is None or currentCount > maxCount:
                                maxId = currentId
                                maxCount = currentCount
                        if minId is not None and maxId is not None and minId == maxId:
                            maxId = None
                        if blackBucketLabel not in eliminatedIds:
                            eliminatedIds[blackBucketLabel] = dict()
                        if minId is not None:
                            eliminatedIds[blackBucketLabel][minId] = minCount
                            log("We removed " + str(minId) + " (" + str(minCount) + " tokens) from " + blackBucketLabel + " in remaining", logger, verbose=verbose)
                            del remaining[blackBucketLabel][minId]
                        if maxId is not None:
                            eliminatedIds[blackBucketLabel][maxId] = maxCount
                            log("We removed " + str(maxId) + " (" + str(maxCount) + " tokens) from " + blackBucketLabel + " in remaining", logger, verbose=verbose)
                            del remaining[blackBucketLabel][maxId]
                        if len(remaining[blackBucketLabel]) == 0:
                            del remaining[blackBucketLabel]
                    eliminatedAtLeastOne = True
                    break
            if not eliminatedAtLeastOne:
                log("We didn't eliminated any ids from remaining", logger, verbose=verbose)
    log("eliminatedLabels:\n" + b(eliminatedLabels), logger, verbose=verbose)
    log("eliminatedIds:", logger, verbose=verbose)
    bp(eliminatedIds, 4, logger)
    log("Count of labels in eliminatedIds: " + str(len(eliminatedIds)), logger, verbose=verbose)
    idsCount = 0
    for label, value in eliminatedIds.items():
        idsCount += len(value)
    log("Count of ids in eliminatedIds: " + str(idsCount), logger, verbose=verbose)
    if returnEliminatedIds:
        return eliminatedIds, buckets
    else:
        return buckets



def makeBucket\
(
    remaining, allocated, maxTokensPerBucket,
    maxLabelsPerBucket=None, allowAllocatedPriority=False,
    defaultMaxLabelsPerBucketDivider=4,
    logger=None, verbose=True,
):
    # We auto calculate minTokensPerBucket and maxLabelsPerBucket:
    if maxLabelsPerBucket is None:
        maxLabelsPerBucket = int(len(set(list(remaining.keys()) + list(allocated.keys()))) / defaultMaxLabelsPerBucketDivider)
    if maxLabelsPerBucket < 4:
        logError("maxLabelsPerBucket is lower than 4", logger, verbose=verbose)
        maxLabelsPerBucket = 4
    # If we have no remaining ids to push in the bucket:
    if len(remaining) == 0:
        return None
    else:
        # We take labels with priority on remaining:
        choosenLabels = set()
        for label in getAscLabels(remaining):
            choosenLabels.add(label)
            if len(choosenLabels) >= maxLabelsPerBucket:
                break
        # And then we add already allocated labels until we reach maxLabelsPerBucket,
        # we take labels that have a lot of tokens in priority:
        for key in getDescLabels(allocated):
            choosenLabels.add(key)
            if len(choosenLabels) >= maxLabelsPerBucket:
                break
        # Here we want to know how many tokens we can get at least for these labels:
        minTokensBoth = None
        for key in choosenLabels:
            c = 0
            try:
                c += countLabelTokens(remaining[key])
            except: pass
            try:
                c += countLabelTokens(allocated[key])
            except: pass
            if minTokensBoth is None or c < minTokensBoth:
                minTokensBoth = c
        # We calculate the number of tokens:
        # log("minTokensBoth: " + str(minTokensBoth), logger)
        totalTokensCount = minTokensBoth * len(choosenLabels)
        if totalTokensCount > maxTokensPerBucket:
            minTokensBoth = int(maxTokensPerBucket / len(choosenLabels))
            # log("minTokensBoth after editing: " + str(minTokensBoth), logger)
        # Then we get all ids with the priority on remaining:
        bucket = dict()
        for key in choosenLabels:
            currentSum = 0
            bucket[key] = dict()
            if key in remaining:
                bucket[key] = getOptimalIdsSet(remaining[key], minTokensBoth)
            if key in allocated:
                bucket[key] = getOptimalIdsSet(allocated[key], minTokensBoth, selectedIds=bucket[key])
            currentSum = countLabelTokens(bucket[key])
            # For security, in case the selection is bad:
            if allowAllocatedPriority:
                if (currentSum > minTokensBoth + minTokensBoth * 0.05) or \
                (currentSum < minTokensBoth - minTokensBoth * 0.05):
                    log("Trying to pick in allocated in priority", logger, verbose=verbose)
                    # We delete a large of the selection (due to remaining):
                    toDeleteCandidates = shuffle(list(bucket[key].keys()))
                    toDeleteIndex = int(len(toDeleteCandidates) * 0.8)
                    if toDeleteIndex == 0:
                        toDeleteIndex = 1
                    toDelete = toDeleteCandidates[:toDeleteIndex]
                    for current in toDelete:
                        del bucket[key][current]
                    # And now we get ids from allocated in priority instead of remaining:
                    if key in allocated:
                        bucket[key] = getOptimalIdsSet(allocated[key], minTokensBoth, selectedIds=bucket[key])
                    if key in remaining:
                        bucket[key] = getOptimalIdsSet(remaining[key], minTokensBoth, selectedIds=bucket[key])
        return bucket



def isValidBucketFunct(bucket, idTokensCount=None, maxVarianceRatio=0.2):
    minTokensCount = None
    maxTokensCount = None
    totalTokensCount = 0
    totalIdsCount = 0
    for label in bucket:
        currentSum = 0
        for currentId in bucket[label]:
            currentSum += bucket[label][currentId]
        if minTokensCount is None or currentSum < minTokensCount:
            minTokensCount = currentSum
        if maxTokensCount is None or currentSum > maxTokensCount:
            maxTokensCount = currentSum
        totalTokensCount += currentSum
    meanTokensCount = int(totalTokensCount / len(bucket))
    isValidBucket = minTokensCount > meanTokensCount - maxVarianceRatio * meanTokensCount
    isValidBucket = isValidBucket and (maxTokensCount < meanTokensCount + maxVarianceRatio * meanTokensCount)
    return isValidBucket


def getOptimalIdsSet\
(
    availableIds,
    minTokens,
    selectedIds=None,
    maxIterations=100,
    maxVariationRatio=0.01,
    toDeleteRatio=1.0,
):
    if selectedIds is None:
        selectedIds = dict()
    else:
        selectedIds = copy.deepcopy(selectedIds)
    idsHistory = []
    alreadySelectedLabels = set(selectedIds.keys())
    for i in range(maxIterations):
        currentSum = sum([e[1] for e in selectedIds.items()])
        for currentId in shuffle(list(availableIds.keys())):
            if currentId not in selectedIds:
                if currentSum + availableIds[currentId] < minTokens:
                    selectedIds[currentId] = availableIds[currentId]
                    currentSum += availableIds[currentId]
        idsHistory.append((abs(currentSum - minTokens), copy.deepcopy(selectedIds)))
        if (currentSum > minTokens + minTokens * maxVariationRatio) or \
        (currentSum < minTokens - minTokens * maxVariationRatio):
            toDeleteCandidates = shuffle(list(setSubstract(set(selectedIds.keys()), alreadySelectedLabels)))
            toDeleteIndex = int(len(toDeleteCandidates) * toDeleteRatio)
            if toDeleteIndex == 0:
                toDeleteIndex = 1
            toDelete = toDeleteCandidates[:toDeleteIndex]
            for current in toDelete:
                del selectedIds[current]
        else:
            break
    idsHistory = sorted(idsHistory, key=lambda x: x[0], reverse=False)
    return idsHistory[0][1]

def bucketStats(bucket, idTokensCount=None, remaining=None, allocated=None, tab="\t", maxVarianceRatio=0.2, logger=None, verbose=True):
    # Counts in the bucket:
    text = ""
    minTokensCount = None
    minTokensLabel = None
    maxTokensCount = None
    maxTokensLabel = None
    totalTokensCount = 0
    minIdsCount = None
    minIdsLabel = None
    maxIdsCount = None
    maxIdsLabel = None
    totalIdsCount = 0
    for label in bucket:
        currentSum = 0
        currentAmountOfIds = 0
        for currentId in bucket[label]:
            currentSum += bucket[label][currentId]
            currentAmountOfIds += 1
            if remaining is not None:
                assert (label not in remaining) or (currentId not in remaining[label])
            if allocated is not None:
                assert (label in allocated) and (currentId in allocated[label])
            if allocated is not None and idTokensCount is not None:
                assert allocated[label][currentId] == idTokensCount[label][currentId]
        if minTokensCount is None or currentSum < minTokensCount:
            minTokensCount = currentSum
            minTokensLabel = label
            minIdsCount = currentAmountOfIds
        if maxTokensCount is None or currentSum > maxTokensCount:
            maxTokensCount = currentSum
            maxTokensLabel = label
            maxIdsCount = currentAmountOfIds
        totalTokensCount += currentSum
        totalIdsCount += currentAmountOfIds
    meanTokensCount = int(totalTokensCount / len(bucket))
    meanIdsCount = int(totalIdsCount / len(bucket))
    text += "Labels:" + "\n"
    text += tab + "min: " + str(minTokensLabel) + "\n"
    text += tab + "max: " + str(maxTokensLabel) + "\n"
    text += "Tokens count:" + "\n"
    text += tab + "min: " + str(minTokensCount) + "\n"
    text += tab + "max: " + str(maxTokensCount) + "\n"
    text += tab + "mean: " + str(int(meanTokensCount)) + "\n"
    if totalTokensCount / 1e6 <  0.0:
        text += tab + "total: " + str(totalTokensCount) + "\n"
    else:
        text += tab + "total: " + str(truncateFloat(totalTokensCount / 1e6, 2)) + "M" + "\n"
    text += "Ids count:" + "\n"
    text += tab + "min: " + str(minIdsCount) + "\n"
    text += tab + "max: " + str(maxIdsCount) + "\n"
    text += tab + "mean: " + str(int(meanIdsCount)) + "\n"
    text += tab + "total: " + str(totalIdsCount) + "\n"
    # Remaining count:
    if remaining is not None:
        text += "Remaining count:" + "\n"
        text += tab + str(countTokens(remaining)) + " tokens" + "\n"
    # Allocated count:
    if allocated is not None:
        text += "Allocated count:" + "\n"
        text += tab + str(countTokens(allocated)) + " tokens" + "\n"
    # Assertions:
    isValidBucket = minTokensCount > meanTokensCount - maxVarianceRatio * meanTokensCount
    isValidBucket = isValidBucket and (maxTokensCount < meanTokensCount + maxVarianceRatio * meanTokensCount)
    text += "Is it a valid bucket?" + "\n"
    text += tab + ("YES" if isValidBucket else "NO") + "\n"
    log(text + "\n" * 2, logger=logger, verbose=verbose)
    return isValidBucket


# Misc functions:
def getAscLabels(idTokensCount):
    return getPriorityLabels(idTokensCount, desc=False)
def getDescLabels(idTokensCount):
    return getPriorityLabels(idTokensCount, desc=True)
def getPriorityLabels(idTokensCount, desc=False):
    aaa = dict()
    for key in idTokensCount:
        aaa[key] = sum([e[1] for e in idTokensCount[key].items()])
    return [e[0] for e in sorted(aaa.items(), key=lambda x: x[1], reverse=desc)]
def getMinLabel(bucket):
    minTokensCount = None
    minTokensLabel = None
    for label, ids in bucket.items():
        currentCount = countLabelTokens(ids)
        if minTokensCount is None or currentCount < minTokensCount:
            minTokensCount = currentCount
            minTokensLabel = label
    return minTokensLabel
def countTokens(data):
    totalTokens = 0
    for key in data:
        for id in data[key]:
            totalTokens += data[key][id]
    return totalTokens
def countLabelTokens(d):
    return sum([e[1] for e in d.items()])