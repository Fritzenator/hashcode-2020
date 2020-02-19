import numpy as np
import numba


@numba.jit(nopython=True)
def knapsack(values, weights, max_weight):
    """
    Returns tuple (total summed value, chosen items)
    """
    t = np.zeros((len(values), max_weight + 1), dtype=np.float64)

    # Fill-in the value table using the recurrence formula
    for i in range(len(values)):
        for w in range(max_weight + 1):
            # If weight is above the sub-problem max weight, we can only not-choose
            # Thus we take the value of the previous best solution for sub-problem without
            # this item
            if weights[i] > w:
                # Check bounds. For sub-problem with 0 items we must have 0 value
                if i > 0:
                    t[i, w] = t[i - 1, w]
                else:
                    t[i, w] = 0
            # If we can choose, see if the choice yields a higher value
            else:
                # If chosen the value is value of previous best with 
                # the weight subtracted plus the value of this item
                value_if_chosen = t[i - 1, w - weights[i]] + values[i]
                # Previous row, current column is the value of the sub-problem
                # for same max weight but without this item (item not chosen)
                value_if_ignored = t[i - 1, w]

                t[i, w] = max(value_if_chosen, value_if_ignored)
    # print(t)

    # Backtrack from the value table to find the items chosen
    chosen_items = []
    w = max_weight

    for i in range(len(values) - 1, -1, -1):
        # If we are in first row (sub-problem with only one item)
        # we need to check if the value is greater than 0
        # if it is, it means the item 0 was chosen
        if i == 0 and t[i, w] != 0:
            chosen_items.append(0)
        # If we have higher value for choosing than for non-choosing
        # It means we have chosen i
        elif t[i, w] > t[i - 1, w]:
            chosen_items.append(i)
            w = w - weights[i]
        # Otherwise do nothing as i was not chosen

    return t[-1, -1], chosen_items


# Find minimal number of coins needed using DP
# Backtracking to find the actual coins is not yet implemented
@numba.jit(nopython=True)
def change_minimal_coins(money, coins):
    array = [money + 1 for _ in range(money + 1)]
    array[0] = 0

    for i in range(1, money + 1):
        array[i] = 1 + min([array[i - c] for c in coins if i - c >= 0])

    # print(array)

    return array[-1]
