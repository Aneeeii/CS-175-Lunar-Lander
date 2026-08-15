def calculate_avg_n_rewards(results, n):
    # gets the last n reward values from a set of results and averages them
    
    if len(results) < n:
        raise IndexError(f'Cannot calculate average for {n} results because results is smaller than {n}.')

    result_set = results[-10, -1]
    reward_sum = 0

    for result in enumerate(result_set):
        reward_sum += result_set.get("return")

    return reward_sum / n