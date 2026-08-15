def calculate_avg_rewards(results):
    reward_sum = 0

    for result in results:
        reward_sum += result.get("return")

    return reward_sum / len(results)