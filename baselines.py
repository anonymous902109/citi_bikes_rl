import heapq
import random

ACTION_SPACE = [[0, 2], [0, 2], [0, 10]]


def random_policy(obs):
    from_station = random.randint(*ACTION_SPACE[0])
    to_station = random.randint(*ACTION_SPACE[1])
    number = random.randint(*ACTION_SPACE[2])

    action = [[from_station, to_station, number]]

    return action, None

def noop_policy(obs):
    return [None], None


def greedy_policy(obs):
    decision_event = obs[0]

    if decision_event == 1:
        # find k target stations with the most empty slots, randomly choose one of them and send as many bikes to
        # it as allowed by the action scope
        top_k_demands = []
        for demand_candidate, available_docks in enumerate(obs[3:8]):
            if demand_candidate == obs[1]:
                continue

            heapq.heappush(top_k_demands, (available_docks, demand_candidate))
            if len(top_k_demands) > 1:
                heapq.heappop(top_k_demands)

        max_reposition, target_station_idx = random.choice(top_k_demands)
        action = [[obs[1], target_station_idx, max_reposition]]
        return action, None

    else:
        # find k source stations with the most bikes, randomly choose one of them and request as many bikes from
        # it as allowed by the action scope
        top_k_supplies = []
        for supply_candidate, available_bikes in enumerate(obs[3:8]):
            if supply_candidate == obs[1]:
                continue

            heapq.heappush(top_k_supplies, (available_bikes, supply_candidate))
            if len(top_k_supplies) > 1:
                heapq.heappop(top_k_supplies)

        max_reposition, source_idx = random.choice(top_k_supplies)
        action = [[source_idx, obs[1], max_reposition]]
        return action, None