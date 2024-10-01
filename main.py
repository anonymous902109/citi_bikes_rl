from matplotlib.patches import FancyArrowPatch
from stable_baselines3 import PPO
from tqdm import tqdm
import numpy as np
import gymnasium as gym
from baselines import random_policy, greedy_policy, noop_policy
from citibikes import CitiBikes
from evaluate import evaluate
import networkx as nx
import matplotlib.pyplot as plt


def print_hi():
    env = CitiBikes()

    baselines = [random_policy,  noop_policy]
    baseline_names = ['RANDOM', 'NOOP']

    for i, policy in enumerate(baselines):
        mean_rew = evaluate(env, policy, n_episodes=100)
        print('{}: Mean reward = {}'.format(baseline_names[i], mean_rew))

    model = PPO('MlpPolicy',
                env,
                verbose=1,
                policy_kwargs={'net_arch': [512, 512]},
                learning_rate=0.0001,
                batch_size=512,
                gamma=0.9)

    # model.learn(total_timesteps=5e5)
    # model.save('ppo_5e5_512.zip')

    model = PPO.load('ppo_5e5_512.zip')

    print(evaluate(env, model.predict))

    n_ep = 10
    from_dict = {(i, j): 0 for i in range(1, 6) for j in range(1, 6)}

    for i in tqdm(range(n_ep)):
        obs, _ = env.reset()
        done = False

        while not done:
            action,_ = model.predict(obs)

            f, to, n = action

            from_dict[(f + 1, to + 1)] += n

            obs, rew, done, trunc, info = env.step(action)

    G = nx.DiGraph()

    sent = {}
    received = {}
    for (u, v), weight in from_dict.items():
        G.add_edge(u, v, weight=weight)
        sent[u] = sent.get(u, 0) + weight  # Total sent from each node
        received[v] = received.get(v, 0) + weight  # Total received by each node

    # Node properties
    max_sent = max(sent.values()) if sent else 1
    max_received = max(received.values()) if received else 1

    # Node size represents the total amount sent (normalized)
    node_sizes = [3000 + (sent.get(node, 0) / max_sent) * 30000 for node in G.nodes()]

    # Node color represents the total amount received (normalized)
    node_colors = [received.get(node, 0) for node in G.nodes()]

    # Define node shape: Circle for senders, Square for receivers
    node_shapes = ['o' if sent.get(node, 0) >= received.get(node, 0) else 'o' for node in G.nodes()]

    # Position nodes using a spring layout
    pos = nx.kamada_kawai_layout(G)

    # Draw nodes with different shapes and colors
    fig, ax = plt.subplots(figsize=(12, 10))

    def draw_edges_with_border(G, pos, ax, edge_widths):
        for (u, v, data) in G.edges(data=True):
            # Draw edges with a connection style
            arrow = FancyArrowPatch(
                pos[u], pos[v],
                connectionstyle=f"arc3,rad=0.1",
                color='grey',
                alpha=0.7,
                linewidth=edge_widths[(u, v)]
            )
            ax.add_patch(arrow)

    for shape in set(node_shapes):
        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=[size for size, s in zip(node_sizes, node_shapes) if s == shape],
            node_color=[color for color, s in zip(node_colors, node_shapes) if s == shape],
            node_shape=shape,
            cmap=plt.cm.Blues,
            alpha=1.0,
            vmin=min(node_colors),
            vmax=max(node_colors)
        )

    max_edge_weight = max(from_dict.values())
    edge_widths = {(u, v): 1 + (G[u][v]['weight'] / max_edge_weight)*10 for u, v in G.edges()}

    # Draw edges with varying thickness based on weight
    # nx.draw_networkx_edges(G, pos, width=edge_widths, arrowstyle='-|>', arrowsize=15, edge_color='grey', alpha=0.7)
    draw_edges_with_border(G, pos, ax, edge_widths)

    # Add labels to nodes
    nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')

    # Add edge labels (sent content)
    # edge_labels = {(u, v): f"{data['weight']}" for u, v, data in G.edges(data=True)}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    # Add a color bar for the node colors
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
    plt.colorbar(sm, label='Total Received Bikes')

    plt.title('Distribution of bikes in CitiBikes stations following the policy $\\pi$')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    print_hi()

