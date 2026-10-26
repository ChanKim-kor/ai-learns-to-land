import csv
from pathlib import Path

import matplotlib.pyplot as plt

if __name__ == "__main__":
    input_path = (
        Path(__file__).resolve().parents[1]
        / "runs"
        / "evaluation_history.csv"
    )

    with input_path.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        evaluations = list(csv.DictReader(file))

    episodes = [
        int(row["episode"]) for row in evaluations
    ]
    rewards = [
        float(row["final_reward"]) for row in evaluations
    ]


    fig, ax = plt.subplots()

    ax.plot(episodes, rewards, color="gray", alpha=0.5)

    result_styles = [
        ("landed", "green", "Landed"),
        ("crashed", "red", "Crashed"),
        ("in_flight", "orange", "Time limit"),
    ]

    for status, color, label in result_styles:
        matching_rows = [
            row for row in evaluations
            if row["status"] == status
        ]

        if not matching_rows:
            continue

        ax.scatter(
            [int(row["episode"]) for row in matching_rows],
            [float(row["final_reward"]) for row in matching_rows],
            color=color,
            label=label,
            zorder=3,
        )

    ax.legend()

    ax.set_title("Evaluation Reward During Training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")

    plt.show()