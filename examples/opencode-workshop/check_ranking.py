"""Independent acceptance check. The unmodified exercise fails the tie case."""

from ranking import winner


def main() -> None:
    assert winner([("alpha", 2), ("zeta", 5)]) == "zeta", "Highest score must win."
    assert winner([("zeta", 5), ("alpha", 5)]) == "alpha", "Ties must break by ascending label."
    assert winner([("alpha", 5), ("zeta", 5)]) == "alpha", "The result must be order independent."
    print("Ranking checks passed.")


if __name__ == "__main__":
    main()
