"""Generate immutable ticket-routing splits with a reproducibility manifest."""
import argparse
from ticket_data import generate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--count', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    generate(args.output, args.count, args.seed)
