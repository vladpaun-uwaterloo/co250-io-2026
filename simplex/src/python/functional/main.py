from parser import parse_args, build_lp
from solver import solve


def main():
    args = parse_args()
    lp = build_lp(args)

    result = solve(lp)

    print(result)


if __name__ == "__main__":
    main()