"""Dependency-light runner for the Bridge-Inside CPU regressions."""
import test_bridge_inside_sft as tests


def main():
    names = sorted(name for name in dir(tests) if name.startswith("test_"))
    for name in names:
        getattr(tests, name)()
        print(f"PASS {name}")
    print(f"CPU_TESTS_PASS={len(names)}")


if __name__ == "__main__":
    main()
