from utils import Mode, get_args, get_eval_args, print_segment, run_epoch


@print_segment()
def evaluate_net(net, loader, device, **_):
    run_epoch(net, loader, device, mode=Mode.Test)


def evaluate(nets, loader, mode, device="cpu"):
    # implement the evaluation function here
    match mode:
        case "valid":
            mode = "validation"
        case "test":
            mode = "testing"
        case _:
            raise ValueError("legal values for mode: 'valid', 'test'")
    for net in nets:
        evaluate_net(net, loader, device, title=f"Evaluating {net.name} on {mode} set:")


def main():
    args = get_args()
    for mode, verbose in zip(("valid", "test"), (True, False)):
        nets, loader, device, _ = get_eval_args(args, mode, verbose)
        evaluate(nets, loader, mode, device)


if __name__ == "__main__":
    main()
